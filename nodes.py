import ctypes
import gc
import os
import re
import shutil
import subprocess
import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
    _COMFY_OUTPUT = folder_paths.get_output_directory()
except Exception:
    _COMFY_OUTPUT = "/workspace/ComfyUI/output"


TEMP_ROOT = "/workspace/temp_chunks"
CGROUP_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
CGROUP_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

# glibc malloc_trim — returns freed heap pages to the OS so RSS actually drops.
# Python's allocator retains arenas after del/gc; without this, top/nvidia-smi
# still show the memory as held even though Python released it.
try:
    _LIBC = ctypes.CDLL("libc.so.6")
    _LIBC.malloc_trim.argtypes = [ctypes.c_size_t]
    _LIBC.malloc_trim.restype = ctypes.c_int
except Exception:
    _LIBC = None


def _read_cgroup(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _ram_str():
    used = _read_cgroup(CGROUP_USAGE)
    lim = _read_cgroup(CGROUP_LIMIT)
    if used is None:
        return "ram unavailable"
    s = f"ram={used / 1024**3:.2f}GB"
    if lim and lim < (1 << 62):
        s += f"/{lim / 1024**3:.2f}GB"
    # also process RSS for comparison
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    s += f" rss={rss_kb / 1024**2:.2f}GB"
                    break
    except Exception:
        pass
    return s


def _vram_str():
    if not torch.cuda.is_available():
        return "cuda unavailable"
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    return f"vram alloc={a:.2f}GB reserved={r:.2f}GB"


def _purge(trim_ram=True):
    gc.collect()
    gc.collect()  # second pass catches cycles freed by the first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if trim_ram and _LIBC is not None:
        _LIBC.malloc_trim(0)


class WanVideoChunkWriter:
    """
    Writes an IMAGE batch (one chunk) to disk as a PNG sequence, then frees it.
    Returns only the trailing `overlap_frames` for blending into the next chunk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "session_id": ("STRING", {"default": "run01"}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "overlap_frames": ("INT", {"default": 0, "min": 0, "max": 256}),
                "purge_vram": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("overlap_tail", "session_id", "next_chunk_index")
    FUNCTION = "write"
    CATEGORY = "WanChunkIO"
    OUTPUT_NODE = True

    def write(self, images, session_id, chunk_index, overlap_frames, purge_vram):
        ram_before = _ram_str()
        vram_before = _vram_str()

        out_dir = os.path.join(TEMP_ROOT, session_id, f"chunk_{chunk_index:04d}")
        os.makedirs(out_dir, exist_ok=True)

        n, h, w, c = images.shape

        # Pull the overlap tail FIRST (small slice, cheap clone on CPU)
        # so we can drop the big tensor right after the write loop.
        if overlap_frames > 0 and n >= overlap_frames:
            tail = images[-overlap_frames:].detach().clone().cpu().contiguous()
        else:
            tail = torch.zeros((1, h, w, c), dtype=torch.float32)

        # Stream to disk one frame at a time. Avoids allocating a second
        # full-chunk numpy copy in RAM during conversion.
        images_cpu = images.detach().cpu()
        for i in range(n):
            frame = (
                images_cpu[i].clamp(0, 1).mul(255).to(torch.uint8).numpy()
            )
            Image.fromarray(frame).save(
                os.path.join(out_dir, f"{i:06d}.png"), compress_level=1
            )
            del frame

        del images, images_cpu
        if purge_vram:
            _purge(trim_ram=True)

        ram_after = _ram_str()
        vram_after = _vram_str()
        print(
            f"[ChunkWriter] session={session_id} chunk={chunk_index} "
            f"wrote {n} frames {w}x{h} -> {out_dir}\n"
            f"  before: {ram_before} | {vram_before}\n"
            f"  after:  {ram_after} | {vram_after}"
        )

        return (tail, session_id, chunk_index + 1)


class WanVideoChunkAssembler:
    """
    Concatenates all chunks for a session into a single mp4 via ffmpeg,
    WITHOUT loading frames into RAM. Optionally drops the first
    `skip_overlap_frames` of every chunk after the first, to remove
    the overlap region you used for blending.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "run01"}),
                "fps": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "video/vace"}),
                "overlap_frames": ("INT", {"default": 17, "min": 0, "max": 256, "tooltip": "Number of overlap frames between consecutive chunks. Must match what your loop generated. For WAN VAE, use 4N+1 (5, 9, 13, 17, 21...)."}),
                "seam_mode": (
                    ["blend", "hard_cut_keep_earlier", "hard_cut_keep_later", "none"],
                    {"default": "blend", "tooltip": "blend = linear crossfade across overlap region (recommended). hard_cut_* = pick one chunk's frames at the seam. none = concatenate raw, overlap appears duplicated."},
                ),
                "crf": ("INT", {"default": 17, "min": 0, "max": 51}),
                "delete_chunks_after": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Wire the writer's next_chunk_index into this so the DAG
                # forces the assembler to run AFTER the writer in the same graph.
                "wait_for": ("INT", {"forceInput": True}),
                "audio": ("AUDIO",),
                "audio_offset_sec": ("FLOAT", {"default": 0.0, "min": -600.0, "max": 600.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "assemble"
    CATEGORY = "WanChunkIO"
    OUTPUT_NODE = True

    def assemble(
        self,
        session_id,
        fps,
        filename_prefix,
        overlap_frames,
        crf,
        delete_chunks_after,
        wait_for=None,
        audio=None,
        audio_offset_sec=0.0,
        seam_mode="blend",
    ):
        session_dir = os.path.join(TEMP_ROOT, session_id)
        if not os.path.isdir(session_dir):
            raise RuntimeError(f"No session dir: {session_dir}")

        chunk_dirs = sorted(
            d for d in os.listdir(session_dir)
            if d.startswith("chunk_") and os.path.isdir(os.path.join(session_dir, d))
        )
        if not chunk_dirs:
            raise RuntimeError(f"No chunks in {session_dir}")

        # Build a flat dir of sequentially-numbered PNGs (symlinks for raw frames,
        # real files for blended seam frames) and feed it to ffmpeg image2.
        flat_dir = os.path.join(session_dir, "_flat")
        blend_dir = os.path.join(session_dir, "_blend")
        for d in (flat_dir, blend_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)

        # Cache each chunk's sorted file list once.
        chunk_paths = [os.path.join(session_dir, cd) for cd in chunk_dirs]
        chunk_files = [
            sorted(f for f in os.listdir(p) if f.endswith(".png"))
            for p in chunk_paths
        ]
        last_idx = len(chunk_dirs) - 1

        def _link(src, total):
            os.symlink(src, os.path.join(flat_dir, f"{total:08d}.png"))

        def _blend_pair(a_path, b_path, alpha, out_path):
            a = np.asarray(Image.open(a_path), dtype=np.float32)
            b = np.asarray(Image.open(b_path), dtype=np.float32)
            out = (1.0 - alpha) * a + alpha * b
            Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
                out_path, compress_level=1
            )

        total = 0
        ov = overlap_frames if seam_mode != "none" else 0

        for i, files in enumerate(chunk_files):
            cd_path = chunk_paths[i]
            n = len(files)

            if seam_mode == "blend" and ov > 0 and i > 0:
                # write blended seam between prev chunk's tail and this chunk's head
                prev_path = chunk_paths[i - 1]
                prev_files = chunk_files[i - 1]
                for j in range(ov):
                    alpha = (j + 1) / (ov + 1) if ov > 1 else 0.5
                    a = os.path.join(prev_path, prev_files[-ov + j])
                    b = os.path.join(cd_path, files[j])
                    out = os.path.join(blend_dir, f"seam_{i:04d}_{j:04d}.png")
                    _blend_pair(a, b, alpha, out)
                    _link(out, total)
                    total += 1

            # decide which slice of this chunk's raw frames to emit
            if seam_mode == "blend":
                start = ov if i > 0 else 0
                end = n - ov if i < last_idx else n
            elif seam_mode == "hard_cut_keep_earlier":
                start = 0
                end = n - ov if i < last_idx else n
            elif seam_mode == "hard_cut_keep_later":
                start = ov if i > 0 else 0
                end = n
            else:  # none
                start, end = 0, n

            for j in range(start, end):
                _link(os.path.join(cd_path, files[j]), total)
                total += 1

        if total == 0:
            raise RuntimeError("No frames after seam processing")
        print(f"[ChunkAssembler] seam_mode={seam_mode} overlap={ov} -> {total} frames")

        # Resolve output via ComfyUI's standard helper so behavior matches
        # SaveImage / VHS_VideoCombine: prefix becomes a path under output dir,
        # with auto-incrementing 5-digit counter.
        try:
            full_output_folder, filename, counter, subfolder, _ = (
                folder_paths.get_save_image_path(filename_prefix, _COMFY_OUTPUT)
            )
        except Exception:
            # Fallback if folder_paths isn't importable
            full_output_folder = os.path.join(_COMFY_OUTPUT, os.path.dirname(filename_prefix))
            os.makedirs(full_output_folder, exist_ok=True)
            filename = os.path.basename(filename_prefix) or "video"
            existing = [f for f in os.listdir(full_output_folder) if f.startswith(filename + "_")]
            counter = len(existing) + 1
            subfolder = os.path.dirname(filename_prefix)

        out_filename = f"{filename}_{counter:05d}.mp4"
        output_path = os.path.join(full_output_folder, out_filename)
        os.makedirs(full_output_folder, exist_ok=True)

        # Optional: dump AUDIO dict to a temp wav for muxing
        audio_path = None
        if audio is not None:
            try:
                wav = audio["waveform"]
                sr = int(audio["sample_rate"])
                # waveform shape can be [B, C, S] or [C, S]; normalize to [C, S]
                if wav.dim() == 3:
                    wav = wav[0]
                wav = wav.detach().cpu().to(torch.float32).clamp(-1.0, 1.0)
                # interleave channels for soundfile-free wav write via ffmpeg pipe
                # easier: use scipy/soundfile if present, else ffmpeg from raw f32le
                audio_path = os.path.join(session_dir, "_audio.wav")
                channels, samples = wav.shape
                raw = wav.t().contiguous().numpy().tobytes()
                ffmpeg_in = subprocess.Popen(
                    [
                        "ffmpeg", "-y",
                        "-f", "f32le",
                        "-ar", str(sr),
                        "-ac", str(channels),
                        "-i", "pipe:0",
                        audio_path,
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                ffmpeg_in.communicate(raw)
                if ffmpeg_in.returncode != 0:
                    print("[ChunkAssembler] audio wav write failed; muxing video only")
                    audio_path = None
                else:
                    print(f"[ChunkAssembler] audio: {channels}ch @ {sr}Hz, {samples / sr:.2f}s")
            except Exception as e:
                print(f"[ChunkAssembler] audio prep failed ({e}); muxing video only")
                audio_path = None

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(flat_dir, "%08d.png"),
        ]
        if audio_path is not None:
            cmd += ["-itsoffset", str(audio_offset_sec), "-i", audio_path]
        cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", str(crf), "-preset", "medium",
            "-r", str(fps),
        ]
        if audio_path is not None:
            cmd += [
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
            ]
        cmd += [output_path]

        print(f"[ChunkAssembler] {total} frames -> {output_path}"
              + (" (with audio)" if audio_path else ""))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("[ChunkAssembler] ffmpeg stderr:\n" + result.stderr[-3000:])
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode})")

        if delete_chunks_after:
            shutil.rmtree(session_dir)
            print(f"[ChunkAssembler] deleted {session_dir}")

        ui = {"gifs": [{
            "filename": out_filename,
            "subfolder": subfolder,
            "type": "output",
            "format": "video/mp4",
            "frame_rate": float(fps),
            "fullpath": output_path,
        }]}
        print(f"[ChunkAssembler] preview at {output_path}")

        return {"ui": ui, "result": (output_path,)}


class WanVideoChunkSessionAuto:
    """
    Auto-increments session id (run1, run2, run3, ...) by scanning TEMP_ROOT.
    Set `reset` to True to wipe and reuse, otherwise the next free slot is picked.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "run"}),
                "create": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("session_id", "session_index")
    FUNCTION = "pick"
    CATEGORY = "WanChunkIO"

    @classmethod
    def IS_CHANGED(cls, prefix, create):
        # Force re-evaluation every queue so we get a fresh next-slot.
        return float("nan")

    def pick(self, prefix, create):
        os.makedirs(TEMP_ROOT, exist_ok=True)
        pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        used = []
        for d in os.listdir(TEMP_ROOT):
            m = pat.match(d)
            if m and os.path.isdir(os.path.join(TEMP_ROOT, d)):
                used.append(int(m.group(1)))
        next_idx = (max(used) + 1) if used else 1
        sid = f"{prefix}{next_idx}"
        if create:
            os.makedirs(os.path.join(TEMP_ROOT, sid), exist_ok=True)
        print(f"[ChunkSessionAuto] picked {sid}")
        return (sid, next_idx)


class WanVideoChunkSessionReset:
    """Wipes a session directory before a new run, so stale chunks don't leak in."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "run01"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("session_id",)
    FUNCTION = "reset"
    CATEGORY = "WanChunkIO"

    def reset(self, session_id):
        p = os.path.join(TEMP_ROOT, session_id)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f"[ChunkSessionReset] cleared {p}")
        os.makedirs(p, exist_ok=True)
        return (session_id,)


NODE_CLASS_MAPPINGS = {
    "WanVideoChunkWriter": WanVideoChunkWriter,
    "WanVideoChunkAssembler": WanVideoChunkAssembler,
    "WanVideoChunkSessionReset": WanVideoChunkSessionReset,
    "WanVideoChunkSessionAuto": WanVideoChunkSessionAuto,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoChunkWriter": "Wan Video Chunk Writer",
    "WanVideoChunkAssembler": "Wan Video Chunk Assembler (ffmpeg)",
    "WanVideoChunkSessionReset": "Wan Video Chunk Session Reset",
    "WanVideoChunkSessionAuto": "Wan Video Chunk Session Auto-Increment",
}
