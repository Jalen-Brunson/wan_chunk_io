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
# cgroup v2 first, v1 fallback
_CGROUP_V2_USAGE = "/sys/fs/cgroup/memory.current"
_CGROUP_V2_LIMIT = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
if os.path.exists(_CGROUP_V2_USAGE):
    CGROUP_USAGE, CGROUP_LIMIT = _CGROUP_V2_USAGE, _CGROUP_V2_LIMIT
else:
    CGROUP_USAGE, CGROUP_LIMIT = _CGROUP_V1_USAGE, _CGROUP_V1_LIMIT

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
            v = f.read().strip()
        if v == "max":  # cgroup v2 "unlimited"
            return None
        return int(v)
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

    @classmethod
    def VALIDATE_INPUTS(cls, audio_offset_sec=0.0, fps=16.0, **kwargs):
        # Allow null/empty/stale widget values; assemble() coerces to defaults.
        return True

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
        # Tolerate empty-string / stale widget values from older workflow JSONs
        try:
            audio_offset_sec = float(audio_offset_sec) if audio_offset_sec != "" else 0.0
        except (TypeError, ValueError):
            audio_offset_sec = 0.0
        try:
            fps = float(fps) if fps != "" else 16.0
        except (TypeError, ValueError):
            fps = 16.0
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
            full_output_folder, filename, _counter_unused, subfolder, _ = (
                folder_paths.get_save_image_path(filename_prefix, _COMFY_OUTPUT)
            )
        except Exception:
            # Fallback if folder_paths isn't importable
            full_output_folder = os.path.join(_COMFY_OUTPUT, os.path.dirname(filename_prefix))
            filename = os.path.basename(filename_prefix) or "video"
            subfolder = os.path.dirname(filename_prefix)
        os.makedirs(full_output_folder, exist_ok=True)

        # get_save_image_path's counter is based on an IMAGE pattern
        # (foo_00001_.png with trailing underscore) and returns 1 for .mp4
        # outputs — always overwriting. Scan existing mp4s matching our
        # pattern and take max+1 instead.
        mp4_re = re.compile(rf"^{re.escape(filename)}_(\d+)\.mp4$")
        existing_nums = [
            int(m.group(1))
            for f in os.listdir(full_output_folder)
            if (m := mp4_re.match(f))
        ]
        counter = (max(existing_nums) + 1) if existing_nums else 1

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


class WanVideoChunkCount:
    """
    Returns the number of chunks in a session. Use as the loop-count input
    for whatever iterator node you're driving post-processing with.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"session_id": ("STRING", {"default": "run01"})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("chunk_count",)
    FUNCTION = "count"
    CATEGORY = "WanChunkIO"

    @classmethod
    def IS_CHANGED(cls, session_id):
        return float("nan")

    def count(self, session_id):
        p = os.path.join(TEMP_ROOT, session_id)
        if not os.path.isdir(p):
            return (0,)
        n = sum(
            1 for d in os.listdir(p)
            if d.startswith("chunk_") and os.path.isdir(os.path.join(p, d))
        )
        print(f"[ChunkCount] session={session_id} has {n} chunks")
        return (n,)


class WanVideoChunkLoader:
    """
    Loads one chunk's PNG sequence from disk back into an IMAGE batch,
    for post-processing (upscale / interpolate / etc.) before assembly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "run01"}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "frame_count", "chunk_count_in_session")
    FUNCTION = "load"
    CATEGORY = "WanChunkIO"

    @classmethod
    def IS_CHANGED(cls, session_id, chunk_index):
        p = os.path.join(TEMP_ROOT, session_id, f"chunk_{chunk_index:04d}")
        if not os.path.isdir(p):
            return float("nan")
        mt = max((os.path.getmtime(os.path.join(p, f)) for f in os.listdir(p)), default=0)
        return f"{p}:{mt}"

    def load(self, session_id, chunk_index):
        session_dir = os.path.join(TEMP_ROOT, session_id)
        chunk_dir = os.path.join(session_dir, f"chunk_{chunk_index:04d}")
        if not os.path.isdir(chunk_dir):
            raise RuntimeError(f"Chunk dir missing: {chunk_dir}")

        files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".png"))
        if not files:
            raise RuntimeError(f"No PNGs in {chunk_dir}")

        frames = []
        for f in files:
            arr = np.asarray(Image.open(os.path.join(chunk_dir, f)).convert("RGB"),
                             dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr))
        images = torch.stack(frames, dim=0)  # [N, H, W, 3]

        all_chunks = sum(
            1 for d in os.listdir(session_dir)
            if d.startswith("chunk_") and os.path.isdir(os.path.join(session_dir, d))
        )
        print(f"[ChunkLoader] {session_id}/chunk_{chunk_index:04d}: {len(files)} frames")
        return (images, len(files), all_chunks)


class WanVideoChunkSaveAs:
    """
    Writes an IMAGE batch as a chunk under a DIFFERENT session id,
    so post-processed chunks don't overwrite the originals.
    Use this after upscaling/interpolating to stage the assembler input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_session_id": ("STRING", {"default": "run01_hi"}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "purge_ram": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("target_session_id", "next_chunk_index")
    FUNCTION = "save"
    CATEGORY = "WanChunkIO"
    OUTPUT_NODE = True

    def save(self, images, target_session_id, chunk_index, purge_ram):
        before = _ram_str()
        out_dir = os.path.join(TEMP_ROOT, target_session_id, f"chunk_{chunk_index:04d}")
        os.makedirs(out_dir, exist_ok=True)
        n, h, w, _ = images.shape
        imgs_cpu = images.detach().cpu()
        for i in range(n):
            frame = imgs_cpu[i].clamp(0, 1).mul(255).to(torch.uint8).numpy()
            Image.fromarray(frame).save(
                os.path.join(out_dir, f"{i:06d}.png"), compress_level=1
            )
        del images, imgs_cpu
        if purge_ram:
            _purge(trim_ram=True)
        print(f"[ChunkSaveAs] wrote {n} frames {w}x{h} -> {out_dir}\n"
              f"  before: {before}\n  after:  {_ram_str()}")
        return (target_session_id, chunk_index + 1)


class WanMemoryPurge:
    """
    Hammer node: runs gc.collect(), torch.cuda.empty_cache(), malloc_trim(0),
    and optionally soft-unloads ComfyUI's currently-tracked models.
    Place inside your loop to fight allocator fragmentation and glibc heap retention.
    Has a passthrough IMAGE/INT so it can sit inline in the graph.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("INT", {"default": 0, "forceInput": True}),
                "unload_models": ("BOOLEAN", {"default": False, "tooltip": "Offload ComfyUI-tracked models from VRAM. Usually leave OFF — helps VRAM, hurts RAM."}),
                "aggressive": ("BOOLEAN", {"default": True, "tooltip": "Double-pass gc + malloc_trim"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("trigger_passthrough",)
    FUNCTION = "purge"
    CATEGORY = "WanChunkIO"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def purge(self, trigger, unload_models, aggressive):
        before = _ram_str()
        if unload_models:
            try:
                import comfy.model_management as mm
                mm.unload_all_models()
                mm.soft_empty_cache()
            except Exception as e:
                print(f"[MemoryPurge] model unload skipped: {e}")
        _purge(trim_ram=True)
        if aggressive:
            _purge(trim_ram=True)
        after = _ram_str()
        print(f"[MemoryPurge] before: {before}\n           after:  {after}")
        return (trigger,)


_LEAK_PREV_BY_LABEL = {}


class _AnyType(str):
    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


class WanTensorLeakProbe:
    """
    Enumerates all live torch.Tensor objects and prints top-N by memory,
    plus growth delta from the previous call with the same label. Insert in
    a loop body wired to any upstream output (IMAGE / INT / LATENT / etc.)
    via `trigger` to force execution order; passes the value through.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "probe"}),
                "trigger": (_ANY, {"forceInput": True}),
                "top_n": ("INT", {"default": 15, "min": 1, "max": 100}),
                "show_referrers": ("BOOLEAN", {"default": False, "tooltip": "For the single largest tensor, walk gc referrers to find what holds it."}),
            }
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "probe"
    CATEGORY = "WanChunkIO"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def probe(self, label, trigger, top_n, show_referrers):
        gc.collect()
        tensors = []
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor):
                    nbytes = obj.element_size() * obj.nelement()
                    if nbytes == 0:
                        continue
                    tensors.append((nbytes, tuple(obj.shape), str(obj.dtype), str(obj.device), id(obj)))
            except Exception:
                continue

        groups = {}
        for nbytes, shape, dtype, device, _id in tensors:
            key = (shape, dtype, device)
            if key not in groups:
                groups[key] = [0, 0]
            groups[key][0] += 1
            groups[key][1] += nbytes

        total_bytes = sum(v[1] for v in groups.values())
        total_count = sum(v[0] for v in groups.values())

        print(f"\n=== [LeakProbe:{label}] live tensors={total_count}  total={total_bytes/1e9:.2f} GB ===")
        print(f"{'cnt':>5} {'GB':>8}  shape                              dtype              device")
        sorted_groups = sorted(groups.items(), key=lambda x: -x[1][1])
        for (shape, dtype, device), (c, nb) in sorted_groups[:top_n]:
            print(f"{c:>5} {nb/1e9:>8.3f}  {str(shape):<34} {dtype:<18} {device}")

        prev = _LEAK_PREV_BY_LABEL.get(label)
        if prev:
            prev_groups, prev_total = prev
            print(f"--- Δ vs last [{label}]  totalΔ={((total_bytes-prev_total)/1e9):+.2f} GB  countΔ={total_count - sum(v[0] for v in prev_groups.values()):+d}")
            diffs = []
            all_keys = set(groups) | set(prev_groups)
            for key in all_keys:
                c1, b1 = groups.get(key, (0, 0))
                c0, b0 = prev_groups.get(key, (0, 0))
                if c1 != c0 or b1 != b0:
                    diffs.append((b1 - b0, c1 - c0, key))
            diffs.sort(key=lambda x: -abs(x[0]))
            for delta_b, delta_c, (shape, dtype, device) in diffs[:top_n]:
                if delta_b == 0 and delta_c == 0:
                    continue
                print(f"  Δcnt={delta_c:+5d}  ΔGB={delta_b/1e9:+8.3f}  {str(shape):<34} {dtype:<18} {device}")

        _LEAK_PREV_BY_LABEL[label] = ({k: tuple(v) for k, v in groups.items()}, total_bytes)

        if show_referrers and tensors:
            tensors.sort(key=lambda x: -x[0])
            biggest_id = tensors[0][4]
            target = None
            for obj in gc.get_objects():
                try:
                    if id(obj) == biggest_id:
                        target = obj
                        break
                except Exception:
                    continue
            if target is not None:
                print(f"--- referrers of biggest tensor shape={tuple(target.shape)} dtype={target.dtype} ---")
                refs = gc.get_referrers(target)
                for i, r in enumerate(refs[:10]):
                    try:
                        rtype = type(r).__name__
                        rsize = len(r) if hasattr(r, "__len__") else "-"
                        preview = str(r)[:120].replace("\n", " ")
                        print(f"  [{i}] {rtype}(len={rsize}): {preview}")
                    except Exception as e:
                        print(f"  [{i}] <repr-fail: {e}>")

        return (trigger,)


_OBJ_PREV_BY_LABEL = {}


def _size_of(obj):
    """Estimate byte size for common large-object types. Returns 0 if unknown/small."""
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        from PIL import Image as _PILImage
    except Exception:
        _PILImage = None

    t = type(obj)
    tn = t.__name__
    # torch.Tensor handled separately by the tensor probe
    if isinstance(obj, torch.Tensor):
        return 0
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.nbytes
    if _PILImage is not None and isinstance(obj, _PILImage.Image):
        try:
            w, h = obj.size
            m = obj.mode
            bpp = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3, "I": 4, "F": 4}.get(m, 4)
            return w * h * bpp
        except Exception:
            return 0
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, memoryview):
        try:
            return obj.nbytes
        except Exception:
            return 0
    # large str/list/tuple/dict — count only if big
    if isinstance(obj, str):
        n = len(obj)
        return n if n > 1_000_000 else 0
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        return (n * 8) if n > 100_000 else 0  # rough pointer cost only
    if isinstance(obj, dict):
        n = len(obj)
        return (n * 16) if n > 100_000 else 0
    return 0


class WanObjectLeakProbe:
    """
    Scans all live Python objects (non-tensor) for large allocations:
    numpy arrays, PIL images, bytes/bytearray, memoryview, plus unusually
    large str/list/tuple/dict. Prints top-N and growth deltas.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "obj"}),
                "trigger": (_ANY, {"forceInput": True}),
                "top_n": ("INT", {"default": 20, "min": 1, "max": 200}),
                "min_mb": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10240.0, "step": 0.1, "tooltip": "Only report individual objects ≥ this size (MB)."}),
                "show_referrers_biggest": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "probe"
    CATEGORY = "WanChunkIO"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def probe(self, label, trigger, top_n, min_mb, show_referrers_biggest):
        gc.collect()
        min_bytes = int(min_mb * 1024 * 1024)

        # Per-type aggregate
        by_type_count = {}
        by_type_bytes = {}
        # Big individual items (for referrer hunt)
        big_items = []  # (nbytes, type_name, detail, id)

        for obj in gc.get_objects():
            try:
                nbytes = _size_of(obj)
                if nbytes == 0:
                    continue
                tn = type(obj).__name__
                by_type_count[tn] = by_type_count.get(tn, 0) + 1
                by_type_bytes[tn] = by_type_bytes.get(tn, 0) + nbytes
                if nbytes >= min_bytes:
                    try:
                        import numpy as _np
                        if isinstance(obj, _np.ndarray):
                            detail = f"shape={obj.shape} dtype={obj.dtype}"
                        elif hasattr(obj, "size") and hasattr(obj, "mode"):
                            detail = f"{obj.size} mode={obj.mode}"
                        elif isinstance(obj, (bytes, bytearray, memoryview)):
                            detail = f"len={nbytes}"
                        else:
                            detail = f"len={len(obj) if hasattr(obj, '__len__') else '-'}"
                    except Exception:
                        detail = "-"
                    big_items.append((nbytes, tn, detail, id(obj)))
            except Exception:
                continue

        total_bytes = sum(by_type_bytes.values())
        print(f"\n=== [ObjLeakProbe:{label}] non-tensor objects total={total_bytes/1e9:.3f} GB types={len(by_type_count)} ===")
        print(f"{'type':<20} {'count':>9} {'GB':>10}")
        for tn, nb in sorted(by_type_bytes.items(), key=lambda x: -x[1])[:top_n]:
            print(f"{tn:<20} {by_type_count[tn]:>9} {nb/1e9:>10.3f}")

        big_items.sort(key=lambda x: -x[0])
        if big_items:
            print(f"--- individual objects ≥ {min_mb} MB (top {top_n}) ---")
            for nbytes, tn, detail, _id in big_items[:top_n]:
                print(f"  {nbytes/1e9:>7.3f} GB  {tn:<20} {detail}")

        prev = _OBJ_PREV_BY_LABEL.get(label)
        if prev:
            prev_counts, prev_bytes, prev_total = prev
            print(f"--- Δ vs last [{label}]  totalΔ={((total_bytes-prev_total)/1e9):+.3f} GB")
            all_tn = set(by_type_bytes) | set(prev_bytes)
            diffs = []
            for tn in all_tn:
                b1 = by_type_bytes.get(tn, 0); b0 = prev_bytes.get(tn, 0)
                c1 = by_type_count.get(tn, 0); c0 = prev_counts.get(tn, 0)
                if b1 != b0 or c1 != c0:
                    diffs.append((b1 - b0, c1 - c0, tn))
            diffs.sort(key=lambda x: -abs(x[0]))
            for delta_b, delta_c, tn in diffs[:top_n]:
                if delta_b == 0 and delta_c == 0:
                    continue
                print(f"  Δcnt={delta_c:+6d}  ΔGB={delta_b/1e9:+8.3f}  {tn}")

        _OBJ_PREV_BY_LABEL[label] = (dict(by_type_count), dict(by_type_bytes), total_bytes)

        if show_referrers_biggest and big_items:
            target_id = big_items[0][3]
            target = None
            for o in gc.get_objects():
                try:
                    if id(o) == target_id:
                        target = o
                        break
                except Exception:
                    continue
            if target is not None:
                print(f"--- referrers of biggest non-tensor object ({type(target).__name__}, {big_items[0][0]/1e9:.3f} GB) ---")
                for i, r in enumerate(gc.get_referrers(target)[:10]):
                    try:
                        rtype = type(r).__name__
                        rlen = len(r) if hasattr(r, "__len__") else "-"
                        preview = str(r)[:140].replace("\n", " ")
                        print(f"  [{i}] {rtype}(len={rlen}): {preview}")
                    except Exception as e:
                        print(f"  [{i}] <repr-fail: {e}>")

        return (trigger,)


NODE_CLASS_MAPPINGS = {
    "WanVideoChunkWriter": WanVideoChunkWriter,
    "WanVideoChunkAssembler": WanVideoChunkAssembler,
    "WanVideoChunkSessionReset": WanVideoChunkSessionReset,
    "WanVideoChunkSessionAuto": WanVideoChunkSessionAuto,
    "WanMemoryPurge": WanMemoryPurge,
    "WanVideoChunkLoader": WanVideoChunkLoader,
    "WanTensorLeakProbe": WanTensorLeakProbe,
    "WanObjectLeakProbe": WanObjectLeakProbe,
    "WanVideoChunkSaveAs": WanVideoChunkSaveAs,
    "WanVideoChunkCount": WanVideoChunkCount,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoChunkWriter": "Wan Video Chunk Writer",
    "WanVideoChunkAssembler": "Wan Video Chunk Assembler (ffmpeg)",
    "WanVideoChunkSessionReset": "Wan Video Chunk Session Reset",
    "WanVideoChunkSessionAuto": "Wan Video Chunk Session Auto-Increment",
    "WanMemoryPurge": "Wan Memory Purge (gc + malloc_trim)",
    "WanVideoChunkLoader": "Wan Video Chunk Loader (disk -> IMAGE)",
    "WanVideoChunkSaveAs": "Wan Video Chunk Save-As (new session)",
    "WanVideoChunkCount": "Wan Video Chunk Count (session -> INT)",
    "WanTensorLeakProbe": "Wan Tensor Leak Probe",
    "WanObjectLeakProbe": "Wan Object Leak Probe (non-tensor)",
}
