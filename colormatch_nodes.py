"""
Color-match a new VACE chunk's frames against the overlap tail returned by
WanVideoChunkWriter, so each continuation anchors to the *previous chunk's
actual output* instead of a static base reference.

Drop-in replacement for KJNodes ColorMatchV2 in the continuation branch of
a long-form WAN VACE extension workflow:

    WanVideoChunkWriter -> overlap_tail
                                 |
                                 v
        new_chunk_frames -> ColorMatchToOverlapTail -> (matched frames)

Method options match ColorMatchV2 exactly so it is a true drop-in.

Why this exists
---------------
A workflow that prep-color-matches each continuation against `Get_VideoBase`
(initial frames) accumulates a small but monotonic luma/saturation drift per
chunk boundary, because the static reference does not track the live state
of the run. Anchoring to the live overlap tail removes that accumulator.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


def _is_valid_ref(image_ref, min_frames):
    if image_ref is None:
        return False
    if not isinstance(image_ref, torch.Tensor):
        return False
    if image_ref.dim() != 4:
        return False
    if image_ref.size(0) < min_frames:
        return False
    return True


def _build_ref_index_map(N_src, N_ref, mode):
    """For frame-i of source, returns the ref-frame index to match against.
    Returns None for 'broadcast_mean' (caller must pool stats instead)."""
    if mode == "frame_aligned":
        return [min(i, N_ref - 1) for i in range(N_src)]
    if mode == "broadcast_last":
        return [N_ref - 1] * N_src
    if mode == "broadcast_mean":
        return None
    raise ValueError(f"unknown reference_mode: {mode}")


class ColorMatchToOverlapTail:
    """
    Color-match `image_target` (a new chunk's frames) against `image_ref`
    (the overlap_tail from WanVideoChunkWriter, or any IMAGE batch).

    `reference_mode` controls how each target frame picks its ref:
      - frame_aligned: target[i] matches ref[min(i, N_ref-1)]. Best for the
        N overlap frames at the head — they are the SAME timeline positions
        as the prev chunk's tail, so a per-frame match is most accurate.
        Beyond the overlap, the last ref frame is broadcast.
      - broadcast_last: every target frame matches against ref[-1].
        Simpler; loses temporal info inside the overlap region.
      - broadcast_mean: pool all ref frames into one distribution and match
        every target frame to that single distribution.

    `bypass_min_ref_frames` handles the WanVideoChunkWriter no-tail fallback
    (a single zero frame when there is no previous chunk): if the ref has
    fewer frames than the threshold, the node passes target through unchanged
    so it is safe to keep wired in chunk 0 too.

    Methods match KJNodes ColorMatchV2 — including `reinhard_lab_gpu` which
    runs on CUDA via Kornia for ~10x throughput on long batches.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "image_ref": ("IMAGE",),
                "method": (
                    ["mkl", "hm", "reinhard", "mvgd",
                     "hm-mvgd-hm", "hm-mkl-hm", "reinhard_lab_gpu"],
                    {"default": "mkl"},
                ),
                "reference_mode": (
                    ["frame_aligned", "broadcast_last", "broadcast_mean"],
                    {"default": "frame_aligned"},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "bypass_min_ref_frames": (
                    "INT",
                    {"default": 2, "min": 0, "max": 1024,
                     "tooltip": "If image_ref has fewer frames than this, "
                                "passthrough. Handles WanVideoChunkWriter's "
                                "no-tail fallback (1 zero frame)."},
                ),
                "multithread": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "match"
    CATEGORY = "WanChunkIO"

    def match(self, image_target, image_ref, method, reference_mode,
              strength, bypass_min_ref_frames, multithread):
        if strength == 0:
            print("[ColorMatchToOverlapTail] strength=0; passthrough")
            return (image_target,)
        if not _is_valid_ref(image_ref, bypass_min_ref_frames):
            ref_n = (image_ref.size(0) if isinstance(image_ref, torch.Tensor)
                     and image_ref.dim() == 4 else 0)
            print(f"[ColorMatchToOverlapTail] ref has {ref_n} frames "
                  f"< bypass_min_ref_frames={bypass_min_ref_frames}; "
                  f"passthrough")
            return (image_target,)

        N_src = image_target.size(0)
        N_ref = image_ref.size(0)
        print(f"[ColorMatchToOverlapTail] {N_src} target frames, "
              f"{N_ref} ref frames, method={method}, "
              f"reference_mode={reference_mode}, strength={strength}")

        if method == "reinhard_lab_gpu":
            out = self._reinhard_lab_gpu(
                image_target, image_ref, reference_mode, strength
            )
            return (out,)

        return (self._color_matcher_cpu(
            image_target, image_ref, method, reference_mode,
            strength, multithread
        ),)

    # ------------------------------------------------------------------
    # CPU path via color-matcher library (matches KJNodes ColorMatchV2)
    # ------------------------------------------------------------------
    def _color_matcher_cpu(self, image_target, image_ref, method,
                           reference_mode, strength, multithread):
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            raise RuntimeError(
                "color-matcher not installed. pip install color-matcher"
            )

        N_src = image_target.size(0)
        N_ref = image_ref.size(0)
        index_map = _build_ref_index_map(N_src, N_ref, reference_mode)

        # Pre-build the pooled ref once if broadcast_mean
        if reference_mode == "broadcast_mean":
            # Concatenate ref frames spatially along the height axis. This
            # preserves spatial variation while pooling temporal info into
            # one ref array, which the color-matcher methods can ingest as
            # a single 2D image.
            ref_np = (image_ref
                      .cpu()
                      .reshape(N_ref * image_ref.shape[1],
                               image_ref.shape[2], 3)
                      .numpy())
        else:
            ref_cpu = image_ref.cpu()

        target_cpu = image_target.cpu()

        def process(i):
            cm = ColorMatcher()
            tgt_np = target_cpu[i].numpy()
            if reference_mode == "broadcast_mean":
                ref_i_np = ref_np
            else:
                ref_i_np = ref_cpu[index_map[i]].numpy()
            try:
                result = cm.transfer(src=tgt_np, ref=ref_i_np, method=method)
                if strength != 1.0:
                    result = tgt_np + strength * (result - tgt_np)
                return torch.from_numpy(result)
            except Exception as e:
                print(f"[ColorMatchToOverlapTail] frame {i} failed: {e}; "
                      f"using source")
                return torch.from_numpy(tgt_np)

        if multithread and N_src > 1:
            n_workers = min(os.cpu_count() or 1, N_src)
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                out = list(ex.map(process, range(N_src)))
        else:
            out = [process(i) for i in range(N_src)]

        return torch.stack(out, dim=0).to(torch.float32).clamp_(0, 1)

    # ------------------------------------------------------------------
    # GPU path via Kornia LAB Reinhard (mean+std), batched
    # ------------------------------------------------------------------
    def _reinhard_lab_gpu(self, image_target, image_ref,
                          reference_mode, strength):
        import kornia
        try:
            import comfy.model_management as mm
            device = mm.get_torch_device()
        except Exception:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        N_src = image_target.size(0)
        N_ref = image_ref.size(0)

        src = image_target.to(device).permute(0, 3, 1, 2).contiguous()
        src_lab = kornia.color.rgb_to_lab(src)
        B, C, H, W = src_lab.shape
        src_flat = src_lab.view(B, C, -1)
        src_std, src_mean = torch.std_mean(
            src_flat, dim=-1, keepdim=True, unbiased=False
        )
        src_std = src_std.clamp_min_(1e-6)

        # Compute (ref_mean, ref_std) per source frame
        if reference_mode == "broadcast_mean":
            # Pool every pixel of every ref frame into one distribution.
            ref_all = (image_ref.to(device)
                       .permute(0, 3, 1, 2).contiguous())
            ref_lab_all = kornia.color.rgb_to_lab(ref_all)
            # (N_ref, C, H, W) -> (C, N_ref*H*W)
            ref_pixels = ref_lab_all.permute(1, 0, 2, 3).reshape(C, -1)
            ref_std_g = ref_pixels.std(dim=-1, unbiased=False, keepdim=True)
            ref_mean_g = ref_pixels.mean(dim=-1, keepdim=True)
            ref_std = ref_std_g.unsqueeze(0).expand(B, -1, -1)
            ref_mean = ref_mean_g.unsqueeze(0).expand(B, -1, -1)
        elif reference_mode == "broadcast_last":
            last = image_ref[-1:].to(device).permute(0, 3, 1, 2).contiguous()
            last_lab = kornia.color.rgb_to_lab(last)
            ref_flat = last_lab.view(1, C, -1)
            ref_std_1, ref_mean_1 = torch.std_mean(
                ref_flat, dim=-1, keepdim=True, unbiased=False
            )
            ref_std = ref_std_1.expand(B, -1, -1)
            ref_mean = ref_mean_1.expand(B, -1, -1)
        else:  # frame_aligned
            idx = torch.arange(B, device=device).clamp_max_(N_ref - 1)
            ref_sel = (image_ref[idx.cpu()]
                       .to(device).permute(0, 3, 1, 2).contiguous())
            ref_lab = kornia.color.rgb_to_lab(ref_sel)
            ref_flat = ref_lab.view(B, C, -1)
            ref_std, ref_mean = torch.std_mean(
                ref_flat, dim=-1, keepdim=True, unbiased=False
            )

        corrected_flat = (
            (src_flat - src_mean) * (ref_std / src_std) + ref_mean
        )
        corrected_lab = corrected_flat.view(B, C, H, W)
        corrected_rgb = kornia.color.lab_to_rgb(corrected_lab)

        out = (1.0 - strength) * src + strength * corrected_rgb
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.cpu().float().clamp_(0, 1)


NODE_CLASS_MAPPINGS = {
    "ColorMatchToOverlapTail": ColorMatchToOverlapTail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorMatchToOverlapTail": "Color Match to Overlap Tail",
}
