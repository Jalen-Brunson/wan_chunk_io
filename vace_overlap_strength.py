"""
Per-frame attenuation of the VACE control signal at the head of a
continuation chunk in a long-form WAN VACE extension workflow.

Background
----------
A continuation chunk uses LatentTailContinuation to seed its first N latents
from the previous chunk's tail. Without this node, the VACE control signal
(pose/depth/grey) still drives those head frames at full strength, fighting
the carry-over latents and producing identity drift at the seam.

How it works
------------
For each frame in the overlap region, this node BLENDS `vace_frames` toward
a "neutral" baseline that the VACE block sees as zero contribution:

    blended[t] = mask[t] * original[t] + (1 - mask[t]) * neutral

    mask = [low, low, ..., low,  ramp(low->1), ..., ramp,  1.0, 1.0, ..., 1.0]
            <- overlap region ->  <- transition region ->  <- normal ->

At mask=1 the original VACE features pass through. At mask=0 the VACE block
sees the same "absent position" pattern that vace_step_schedule's pad
helper stamps — `latents_mean` for the 16 content channels (which post-
normalizes to 0 via native's process_latent_in), zeros elsewhere.

Why the blending matters
------------------------
A previous version simply multiplied vace_frames by the mask. At mask=0
this produced raw zeros, which native's `process_latent_in = (x - mean)/std`
then turned into per-channel constants ≈ ±0.5 — a pattern VACE was NOT
trained on. The result was fuzzy/blotchy output in the overlap region.

The neutral-blend approach matches what VACE was trained to ignore, so
mask=0 cleanly says "no VACE contribution here" instead of "decode this
garbage signal."

Spatial-aware mode (optional `inpaint_mask` input)
--------------------------------------------------
Without a spatial mask, attenuation is applied UNIFORMLY across the whole
frame — both inside the inpaint region (where you want it, so carry-over
dominates) and outside (where you don't, so untouched surroundings drift:
"snake -> lamp" hallucinations at low_strength<=0.5).

Pass the same inpaint_mask you fed to WanVaceToVideoScheduled. The node
combines per-latent-frame:

    combined[t,h,w] = 1 - spatial[t,h,w] * (1 - temporal[t])

Outside the inpaint region (spatial=0) combined=1.0, so VACE control is
preserved at full strength. Inside (spatial=1) the temporal curve applies
as before. Soft mask boundaries blend smoothly.

Insert AFTER the VACE encode node (e.g. WanVaceToVideoScheduled) and BEFORE
the sampler. `vace_mask` (regen vs preserve) is left untouched.
"""
import math
from typing import List, Optional

import torch
import torch.nn.functional as F


def _pix_to_lat_frames(pix_frames: int, vae_t_comp: int) -> int:
    """How many latent frames does the first `pix_frames` of pixel video occupy
    under WAN's t/4+1 layout? frame 0 -> latent 0; frames [1+(k-1)*comp,
    1+k*comp) -> latent k for k>=1. Total: T_lat = 1 + ceil((T_pix-1)/comp).
    """
    if pix_frames <= 0:
        return 0
    return 1 + (pix_frames - 1 + vae_t_comp - 1) // vae_t_comp


def _normalize_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    """Comfy MASK shapes vary: (T,H,W), (1,T,H,W), (T,1,H,W), (B,T,H,W).
    Reduce to (T,H,W). Picks the first batch slot for batched masks."""
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4:
        if mask.shape[0] == 1:
            return mask[0]
        if mask.shape[1] == 1:
            return mask[:, 0]
        return mask[0]
    if mask.ndim == 5:
        if mask.shape[0] == 1 and mask.shape[1] == 1:
            return mask[0, 0]
        return mask[0, 0]
    raise ValueError(f"unsupported mask ndim: {mask.ndim} (shape={tuple(mask.shape)})")


def _resize_mask_to_latent(mask: torch.Tensor, T_lat: int,
                            H_lat: int, W_lat: int,
                            vae_t_comp: int) -> torch.Tensor:
    """Resize a pixel-space inpaint mask to latent resolution.

    Spatial: max-pool from (H_pix, W_pix) to (H_lat, W_lat). Max-pool is
    conservative — any inpainted pixel in the spatial cell makes the latent
    cell inpainted. This avoids accidentally exposing partially-masked
    regions to attenuation when nearest-neighbor would have rounded them
    down to "preserve".

    Temporal: max-pool under WAN's t/4+1 layout. Latent 0 maps from pixel
    frame 0; latent k>=1 maps from pixel frames [(k-1)*comp+1, k*comp].

    Returns: (T_lat, H_lat, W_lat) float tensor in [0, 1].
    """
    mask = _normalize_mask_shape(mask).float()
    T_pix, H_pix, W_pix = mask.shape

    # Spatial: max-pool. Use exact pooling when divisible, else nearest interp.
    if H_pix % H_lat == 0 and W_pix % W_lat == 0 and H_pix > 0 and W_pix > 0:
        kh = H_pix // H_lat
        kw = W_pix // W_lat
        m_sp = F.max_pool2d(mask.unsqueeze(1), kernel_size=(kh, kw)).squeeze(1)
    else:
        m_sp = F.interpolate(
            mask.unsqueeze(1),
            size=(H_lat, W_lat),
            mode="nearest",
        ).squeeze(1)

    # Temporal: max-pool under WAN t/4+1.
    out = torch.zeros((T_lat, H_lat, W_lat), dtype=torch.float32,
                      device=mask.device)
    if T_pix > 0:
        out[0] = m_sp[0]
    for k in range(1, T_lat):
        start = (k - 1) * vae_t_comp + 1
        end = k * vae_t_comp + 1
        if start >= T_pix:
            # Past end of pixel mask. Pad with 1.0 (assume body = inpaint).
            # The temporal attenuation curve is 1.0 in the body region anyway,
            # so this pad has no effect on the combined mask there.
            out[k] = 1.0
            continue
        end = min(end, T_pix)
        out[k] = m_sp[start:end].amax(dim=0)

    return out.clamp_(0.0, 1.0)


def _build_lat_mask(T_lat: int, n_low: int, n_trans: int,
                    low: float, curve: str) -> torch.Tensor:
    """Build a (T_lat,) per-latent-frame strength multiplier.

    First n_low frames = low. Next n_trans frames = curve(low -> 1.0). Rest = 1.0.
    n_low + n_trans is clamped to T_lat.
    """
    n_low = max(0, min(n_low, T_lat))
    n_trans = max(0, min(n_trans, T_lat - n_low))
    mask = torch.ones(T_lat, dtype=torch.float32)
    if n_low > 0:
        mask[:n_low] = low
    if n_trans > 0:
        for i in range(n_trans):
            # Distribute breakpoints so neither end hits 0 nor 1 exactly,
            # giving a smooth visual transition off the low region.
            t = (i + 1) / (n_trans + 1)
            if curve == "cosine":
                alpha = 0.5 * (1.0 - math.cos(math.pi * t))
            elif curve == "smoothstep":
                alpha = 3.0 * t * t - 2.0 * t * t * t
            elif curve == "linear":
                alpha = t
            elif curve == "hard":
                alpha = 1.0  # snap straight to 1.0 — no ramp
            else:
                raise ValueError(f"unknown curve: {curve}")
            mask[n_low + i] = low + (1.0 - low) * alpha
    return mask


def _build_vace_neutral(vf: torch.Tensor) -> torch.Tensor:
    """Build a tensor of vf's shape that lands as 0 at the VACE block after
    native's process_latent_in. Mirrors `_neutral_vace_pad` in
    vace_step_schedule: stamp `latents_mean` (via process_out(zeros)) per
    16-channel slab when the channel layout matches WAN VACE's typical
    16-or-32-channel shape; otherwise return raw zeros and trust the caller
    that those positions don't go through process_latent_in.
    """
    out = torch.zeros_like(vf)
    if out.shape[1] % 16 == 0 and out.shape[1] <= 32:
        try:
            import comfy.latent_formats
            lf = comfy.latent_formats.Wan21()
            for i in range(0, out.shape[1], 16):
                out[:, i:i + 16] = lf.process_out(out[:, i:i + 16])
        except Exception as e:
            print(f"[VaceOverlapStrengthMask] could not stamp latents_mean "
                  f"({e}); falling back to raw zeros — VACE may produce "
                  f"artifacts at masked positions")
    return out


def _apply_mask_to_cond_list(cond_list, mask_full: torch.Tensor,
                             verbose: bool, label: str,
                             spatial_mask: Optional[torch.Tensor] = None,
                             vae_t_comp: int = 4):
    """Walk a CONDITIONING list, blend every `vace_frames` tensor toward the
    neutral baseline. Two modes:

    - Temporal-only (spatial_mask=None): scalar attenuation per latent frame
      applied uniformly across H,W. Original behavior.
    - Spatial-aware (spatial_mask given): combined mask
      = 1 - spatial[t,h,w] * (1 - temporal[t]). Outside the inpaint region
      (spatial=0) the combined mask is 1.0 → no attenuation, VACE control
      passes through. Inside (spatial=1) the temporal curve applies as
      before. Soft boundaries blend.

    spatial_mask is expected at latent resolution (T_lat, H_lat, W_lat).
    """
    if not cond_list:
        return cond_list
    out = []
    n_modified_tensors = 0
    n_entries_with_vace = 0
    for entry in cond_list:
        if not (
            isinstance(entry, (list, tuple))
            and len(entry) > 1
            and isinstance(entry[1], dict)
        ):
            out.append(entry)
            continue

        meta = dict(entry[1])
        frames: List[torch.Tensor] = meta.get("vace_frames") or []
        if frames:
            n_entries_with_vace += 1
            new_frames = []
            for vf in frames:
                if not (hasattr(vf, "shape") and vf.ndim >= 3):
                    new_frames.append(vf)
                    continue
                T_here = vf.shape[2]

                # Trim/pad temporal mask to vf's T.
                if T_here <= mask_full.numel():
                    t_mask = mask_full[:T_here]
                else:
                    pad = torch.ones(
                        T_here - mask_full.numel(), dtype=mask_full.dtype
                    )
                    t_mask = torch.cat([mask_full, pad], dim=0)
                t_mask = t_mask.to(device=vf.device, dtype=vf.dtype)

                if spatial_mask is None:
                    # Original temporal-only path.
                    shape = [1] * vf.ndim
                    shape[2] = t_mask.numel()
                    m_b = t_mask.view(*shape)
                else:
                    # Spatial-aware: resize the mask to vf's H,W and trim/pad T.
                    H, W = vf.shape[3], vf.shape[4]
                    s_mask = spatial_mask
                    if s_mask.shape[1] != H or s_mask.shape[2] != W:
                        s_mask = F.interpolate(
                            s_mask.unsqueeze(1),
                            size=(H, W),
                            mode="nearest",
                        ).squeeze(1)
                    if T_here <= s_mask.shape[0]:
                        s_mask = s_mask[:T_here]
                    else:
                        pad = torch.ones(
                            T_here - s_mask.shape[0], H, W,
                            dtype=s_mask.dtype, device=s_mask.device,
                        )
                        s_mask = torch.cat([s_mask, pad], dim=0)
                    s_mask = s_mask.to(device=vf.device, dtype=vf.dtype)

                    # combined[t,h,w] = 1 - spatial[t,h,w] * (1 - temporal[t])
                    t_b = t_mask.view(T_here, 1, 1)
                    combined = 1.0 - s_mask * (1.0 - t_b)
                    # Reshape for broadcast over (B, C, T, H, W).
                    m_b = combined.view(1, 1, T_here, H, W)

                # Blend toward neutral, NOT multiply by mask. Multiplying
                # produces raw zeros at masked positions, which native's
                # process_latent_in turns into ±0.5 garbage that VACE wasn't
                # trained on — the cause of the fuzzy/blotchy overlap region.
                neutral = _build_vace_neutral(vf)
                new_vf = m_b * vf + (1.0 - m_b) * neutral
                new_frames.append(new_vf)
                n_modified_tensors += 1
            meta["vace_frames"] = new_frames

        rebuilt = list(entry)
        rebuilt[1] = meta
        out.append(type(entry)(rebuilt) if isinstance(entry, tuple) else rebuilt)

    if verbose:
        mode = "spatial+temporal" if spatial_mask is not None else "temporal-only"
        print(f"[VaceOverlapStrengthMask] {label}: blended "
              f"{n_modified_tensors} vace_frames tensors ({mode}) "
              f"across {n_entries_with_vace} cond entries")
    return out


class VaceOverlapStrengthMask:
    """
    Attenuate VACE control at the head of a continuation chunk so the latent
    carry-over (LatentTailContinuation) can dominate over the overlap region.

    Inputs (positive, negative): conditioning produced by a VACE encode node
    that stamps `vace_frames` into the conditioning options dict
    (WanVaceToVideoScheduled, stock WanVaceToVideo, etc).

    Effect: multiplies vace_frames along T_lat by a smooth ramp. The user's
    per-step strength schedule (vace_step_schedule) still applies on top —
    the two compound multiplicatively.

    Set overlap_pixel_frames=0 to bypass (e.g. wire the same node into chunk 0
    where there is no overlap).

    Pixel->latent mapping uses WAN's t/4+1 layout. Override with
    `vae_temporal_compression` if you're on a non-WAN VAE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "overlap_pixel_frames": (
                    "INT",
                    {"default": 17, "min": 0, "max": 4096,
                     "tooltip": "Number of pixel frames at the head to "
                                "fully attenuate. Match WanVideoChunkWriter "
                                "overlap_frames. Set 0 to bypass."},
                ),
                "transition_pixel_frames": (
                    "INT",
                    {"default": 17, "min": 0, "max": 4096,
                     "tooltip": "Number of additional pixel frames after the "
                                "overlap to ramp from low_strength back to "
                                "1.0. 0 = hard step at end of overlap."},
                ),
                "low_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Strength multiplier inside the overlap "
                                "region. 0.0 = fully kill VACE control, "
                                "carry-over latent dominates. 0.2-0.4 keeps "
                                "some pose hint while still favoring "
                                "carry-over."},
                ),
                "curve": (
                    ["cosine", "smoothstep", "linear", "hard"],
                    {"default": "cosine"},
                ),
                "vae_temporal_compression": (
                    "INT",
                    {"default": 4, "min": 1, "max": 16,
                     "tooltip": "WAN VAE = 4. Adjust only for other VAEs."},
                ),
                "verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "apply_to_negative": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Whether to also attenuate vace_frames on "
                                "the negative conditioning. Usually True so "
                                "CFG sees a consistent control signal."},
                ),
                "inpaint_mask": (
                    "MASK",
                    {"tooltip": "Optional pixel-space inpaint mask "
                                "(same one fed to WanVaceToVideoScheduled). "
                                "When provided, temporal attenuation only "
                                "applies INSIDE the inpaint region. Outside "
                                "(mask=0, 'preserve'), VACE control passes "
                                "through at full strength so untouched "
                                "surroundings don't drift. Without this, "
                                "attenuation is uniform across the frame."},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "WanChunkIO"

    def apply(self, positive, negative, overlap_pixel_frames,
              transition_pixel_frames, low_strength, curve,
              vae_temporal_compression, verbose, apply_to_negative=True,
              inpaint_mask=None):
        if overlap_pixel_frames == 0 and transition_pixel_frames == 0:
            if verbose:
                print("[VaceOverlapStrengthMask] overlap=0, transition=0; "
                      "passthrough")
            return (positive, negative)
        if low_strength == 1.0 and curve != "hard":
            if verbose:
                print("[VaceOverlapStrengthMask] low_strength=1.0; "
                      "passthrough (no attenuation)")
            return (positive, negative)

        # Find the largest T_lat across both conditionings so we build a
        # mask that covers everything we might need to scale. Also find the
        # H_lat,W_lat for spatial-mask resizing.
        max_T = 0
        H_lat = 0
        W_lat = 0
        for cond_list in (positive, negative):
            if not cond_list:
                continue
            for entry in cond_list:
                if (isinstance(entry, (list, tuple))
                        and len(entry) > 1
                        and isinstance(entry[1], dict)):
                    for vf in entry[1].get("vace_frames") or []:
                        if hasattr(vf, "shape") and vf.ndim >= 3:
                            max_T = max(max_T, vf.shape[2])
                            if vf.ndim >= 5:
                                H_lat = max(H_lat, vf.shape[3])
                                W_lat = max(W_lat, vf.shape[4])
        if max_T == 0:
            if verbose:
                print("[VaceOverlapStrengthMask] no vace_frames found in "
                      "either conditioning; passthrough")
            return (positive, negative)

        n_low = _pix_to_lat_frames(overlap_pixel_frames, vae_temporal_compression)
        n_trans = _pix_to_lat_frames(transition_pixel_frames, vae_temporal_compression)
        # _pix_to_lat_frames returns 1 for pix=1 — but transition_pixel_frames=0
        # should yield n_trans=0, not 1. Fix the edge case explicitly.
        if transition_pixel_frames == 0:
            n_trans = 0
        if overlap_pixel_frames == 0:
            n_low = 0

        mask = _build_lat_mask(max_T, n_low, n_trans, float(low_strength), curve)

        # Optional spatial mask — restricts attenuation to inside the inpaint
        # region so untouched surroundings keep full VACE control and don't
        # drift (e.g., snake -> lamp at low_strength=0.5).
        spatial_mask = None
        if inpaint_mask is not None and H_lat > 0 and W_lat > 0:
            try:
                spatial_mask = _resize_mask_to_latent(
                    inpaint_mask, max_T, H_lat, W_lat,
                    vae_temporal_compression,
                )
            except Exception as e:
                print(f"[VaceOverlapStrengthMask] failed to resize "
                      f"inpaint_mask ({e}); falling back to temporal-only "
                      f"attenuation")
                spatial_mask = None

        if verbose:
            preview_n = min(12, max_T)
            print(f"[VaceOverlapStrengthMask] T_lat={max_T} "
                  f"n_low_lat={n_low} n_trans_lat={n_trans} "
                  f"low={low_strength} curve={curve}")
            print(f"  mask first {preview_n}: "
                  f"{[f'{v:.3f}' for v in mask[:preview_n].tolist()]}")
            if max_T > preview_n:
                print(f"  mask last 4: "
                      f"{[f'{v:.3f}' for v in mask[-4:].tolist()]}")
            if spatial_mask is not None:
                covered = float((spatial_mask > 0.5).float().mean())
                print(f"  spatial_mask: latent shape="
                      f"({spatial_mask.shape[0]},{spatial_mask.shape[1]},"
                      f"{spatial_mask.shape[2]}) inpaint_coverage="
                      f"{covered*100:.1f}%")

        pos_out = _apply_mask_to_cond_list(
            positive, mask, verbose, "positive",
            spatial_mask=spatial_mask,
            vae_t_comp=vae_temporal_compression,
        )
        if apply_to_negative:
            neg_out = _apply_mask_to_cond_list(
                negative, mask, verbose, "negative",
                spatial_mask=spatial_mask,
                vae_t_comp=vae_temporal_compression,
            )
        else:
            neg_out = negative
        return (pos_out, neg_out)


NODE_CLASS_MAPPINGS = {
    "VaceOverlapStrengthMask": VaceOverlapStrengthMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VaceOverlapStrengthMask": "VACE Overlap Strength Mask",
}
