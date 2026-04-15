# ComfyUI-WanChunkIO

Memory-efficient chunk writer / assembler nodes for long-video VACE generation
with the **WanVideoWrapper**. Designed for workflows that loop over long source
videos (mask / pose / depth / source) in segments — instead of accumulating all
chunks in RAM until the end, each chunk is streamed to disk as a PNG sequence
and only a small overlap tail is kept in memory for blending into the next
iteration.

## Why

The default pattern of generating N video chunks, holding them all as `IMAGE`
tensors, then concatenating at the end can balloon system RAM well past the
container limit (a 2800-frame run at 848×480 float32 = ~14 GB *per stream*).

This pack moves the chunk store from RAM to disk while keeping your
in-graph blending workflow (e.g. KJNodes `ImageBatchExtendWithOverlap`)
intact.

## Nodes

### `Wan Video Chunk Session Auto-Increment`
Picks the next free `runN` session id by scanning `/workspace/temp_chunks/`.
Forces re-evaluation every queue so each run gets a unique id automatically.

- `prefix` — defaults to `"run"` (→ `run1`, `run2`, …)
- `create` — make the dir immediately

### `Wan Video Chunk Session Reset`
Wipes a session dir before reuse. Use this if you want a fixed `session_id`
instead of auto-increment.

### `Wan Video Chunk Writer`
Terminal sink for a chunk's `IMAGE` output. Writes frames to
`/workspace/temp_chunks/{session_id}/chunk_{NNNN}/######.png` and returns
only the trailing `overlap_frames` for the next iteration's blend.

- `images` — full chunk to write
- `session_id`, `chunk_index` — wired from session + loop counter
- `overlap_frames` — how many tail frames to keep (must match your blend node)
- `purge_vram` — runs `gc.collect()`, `torch.cuda.empty_cache()`, and
  `glibc malloc_trim(0)` so freed RSS is actually returned to the OS

Logs cgroup RAM, process RSS, and VRAM before/after every write so you can
verify memory is genuinely released.

### `Wan Video Chunk Assembler (ffmpeg)`
Concatenates all session chunks into a single mp4 via `ffmpeg image2`,
without loading frames back into RAM. Optional audio mux. Generates a
preview metadata payload so the assembled video shows inline on the node
(same UX as `VHS_VideoCombine` / `SaveVideo`).

- `session_id`, `fps`, `output_path`, `crf`
- `skip_overlap_frames` — drops the duplicate overlap frames at seams
- `overlap_trim_mode`:
  - `trailing_on_earlier` — for **in-graph blending** (e.g. KJNodes
    `ImageBatchExtendWithOverlap`); drops the unblended tail of every chunk
    *except the last*, keeping the blended head at the start of each next
    chunk.
  - `leading_on_later` — for **hard-cut** workflows (no in-graph blend);
    drops the duplicate overlap at the start of every chunk after the first.
  - `none` — keep everything as-is.
- `delete_chunks_after` — wipe the session dir after assembly
- `audio` (optional) — `AUDIO` input, muxed in as AAC 192k
- `audio_offset_sec` — shift audio relative to video
- `wait_for` (optional INT) — wire `WanVideoChunkWriter.next_chunk_index`
  here in single-graph workflows to force the assembler to run *after* the
  writer (otherwise ComfyUI may run them in parallel).

## Recommended wiring

```
SessionAuto ──► session_id ──┬─► Writer
                             └─► Assembler

Generate chunk ──► (KJNodes ImageBatchExtendWithOverlap with prev tail)
                            │
                            └─► Writer.images
                                   │
                                   ├─► overlap_tail ─► next iteration's blend
                                   └─► next_chunk_index ─► Assembler.wait_for
```

Settings for a typical setup with 81-frame chunks and 16-frame overlap:

| Param | Value |
|---|---|
| Writer `overlap_frames` | 16 |
| Assembler `skip_overlap_frames` | 16 |
| Assembler `overlap_trim_mode` | `trailing_on_earlier` |

## Verifying memory release

The writer prints lines like:

```
[ChunkWriter] session=run7 chunk=3 wrote 81 frames 848x480 -> .../chunk_0003
  before: ram=78.21GB/200.00GB rss=77.95GB | vram alloc=18.42GB reserved=22.10GB
  after:  ram=76.88GB/200.00GB rss=76.60GB | vram alloc=2.31GB reserved=4.50GB
```

If `ram` is roughly flat across iterations (not climbing), the writer is doing
its job. If it climbs, something upstream is still holding a reference to the
chunk tensor — common culprits: `PreviewImage`, `SaveImage`, or a passthrough
node downstream of the same `IMAGE` output.

## Requirements

- ComfyUI
- ffmpeg in `$PATH`
- Linux (uses cgroup memory paths and `glibc malloc_trim`)
- A writable directory at `/workspace/temp_chunks/` (override `TEMP_ROOT`
  in `nodes.py` if needed)

## License

MIT
