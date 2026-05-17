"""
Monkey-patch psutil.virtual_memory() so that ComfyUI's RAMPressureCache
sees our cgroup-v2 (or v1) container memory instead of the host's memory.

On RunPod / Kubernetes / Docker, psutil reports the shared host's free RAM
(multiple TB), which means --cache-ram THRESHOLD never triggers eviction
because host available never drops below THRESHOLD GB.

This patch overrides psutil.virtual_memory() so:
    total     = cgroup memory.max
    available = memory.max - (memory.current - inactive_file)
    used      = memory.current - inactive_file
    percent   = 100 * used / total

inactive_file (from memory.stat) is reclaimable page cache that the kernel
frees on demand. Without subtracting it, "used" includes cached model files
read from disk, making available appear much lower than reality (e.g. 21 GB
reported vs 184 GB actual on Vast.ai with warm file caches).

Other psutil fields are left at host defaults (rarely used by ComfyUI).
"""

import logging
import os
from collections import namedtuple

import psutil

_V2_CURRENT = "/sys/fs/cgroup/memory.current"
_V2_MAX = "/sys/fs/cgroup/memory.max"
_V1_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"


def _cgroup_paths():
    if os.path.exists(_V2_CURRENT):
        return _V2_CURRENT, _V2_MAX
    if os.path.exists(_V1_USAGE):
        return _V1_USAGE, _V1_LIMIT
    return None, None


_USAGE_PATH, _LIMIT_PATH = _cgroup_paths()


def _read(path):
    try:
        with open(path) as f:
            s = f.read().strip()
        if s == "max":
            return None
        return int(s)
    except Exception:
        return None


_ORIG_VIRTUAL_MEMORY = psutil.virtual_memory
_SAMPLE = _ORIG_VIRTUAL_MEMORY()
_SVMEM = type(_SAMPLE)  # svmem namedtuple class
_SVMEM_FIELDS = _SAMPLE._fields


def _reclaimable_cache():
    """Read inactive_file from cgroup memory.stat (reclaimable page cache)."""
    stat_path = os.path.join(os.path.dirname(_USAGE_PATH), "memory.stat") if _USAGE_PATH else None
    if stat_path is None:
        return 0
    try:
        with open(stat_path) as f:
            for line in f:
                if line.startswith("inactive_file "):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _cgroup_virtual_memory():
    used = _read(_USAGE_PATH) if _USAGE_PATH else None
    total = _read(_LIMIT_PATH) if _LIMIT_PATH else None
    if used is None or total is None or total <= 0 or total > (1 << 62):
        return _ORIG_VIRTUAL_MEMORY()
    reclaimable = _reclaimable_cache()
    available = max(0, total - used + reclaimable)
    effective_used = max(0, used - reclaimable)
    percent = (effective_used / total) * 100.0
    host = _ORIG_VIRTUAL_MEMORY()
    kwargs = {name: getattr(host, name) for name in _SVMEM_FIELDS}
    kwargs["total"] = total
    kwargs["available"] = available
    kwargs["used"] = effective_used
    kwargs["free"] = available
    kwargs["percent"] = percent
    return _SVMEM(**kwargs)


def apply():
    if _USAGE_PATH is None or _LIMIT_PATH is None:
        logging.info("[wan_chunk_io] cgroup psutil patch: no cgroup found, leaving psutil alone")
        return
    total = _read(_LIMIT_PATH)
    if total is None or total > (1 << 62):
        logging.info("[wan_chunk_io] cgroup psutil patch: cgroup memory unlimited, leaving psutil alone")
        return
    psutil.virtual_memory = _cgroup_virtual_memory
    v = psutil.virtual_memory()
    logging.info(
        f"[wan_chunk_io] psutil.virtual_memory monkey-patched to cgroup "
        f"(total={v.total/1e9:.1f} GB available={v.available/1e9:.1f} GB used={v.used/1e9:.1f} GB)"
    )
