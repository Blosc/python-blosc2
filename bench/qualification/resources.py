"""Best-effort resource fingerprint; unknown limits stay null, never zero."""

import math
import os
import platform
import subprocess
from pathlib import Path


def cgroup_limit(path):
    try:
        value = Path(path).read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def cpu_count():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return count


def memory_limit():
    try:
        physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        physical = None
    limits = [
        physical,
        cgroup_limit("/sys/fs/cgroup/memory.max"),
        cgroup_limit("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    return min(value for value in limits if value is not None and value > 0) if any(limits) else None


def cpu_model():
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None
