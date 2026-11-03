"""Timing boundaries, bounded batching and validated measurement settings."""

import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass


class BudgetExhausted(RuntimeError):
    pass


@contextmanager
def compiler_cache(path):
    """Isolate caches without selecting Python's configured-map execution route."""
    names = {"ME_DSL_JIT_CACHE_DIR": str(path), "ME_DSL_JIT_POS_CACHE": "0"}
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True)
class Settings:
    warmups: int = 2
    samples: int = 5
    target_ms: float = 50
    max_batch: int = 10000

    def __post_init__(self):
        if self.warmups < 0 or self.samples < 1 or self.max_batch < 1:
            raise ValueError("Invalid warmup, sample or batch count")
        if not math.isfinite(self.target_ms) or self.target_ms <= 0:
            raise ValueError("target_ms must be finite and positive")


def timed(call, clock=time.perf_counter_ns):
    start = clock()
    result = call()
    elapsed = clock() - start
    return result, elapsed


def warm_samples(call, first_ns, settings, deadline, clock=time.perf_counter_ns):
    """Each result survives its timer; deallocation is outside measured intervals.

    Sum per-call intervals rather than timing a whole Python batch: destruction
    of a prior output must not contaminate a later allocation's measurement.
    Samples in one process are explicitly not independent process observations.
    """
    iterations = min(settings.max_batch, max(1, math.ceil(settings.target_ms * 1e6 / max(first_ns, 1))))
    samples = []
    for sample in range(settings.warmups + settings.samples):
        elapsed = 0
        for _ in range(iterations):
            if time.monotonic() >= deadline:
                raise BudgetExhausted("Measurement budget exhausted")
            result, duration = timed(call, clock)
            elapsed += duration
            del result
        if sample >= settings.warmups:
            samples.append(
                {"elapsed_ns": elapsed, "iterations": iterations, "per_call_ns": elapsed / iterations}
            )
    return samples
