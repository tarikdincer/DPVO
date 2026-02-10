import time
from contextlib import contextmanager
from typing import Optional
import torch


@contextmanager
def timed_block(name, sync_cuda=True, enabled=True):
    if not enabled:
        yield
        return

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    yield

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    print(f"[TIMER] {name}: {(end-start)*1000:.2f} ms")

