import time
from contextlib import contextmanager
from typing import Optional


@contextmanager
def timed_block(
    name: str,
    *, 
    sync_cuda: bool = True,
    enabled: bool = False
):
    """
    Context manager to time a code block.

    Args:
        name: Label printed with timing
        sync_cuda: If True, synchronize CUDA before/after timing
        enabled: If False, timing is skipped
    """
    if not enabled:
        yield
        return

    if sync_cuda:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    start = time.perf_counter()
    yield
    end = time.perf_counter()

    if sync_cuda:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    elapsed = (end - start) * 1000.0
    print(f"[TIMER] {name}: {elapsed:.2f} ms")
