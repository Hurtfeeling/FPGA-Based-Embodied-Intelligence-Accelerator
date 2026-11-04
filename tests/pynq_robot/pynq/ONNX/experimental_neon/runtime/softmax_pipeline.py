"""Optional CPU-copy/PL Softmax pipeline using an existing CMA buffer pair.

One Softmax IP runs asynchronously while the caller prepares its next input
and copies the preceding result. No worker thread or additional CMA is used.
Both PYNQ buffers must synchronize only the contiguous view on which sync_*
is called; check that property on the board before selecting this path.
"""
import time

import numpy as np


def run_softmax_pipeline(control, source, destination, x, out, timeout_s=30.0,
                         minimum_elements=None):
    """Run FP32 last-axis Softmax and return task/timing statistics.

    ``source`` and ``destination`` are caller-owned, disjoint, live CMA
    buffers, each at least 32 FP32 elements. They are split into two slots.
    Calls using this pair must be serialized, including against any other
    operator that shares the underlying scratch arena.

    The default uses the original full-buffer sequential path when a tensor
    fits in that buffer: splitting such calls would add a hardware task.
    ``minimum_elements=0`` forces the split path for benchmark comparisons.
    Timings of CPU copy/sync and submit/wait can overlap PL execution; they
    must not be added to infer a hardware execution time.
    """
    if (x.dtype != np.float32 or out.dtype != np.float32 or
            x.shape != out.shape or x.ndim < 1 or not x.shape[-1]):
        raise ValueError('Softmax requires matching nonempty FP32 tensors')
    if not x.flags.c_contiguous or not out.flags.c_contiguous:
        raise ValueError('Softmax pipeline requires contiguous host tensors')
    if np.shares_memory(x, out):
        raise ValueError('Softmax pipeline host input and output must be disjoint')
    for buffer in (source, destination):
        if (buffer.dtype != np.float32 or not buffer.flags.c_contiguous or
                not hasattr(buffer, 'device_address') or
                getattr(buffer, 'freed', False)):
            raise ValueError('Live contiguous FP32 CMA buffers required')
        if int(buffer.device_address) % 64:
            raise ValueError('Pipeline CMA base addresses must be cache-line aligned')
    if np.shares_memory(source, destination):
        raise ValueError('Softmax source and destination CMA must be disjoint')
    if any(np.shares_memory(host, buffer) for host in (x, out)
           for buffer in (source, destination)):
        raise ValueError('Host tensors must not alias the pipeline CMA buffers')
    k = int(x.shape[-1])
    if not 1 <= k <= 4096:
        raise ValueError('Softmax K outside hardware limit')
    stride = (k + 3) // 4 * 4
    # Split at a CPU cache-line boundary, not merely the IP's 16-byte boundary.
    # PYNQ cache maintenance may round start/end to cache lines.
    slot_elements = (min(source.size, destination.size) // 2) // 16 * 16
    full_rows = min(65536, source.size // stride, destination.size // stride)
    slot_rows = min(65536, slot_elements // stride)
    if not full_rows:
        raise ValueError('Softmax row does not fit the CMA staging buffers')
    threshold = min(source.size, destination.size) if minimum_elements is None else int(minimum_elements)
    if threshold < 0:
        raise ValueError('minimum_elements must be nonnegative')
    use_pipeline = slot_rows > 0 and int(x.size) > threshold
    rows_per_task = slot_rows if use_pipeline else full_rows
    xx, yy = x.reshape(-1, k), out.reshape(-1, k)
    records = dict(tasks=0, pipelined=use_pipeline,
                   rows_per_task=int(rows_per_task), input_copy_sync_seconds=0.0,
                   submit_seconds=0.0, wait_seconds=0.0,
                   output_copy_sync_seconds=0.0, seconds=0.0)
    begin = time.perf_counter()

    def prepare(first, slot):
        nr = min(rows_per_task, len(xx) - first)
        offset = slot * slot_elements if use_pipeline else 0
        si = source[offset:offset + nr * stride].reshape(nr, stride)
        so = destination[offset:offset + nr * stride].reshape(nr, stride)
        started = time.perf_counter()
        # Padding is outside the hardware K and need not be cleared. Avoid
        # dirtying destination cache lines that PL will later overwrite.
        np.copyto(si[:, :k], xx[first:first + nr])
        si.sync_to_device()
        records['input_copy_sync_seconds'] += time.perf_counter() - started
        return first, nr, si, so

    def submit(block):
        first, nr, si, so = block
        started = time.perf_counter()
        task = control.submit_softmax(int(si.device_address), int(so.device_address), nr, k)
        records['submit_seconds'] += time.perf_counter() - started
        records['tasks'] += 1
        return task

    def wait(task):
        started = time.perf_counter()
        # A timeout retains the task and its buffers in the driver's pending
        # collection. The caller must drain it before reusing/freeing CMA.
        task.wait(timeout_s=timeout_s)
        records['wait_seconds'] += time.perf_counter() - started

    def drain(block):
        first, nr, si, so = block
        started = time.perf_counter()
        so.sync_from_device()
        np.copyto(yy[first:first + nr], so[:, :k])
        records['output_copy_sync_seconds'] += time.perf_counter() - started

    if len(xx):
        current = prepare(0, 0)
        task = submit(current)
        first, slot = current[1], 1
        if use_pipeline:
            while first < len(xx):
                # CPU copies to the other slot while PL reads the current one.
                following = prepare(first, slot)
                wait(task)
                # Start PL immediately, then consume the preceding output.
                task = submit(following)
                drain(current)
                current = following
                first += following[1]
                slot ^= 1
            wait(task)
            drain(current)
        else:
            wait(task)
            drain(current)
            while first < len(xx):
                current = prepare(first, 0)
                task = submit(current)
                wait(task)
                drain(current)
                first += current[1]
    records['seconds'] = time.perf_counter() - begin
    return records
