"""Dynamic micro-batching for model inference.

Concurrent requests each carry one short query. Encoding them one at a time pays
the per-call overhead (tokeniser, Python dispatch, small-matrix kernels) once
per query; encoding them together pays it once per batch. The batcher holds a
request for at most `max_wait_ms` waiting for company, then runs whatever has
arrived -- up to `max_batch` items -- in a single model call.

    batcher = MicroBatcher(lambda texts: model.encode(texts), max_batch=32, max_wait_ms=5)
    vec = await batcher.submit("graph neural networks")

The trade-off is explicit: at low load a lone request waits up to `max_wait_ms`
for nothing (it is flushed early only when the batch fills); at high load the
wait is short because the batch fills quickly. The model call runs in a worker
thread so the event loop keeps accepting requests while a batch is in flight.
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass
class _Item:
    payload: Any
    future: asyncio.Future
    enqueued: float


class MicroBatcher:
    def __init__(self, fn: Callable[[list], Sequence], max_batch: int = 32,
                 max_wait_ms: float = 5.0, workers: int = 1):
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self.fn = fn
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.workers = workers
        self._queue: asyncio.Queue[_Item] | None = None
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="batch")
        self._tasks: list[asyncio.Task] = []
        self.batch_sizes: Counter[int] = Counter()
        self.items_processed = 0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._tasks:
            return
        self._queue = asyncio.Queue()
        self._tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        self._pool.shutdown(wait=False)

    # ------------------------------------------------------------- API
    async def submit(self, payload: Any) -> tuple[Any, dict]:
        """Returns (result, info) where info has queue_ms, inference_ms, batch_size."""
        if not self._tasks:
            self.start()
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put(_Item(payload, fut, time.perf_counter()))
        return await fut

    def stats(self) -> dict:
        n = sum(self.batch_sizes.values())
        return {"max_batch": self.max_batch, "max_wait_ms": self.max_wait * 1000,
                "batches": n, "items": self.items_processed,
                "mean_batch_size": round(self.items_processed / n, 2) if n else 0.0,
                "batch_size_histogram": dict(sorted(self.batch_sizes.items()))}

    def reset_stats(self) -> None:
        self.batch_sizes.clear()
        self.items_processed = 0

    # ------------------------------------------------------------- internals
    async def _collect(self) -> list[_Item]:
        first = await self._queue.get()
        batch = [first]
        deadline = time.perf_counter() + self.max_wait
        while len(batch) < self.max_batch:
            # drain whatever is already queued without waiting
            while len(batch) < self.max_batch and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            remaining = deadline - time.perf_counter()
            if len(batch) >= self.max_batch or remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            started = time.perf_counter()
            try:
                results = await loop.run_in_executor(
                    self._pool, self.fn, [it.payload for it in batch])
                if len(results) != len(batch):
                    raise RuntimeError(f"batch fn returned {len(results)} results "
                                       f"for {len(batch)} inputs")
                err = None
            except Exception as e:          # one bad batch must not kill the worker
                results, err = None, e
            done = time.perf_counter()
            self.batch_sizes[len(batch)] += 1
            self.items_processed += len(batch)
            for i, it in enumerate(batch):
                if it.future.done():        # caller went away (timeout / cancel)
                    continue
                if err is not None:
                    it.future.set_exception(err)
                else:
                    it.future.set_result((results[i], {
                        "queue_ms": (started - it.enqueued) * 1000,
                        "inference_ms": (done - started) * 1000,
                        "batch_size": len(batch)}))
