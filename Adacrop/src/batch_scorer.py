import time
import threading
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Any, List, Tuple, Optional

from PIL import Image

@dataclass
class _Req:
    payload: object
    future: "ScoreFuture"

class ScoreFuture:
    def __init__(self):
        self._evt = threading.Event()
        self._value = None
        self._exc = None

    def set_result(self, v: float):
        self._value = float(v)
        self._evt.set()

    def set_exception(self, e: Exception):
        self._exc = e
        self._evt.set()

    def result(self, timeout: Optional[float] = None) -> float:
        ok = self._evt.wait(timeout=timeout)
        if not ok:
            raise TimeoutError("ScoreFuture timeout")
        if self._exc is not None:
            raise self._exc
        return float(self._value)

class BatchScorer:
    def __init__(self, scorer: Any, batch_size: int = 48, max_wait_time: float = 0.01, queue_size: int = 1024):
        self.scorer = scorer
        self.batch_size = int(batch_size)
        self.max_wait_time = float(max_wait_time)
        self.q: Queue[_Req] = Queue(maxsize=int(queue_size))

        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, name="BatchScorerWorker", daemon=True)
        self._th.start()
        self._flush_count = 0
        self._ema_ms = None

        if not hasattr(self.scorer, "score_batch"):
            raise TypeError("BatchScorer requires scorer.score_batch(payloads)->scores")

    def submit(self, payload: object) -> ScoreFuture:
        fut = ScoreFuture()
        self.q.put(_Req(payload=payload, future=fut))
        return fut

    def close(self):
        self._stop.set()
        self._th.join(timeout=2.0)

    def _loop(self):
        pending: List[_Req] = []
        last_flush = time.perf_counter()

        while not self._stop.is_set():
            timeout = max(0.0, self.max_wait_time - (time.perf_counter() - last_flush))
            try:
                req = self.q.get(timeout=timeout)
                pending.append(req)
            except Empty:
                pass

            now = time.perf_counter()
            should_flush = (len(pending) >= self.batch_size) or (pending and (now - last_flush) >= self.max_wait_time)

            if not should_flush:
                continue

            batch = pending[: self.batch_size]
            pending = pending[self.batch_size :]
            last_flush = now

            try:
                payloads = [r.payload for r in batch]

                t0 = time.perf_counter()
                scores = self.scorer.score_batch(payloads)
                dt_ms = (time.perf_counter() - t0) * 1000.0

                self._flush_count += 1
                if self._ema_ms is None:
                    self._ema_ms = dt_ms
                else:
                    self._ema_ms = 0.9 * self._ema_ms + 0.1 * dt_ms
                
                #if self._flush_count % 200 == 0:
                qsize = None
                if self._flush_count % 200 == 0:
                    try:
                        qsize = self.q.qsize()
                    except Exception:
                        qsize = None
                    print(f"[BatchScorer] flush#{self._flush_count} n={len(payloads)} dt={dt_ms:.1f}ms ema={self._ema_ms:.1f}ms q={qsize}")
                    
                if len(scores) != len(batch):
                    raise RuntimeError(f"score_batch returned {len(scores)} scores for {len(batch)} reqs")
                for r, s in zip(batch, scores):
                    r.future.set_result(float(s))
            except Exception as e:
                for r in batch:
                    r.future.set_exception(e)