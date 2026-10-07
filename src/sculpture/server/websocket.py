"""Thread -> asyncio fan-out of job events to WebSocket subscribers."""

from __future__ import annotations

import asyncio
import json
import threading


class Broadcaster:
    def __init__(self):
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subs.append((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [(l, s) for (l, s) in self._subs if s is not q]

    def publish(self, message: dict) -> None:
        """Callable from any thread."""
        text = json.dumps(message, default=_default)
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(_put, q, text)
            except RuntimeError:  # loop closed
                self.unsubscribe(q)


def _put(q: asyncio.Queue, text: str) -> None:
    if q.full():  # drop the oldest event rather than blocking the optimizer
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(text)


def _default(o):
    try:
        import numpy as np

        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)
