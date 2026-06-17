"""Async queue uploader for real-time pipeline integration."""

from __future__ import annotations

import queue
import threading
from typing import Dict, Optional

from .models import ImageUploadTask
from .tcp_client import ImageUploadClient


class AsyncImageUploader:
    """Background uploader with bounded queue.

    drop_policy:
    - block: wait when queue is full
    - drop_oldest: drop oldest item to keep real-time behavior
    """

    def __init__(
        self,
        client: ImageUploadClient,
        max_queue_size: int = 100,
        drop_policy: str = "drop_oldest",
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0.")
        if drop_policy not in {"block", "drop_oldest"}:
            raise ValueError("drop_policy must be 'block' or 'drop_oldest'.")

        self.client = client
        self.drop_policy = drop_policy
        self._queue: queue.Queue[ImageUploadTask] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._enqueued = 0
        self._sent = 0
        self._dropped = 0
        self._failed = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="AsyncImageUploader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                break

            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self.client.send_image(task)
                with self._lock:
                    self._sent += 1
            except Exception:
                with self._lock:
                    self._failed += 1
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        """Flush current queue and stop worker."""
        self._stop_event.set()
        self._queue.join()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.client.close()

    def enqueue(self, task: ImageUploadTask) -> bool:
        """Enqueue one task.

        Returns:
        - True: accepted
        - False: dropped
        """
        if self.drop_policy == "block":
            self._queue.put(task, block=True)
            with self._lock:
                self._enqueued += 1
            return True

        # drop_oldest
        try:
            self._queue.put(task, block=False)
            with self._lock:
                self._enqueued += 1
            return True
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            with self._lock:
                self._dropped += 1
            try:
                self._queue.put_nowait(task)
                with self._lock:
                    self._enqueued += 1
                return True
            except queue.Full:
                with self._lock:
                    self._dropped += 1
                return False

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "enqueued": self._enqueued,
                "sent": self._sent,
                "dropped": self._dropped,
                "failed": self._failed,
                "queue_size": self._queue.qsize(),
            }

