from dataclasses import dataclass
from queue import Empty, Full, Queue
import threading
from typing import Callable, List

import cv2
import numpy as np


@dataclass
class PairSaveTask:
    """左右图保存任务。"""

    frame_id: int
    left_path: str
    right_path: str
    left_image: np.ndarray
    right_image: np.ndarray
    image_ext: str
    jpeg_quality: int
    png_compression: int


def normalize_image_ext(image_ext: str) -> str:
    """标准化扩展名，返回不带点的小写后缀。"""
    if image_ext is None:
        return "png"
    ext = image_ext.strip().lower()
    if ext.startswith("."):
        ext = ext[1:]
    if ext not in {"png", "jpg", "jpeg"}:
        raise ValueError("image_ext must be one of: .png, .jpg, .jpeg")
    return ext


def build_imwrite_params(image_ext: str, jpeg_quality: int = 90, png_compression: int = 1) -> List[int]:
    """根据格式生成 OpenCV 编码参数。"""
    ext = image_ext.lower()
    if ext in {"jpg", "jpeg"}:
        q = int(max(1, min(100, jpeg_quality)))
        return [cv2.IMWRITE_JPEG_QUALITY, q]
    if ext == "png":
        c = int(max(0, min(9, png_compression)))
        return [cv2.IMWRITE_PNG_COMPRESSION, c]
    return []


def imwrite_image(path: str, image: np.ndarray, image_ext: str, jpeg_quality: int = 90, png_compression: int = 1) -> bool:
    """按指定编码参数写图。"""
    params = build_imwrite_params(
        image_ext=image_ext,
        jpeg_quality=jpeg_quality,
        png_compression=png_compression,
    )
    return cv2.imwrite(path, image, params)


class AsyncImageSaver:
    """可选异步保存器（仅用于 save_images=True）。

    save_policy:
    - "block": 队列满时阻塞。
    - "drop_oldest": 队列满时丢弃最旧任务。
    """

    def __init__(
        self,
        save_fn: Callable[[PairSaveTask], None],
        on_saved: Callable[[PairSaveTask], None],
        maxsize: int = 256,
        save_policy: str = "block",
    ):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive.")
        if save_policy not in {"block", "drop_oldest"}:
            raise ValueError("save_policy must be 'block' or 'drop_oldest'.")

        self._save_fn = save_fn
        self._on_saved = on_saved
        self._queue: Queue[PairSaveTask] = Queue(maxsize=maxsize)
        self._save_policy = save_policy
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._started = False

        self.saved_pairs = 0
        self.dropped_pairs = 0
        self.failed_pairs = 0

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def start(self):
        """启动后台保存线程。"""
        if not self._started:
            self._thread.start()
            self._started = True

    def submit(self, task: PairSaveTask):
        """提交保存任务到队列。"""
        if self._save_policy == "block":
            self._queue.put(task)
            return

        while True:
            try:
                self._queue.put_nowait(task)
                return
            except Full:
                try:
                    _ = self._queue.get_nowait()
                    self._queue.task_done()
                    self.dropped_pairs += 1
                except Empty:
                    pass

    def flush(self):
        """等待队列中的任务全部完成。"""
        self._queue.join()

    def stop(self):
        """停止后台线程（建议先 flush 再 stop）。"""
        if not self._started:
            return
        self._stop_event.set()
        self._thread.join()
        self._started = False

    def _worker(self):
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                break
            try:
                task = self._queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                self._save_fn(task)
                self.saved_pairs += 1
                self._on_saved(task)
            except Exception as exc:
                self.failed_pairs += 1
                print(f"[WARN] async save failed for frame_id={task.frame_id}: {exc}")
            finally:
                self._queue.task_done()

