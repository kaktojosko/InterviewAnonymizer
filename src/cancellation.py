import subprocess
import threading
from typing import Callable


class TaskCancelledError(Exception):
    """Raised when an active video/audio/transcription job is cancelled by the user."""


class CancellationToken:
    """
    Thread-safe cancellation coordinator for a processing job.
    Tracks active subprocesses (e.g. FFmpeg) and worker-pool abort callbacks so
    that cancelling a job immediately stops CPU, GPU, and I/O work.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._procs: set[subprocess.Popen] = set()
        self._callbacks: list[Callable[[], None]] = []

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise TaskCancelledError("Обработка прервана пользователем.")

    def register_proc(self, proc: subprocess.Popen) -> None:
        terminate_now = False
        with self._lock:
            if self._event.is_set():
                terminate_now = True
            else:
                self._procs.add(proc)
        if terminate_now:
            self._terminate_proc(proc)
            raise TaskCancelledError("Обработка прервана пользователем.")

    def unregister_proc(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.discard(proc)

    def register_cancel_callback(self, cb: Callable[[], None]) -> None:
        run_now = False
        with self._lock:
            if self._event.is_set():
                run_now = True
            else:
                self._callbacks.append(cb)
        if run_now:
            try:
                cb()
            except Exception:
                pass

    def cancel(self) -> bool:
        """
        Triggers cancellation once. Returns True if this call triggered it,
        or False if the token was already cancelled.
        """
        with self._lock:
            if self._event.is_set():
                return False
            self._event.set()
            procs = list(self._procs)
            self._procs.clear()
            callbacks = list(self._callbacks)
            self._callbacks.clear()

        for proc in procs:
            self._terminate_proc(proc)

        for cb in callbacks:
            try:
                cb()
            except Exception:
                pass

        return True

    @staticmethod
    def _terminate_proc(proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
