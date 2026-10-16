"""System-wide Linux MemAvailable watchdog."""
from __future__ import annotations

import os
import threading
import time
import psutil

GIB = 1024 ** 3


class MemoryGuard:
    def __init__(self, hard_floor_gib: float = 2.0, warning_floor_gib: float = 3.0) -> None:
        self.hard_floor_gib = float(hard_floor_gib)
        self.warning_floor_gib = max(float(warning_floor_gib), self.hard_floor_gib)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="qwen38-memory-guard", daemon=True)

    def start(self) -> None:
        available = psutil.virtual_memory().available / GIB
        if available <= self.hard_floor_gib:
            raise MemoryError(
                f"Cannot start: MemAvailable={available:.2f}GiB <= "
                f"hard floor={self.hard_floor_gib:.2f}GiB"
            )
        print(
            f"[RAM GUARD] started | available_gib={available:.2f} | "
            f"warning_gib={self.warning_floor_gib:.2f} | hard_floor_gib={self.hard_floor_gib:.2f}",
            flush=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        warned = False
        while not self._stop.wait(0.25):
            available = psutil.virtual_memory().available / GIB
            if available <= self.hard_floor_gib:
                print(
                    f"[RAM GUARD][FATAL] MemAvailable={available:.2f}GiB crossed "
                    f"the {self.hard_floor_gib:.2f}GiB safety floor. Terminating now.",
                    flush=True,
                )
                os._exit(137)
            if available <= self.warning_floor_gib and not warned:
                print(f"[RAM GUARD][WARN] MemAvailable={available:.2f}GiB", flush=True)
                warned = True
            elif available > self.warning_floor_gib + 0.5:
                warned = False
