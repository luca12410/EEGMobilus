# acquisition.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Optional
import numpy as np

class Acquisition(ABC):
    """Interface for EEG sources."""
    # --- lifecycle ---
    def connect(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

    # --- consumer API ---
    @abstractmethod
    def read_block(self, n: int, timeout: float = 1.0) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def stream(self, hop: int, timeout: float = 1.0) -> Iterable[np.ndarray]:
        raise NotImplementedError

    # --- context manager ---
    def __enter__(self):
        self.connect(); self.start(); return self
    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # --- alias camelCase ---
    def readBlock(self, n: int, timeout: float = 1.0) -> np.ndarray:
        return self.read_block(n, timeout=timeout)
    def readStream(self, hop: int, timeout: float = 1.0) -> Iterable[np.ndarray]:
        return self.stream(hop, timeout=timeout)
