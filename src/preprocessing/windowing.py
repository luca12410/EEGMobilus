import numpy as np
from collections import deque

class RingBuffer:
    """Simple circular bugger for fixed-size windowing."""
    def __init__(self, channels: int, win_size: int):
        self.C, self.N = channels, win_size
        self.buf = deque(maxlen=win_size)

    def push(self, block: np.ndarray):
        """Add element"""
        for i in range(block.shape[1]):
            self.buf.append(block[:, i])

    def get_window(self) -> np.ndarray | None:
        """Return window or None if not full."""
        if len(self.buf) < self.N:
            return None
        return np.stack(self.buf, axis=1)  

def sliding_windows(X: np.ndarray, win_size: int, hop: int):
    """Divides array."""
    T = X.shape[1]
    for start in range(0, T - win_size + 1, hop):
        yield X[:, start:start+win_size]
