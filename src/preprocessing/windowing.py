import numpy as np
from collections import deque

class RingBuffer:
    """Mantiene un buffer circolare di forma [C, N]."""
    def __init__(self, channels: int, win_size: int):
        self.C, self.N = channels, win_size
        self.buf = deque(maxlen=win_size)

    def push(self, block: np.ndarray):
        """Aggiunge un blocco [C, n]."""
        for i in range(block.shape[1]):
            self.buf.append(block[:, i])

    def get_window(self) -> np.ndarray | None:
        """Ritorna finestra [C, N] o None se non piena."""
        if len(self.buf) < self.N:
            return None
        return np.stack(self.buf, axis=1)  # [C, N]

def sliding_windows(X: np.ndarray, win_size: int, hop: int):
    """Divide array [C, T] in finestre [C, win_size] con hop."""
    T = X.shape[1]
    for start in range(0, T - win_size + 1, hop):
        yield X[:, start:start+win_size]
