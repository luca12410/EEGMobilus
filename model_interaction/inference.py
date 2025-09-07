# inference.py
import time
import numpy as np
from collections import deque
from typing import Callable, Optional

from profiles import load_profile
from preprocessing.preprocess import prepare_for_model  # deve fare: preprocess + reshape (1,C,S,1)

class InferenceEngine:
    """
    Carica il profilo e fornisce predict su una finestra [C, Samples].
    """
    def __init__(self, profile_dir: str):
        self.model, self.scaler, self.meta = load_profile(profile_dir)
        # chiavi minime attese nel meta
        for k in ("fs", "chans", "samples", "classes"):
            if k not in self.meta:
                raise KeyError(f"Missing meta key: {k}")

    def predict_window(self, X_win: np.ndarray) -> np.ndarray:
        """
        X_win: ndarray [C, Samples] (grezzo o già filtrato come in calibrazione).
        return: probs ndarray [n_classes]
        """
        X_ready = prepare_for_model(
            X_win,
            fs=self.meta["fs"],
            chans=self.meta["chans"],
            samples=self.meta["samples"],
        )
        probs = self.model.predict(X_ready, verbose=0)[0]  # shape [n_classes]
        return probs


# ----------------------------
# Feeder: blocchi -> finestra -> predict
# ----------------------------

class RingBuffer:
    """Buffer circolare semplice: mantiene gli ultimi win_samples campioni per C canali."""
    def __init__(self, chans: int, win_samples: int):
        self.C, self.N = chans, win_samples
        self.buf = deque(maxlen=win_samples)

    def push(self, block: np.ndarray):
        """Aggiunge un blocco [C, n]."""
        assert block.ndim == 2 and block.shape[0] == self.C, f"Expected [{self.C}, n], got {block.shape}"
        for i in range(block.shape[1]):
            self.buf.append(block[:, i])

    def get_window(self) -> Optional[np.ndarray]:
        """Ritorna [C, N] se pieno, altrimenti None."""
        if len(self.buf) < self.N:
            return None
        return np.stack(self.buf, axis=1)  # [C, N]


def run_inference_stream(
    source,                 # deve esporre .stream(hop) che yielda blocchi [C, hop]
    engine: InferenceEngine,
    fs: int,
    win_sec: float = 1.0,
    hop_sec: float = 0.5,
    on_probs: Optional[Callable[[np.ndarray, float], None]] = None,  # callback(probs, t_ms)
):
    """
    Collante live/offline: prende blocchi dal 'source', compone finestre e invia a 'engine'.
    """
    win = int(round(fs * win_sec))
    hop = int(round(fs * hop_sec))
    rb  = RingBuffer(chans=engine.meta["chans"], win_samples=win)

    t0 = time.time()
    for block in source.stream(hop):          # block: [C, hop]
        if block.size == 0:
            continue
        rb.push(block)
        X_win = rb.get_window()               # [C, win] oppure None
        if X_win is None:
            continue
        probs = engine.predict_window(X_win)  # np.array [n_classes]
        if on_probs:
            on_probs(probs, (time.time() - t0) * 1000.0)
