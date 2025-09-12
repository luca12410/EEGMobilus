# inference.py
import time
import numpy as np
from collections import deque
from typing import Callable, Optional

from model_interaction.profiles import load_profile
from preprocessing.preprocess import prepare_for_model  # deve fare: preprocess + reshape (1,C,S,1)

import os, logging

DEBUG = True
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

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
        if DEBUG:
            log.debug(f"Loaded profile from {profile_dir}: \n")
            log.debug(f"  fs={self.meta['fs']} Hz, chans={self.meta['chans']}, samples={self.meta['samples']} \n")
            log.debug(f"  classes={self.meta['classes']} \n")

    def predict_window(self, X_win: np.ndarray) -> np.ndarray:
        """
        X_win: ndarray [C, Samples] (grezzo o già filtrato come in calibrazione).
        return: probs ndarray [n_classes]
        """
        
        if DEBUG:
            log.debug(f"Predicting window shape: {X_win.shape}")
            if hasattr(X_win, 'shape'):
                c,w = X_win.shape[0], X_win.shape[-1]
                if w != self.meta["samples"]:
                    log.warning(f"  Mismatch Warning: expected {self.meta['samples']} samples, got {w} \n")
            if hasattr(X_win, 'max'):
                log.debug(f"  Data range: min={X_win.min():.3f}, max={X_win.max():.3f} \n")
        
        
        X_ready = prepare_for_model(
            X_win,
            fs=self.meta["fs"],
            chans=self.meta["chans"],
            samples=self.meta["samples"],
        )
        probs = self.model.predict(X_ready, verbose=0)[0]  # shape [n_classes]
        
        if DEBUG:
            _p = probs.ravel()
            log.debug(f"  Predicted probs: {np.round(_p,3)} (sum={_p.sum():.3f}) \n")
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
    
    hop_sec = max(hop_sec, win_sec / 4)
    
    win = int(round(fs * win_sec))
    hop = int(round(fs * hop_sec))
    rb  = RingBuffer(chans=engine.meta["chans"], win_samples=win)

    t0 = time.time()
    if DEBUG:
        log.debug(f"Running inference stream with win={win} samples ({win_sec:.3f} s), hop={hop} samples ({hop_sec:.3f} s)")
    for block in source.stream(hop):          # block: [C, hop]
        if block.size == 0:
            continue
        if DEBUG:
            log.debug(f"Stream block: shape={getattr(block, 'shape', None)}, len={block.shape[-1] if hasattr(block, 'shape') else None} ")
        rb.push(block)
        X_win = rb.get_window()               # [C, win] oppure None
        
        if X_win is None:
            if DEBUG:
                log.debug("Waiting for window...")
                continue
        if DEBUG:
            log.debug(f"Got window: shape={X_win.shape}")  
        
        if X_win is None:
            continue
        try:
            probs = engine.predict_window(X_win)  # np.array [n_classes]
        except Exception as e:
            log.error(f"Error during prediction: {e}")
            continue
        if on_probs:
            on_probs(probs, (time.time() - t0) * 1000.0)


def _ask(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default

def run_inference_interactive(default_profile_dir: str = "profiles/latest",
                              default_mode: str = "file",
                              test_file: str = "model_interaction/files/campione090824_test.txt",
                              analog_channels = ("A4",),
                              win_sec: float = 1.0,
                              hop_sec: float = 0.5):
    """
    Orchestrazione INTERATTIVA dell'inferenza.
    Chiede QUALE PROFILO usare e se inferire da FILE (A4) o LIVE.
    """
    import os
    import numpy as np
    from acquisition.open_signals_txt_eeg import OpenSignalsTxtEEG
    from acquisition.EEG_live_acquisition import LiveSource
    from robot_control.decision import DecisionSmoother, map_class_to_cmd

    profile_dir = _ask("Path del profilo da usare (directory)", default_profile_dir)
    if not os.path.isdir(profile_dir):
        print(f"[!] Profilo non trovato: {profile_dir}")
        raise SystemExit(1)

    mode = _ask("Sorgente inferenza (file/live)", default_mode).lower()
    engine = InferenceEngine(profile_dir)
    smoother = DecisionSmoother(win=3, thr=0.4, refractory_ms=100)

    if mode.startswith("l"):
        print("[*] Inferenza LIVE dal dispositivo…")
        fs = engine.meta["fs"]
        src = LiveSource(fs)

        def on_probs(p, t_ms):
            print(f"[raw] {t_ms:8.1f} ms | probs={np.round(p,3)}", end="")
            cls_idx = smoother.step(p, t_ms)
            if cls_idx is not None:
                label = engine.meta["classes"][cls_idx]
                cmd   = map_class_to_cmd(cls_idx, engine.meta["classes"])
                print(f"{t_ms:8.1f} ms | probs={np.round(p,3)} | label={label} | cmd={cmd}")
            else:
                if DEBUG:
                    print("[dec] {t_ms:8.1f} ms | nessuna classe trovata")

        run_inference_stream(src, engine, fs=fs, win_sec=win_sec, hop_sec=hop_sec, on_probs=on_probs)
    else:
        print("[*] Inferenza su FILE OpenSignals (solo analogico, nessun marker)…")
        if not os.path.exists(test_file):
            print(f"[!] TEST_FILE non trovato: {test_file}")
            raise SystemExit(1)

        src = OpenSignalsTxtEEG(test_file, channels=list(analog_channels))

        def on_probs(p, t_ms):
            cls_idx = smoother.step(p, t_ms)
            if cls_idx is not None:
                label = engine.meta["classes"][cls_idx]
                cmd   = map_class_to_cmd(cls_idx, engine.meta["classes"])
                print(f"{t_ms:8.1f} ms | probs={np.round(p,3)} | label={label} | cmd={cmd}")

        run_inference_stream(src, engine,
                             fs=engine.meta["fs"],
                             win_sec=win_sec,
                             hop_sec=hop_sec,
                             on_probs=on_probs)

    print("[✓] Inference finita.")