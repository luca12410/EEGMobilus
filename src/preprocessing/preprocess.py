import numpy as np
from scipy.signal import butter, lfilter
import os, logging

DEBUG = os.getenv("EEG_DEBUG", "0") == "1"
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def bandpass_filter(data: np.ndarray, fs: int, low: float=1.0, high: float=40.0, order: int=5):
    """Bandpass con guard-rails: clamp automatico in base a fs; se non realizzabile, salta."""
    nyq = 0.5 * fs
    # clamp bordi in Hz
    lo_hz = max(0.01, float(low))              # >= 0.01 Hz
    hi_hz = min(float(high), 0.95 * nyq)       # < Nyquist
    if hi_hz <= lo_hz or hi_hz <= 0.0 or lo_hz >= nyq:
        log.warning(f"[pre] skip bandpass: fs={fs}Hz, requested [{low},{high}]Hz → invalid after clamp [{lo_hz:.3f},{hi_hz:.3f}]Hz")
        return data

    Wn = [lo_hz/nyq, hi_hz/nyq]               # 0< Wn <1
    try:
        b, a = butter(int(order), Wn, btype='band')
        return lfilter(b, a, data, axis=1)
    except Exception as e:
        log.warning(f"[pre] bandpass failed ({e}); returning unfiltered")
        return data


def notch_filter(data: np.ndarray, fs: int, freq: float=50.0, Q: float=30.0):
    """Notch con guard-rails: disattiva se freq >= Nyquist o non valida."""
    from scipy.signal import iirnotch
    nyq = 0.5 * fs
    if freq is None:
        return data
    f0 = float(freq)
    if f0 <= 0.0 or f0 >= 0.95 * nyq:
        log.warning(f"[pre] skip notch: fs={fs}Hz, requested f0={freq}Hz (Nyq={nyq}Hz)")
        return data
    try:
        w0 = f0 / nyq                          # normalizzato (0..1)
        b, a = iirnotch(w0, float(Q))
        return lfilter(b, a, data, axis=1)
    except Exception as e:
        log.warning(f"[pre] notch failed ({e}); returning unfiltered")
        return data

def normalize(data: np.ndarray):
    """Zero mean."""
    return (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

def preprocess_pipeline(X: np.ndarray, fs: int):
    """Base pipeline of the process."""
    X = bandpass_filter(X, fs)
    X = notch_filter(X, fs)
    X = normalize(X)
    return X

def prepare_for_model(X: np.ndarray, fs: int, chans: int, samples: int):
    """
    Preprocessing + reshape for EEGModels.py
    Input:  X [C, N] block (C=channels, N=samples)
    Output: X_ready [1, C, Samples, 1] for Keras EEGNet
    """
    X = preprocess_pipeline(X, fs)
    
    if X.shape[1] > samples:
        X = X[:, :samples]
    elif X.shape[1] < samples:
        pad = samples - X.shape[1]
        X = np.pad(X, ((0,0),(0,pad)), mode='constant')
    
    X = np.expand_dims(X, axis=0)  
    X = np.expand_dims(X, axis=-1)  
    return X