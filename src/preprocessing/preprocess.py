import numpy as np
from scipy.signal import butter, lfilter
import os, logging

DEBUG = os.getenv("EEG_DEBUG", "0") == "1"
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def bandpass_filter(data: np.ndarray, fs: int, low: float=1.0, high: float=40.0, order: int=5):
    """Bandpass filter"""
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return lfilter(b, a, data, axis=1)

def notch_filter(data: np.ndarray, fs: int, freq: float=50.0, Q: float=30.0):
    """Simple notch filter."""
    from scipy.signal import iirnotch
    b, a = iirnotch(freq/(fs/2), Q)
    return lfilter(b, a, data, axis=1)

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