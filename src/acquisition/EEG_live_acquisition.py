import numpy as np
import threading, logging
from queue import Queue, Empty
from bitalino import BITalino
from .acquisition import Acquisition

import os, logging

DEBUG = os.getenv("EEG_DEBUG", "0") == "1"
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

class EEGLiveAcquisition(Acquisition):
    """
    Sorgente EEG live da BITalino (pull-based + thread producer).
    - start() / stop()
    - read_block(n) -> np.ndarray [C, n]
    - stream(hop) -> generator di blocchi [C, hop]
    """
    def __init__(self, mac_address: str, sampling_rate: int = 500,
                 channels = [0,1,2], buffer_size: int = 250,
                 queue_max: int = 20, volts_per_count: float = 1.0):
        self.mac = mac_address
        self.fs = sampling_rate
        self.channels = list(channels)
        self.buf = buffer_size
        self.vpc = volts_per_count
        self.q = Queue(maxsize=queue_max)
        self.dev = None
        self._stop = threading.Event()
        self._th = None
        logging.basicConfig(level=logging.INFO)

    # --- lifecycle ---
    def connect(self):
        self.dev = BITalino(self.mac)
        self.dev.start(self.fs, self.channels)
        logging.info("BITalino connected @ %d Hz (channels=%s)", self.fs, self.channels)


    # --- start lifecycle ---
    def start(self):
        if self.dev is None:
            raise RuntimeError("Device not connected. Call connect() first.")
        self._stop.clear()
        self._th = threading.Thread(target=self._producer, daemon=True)
        self._th.start()

    # --- context manager (comodo nei test/usage) ---
    def __enter__(self):
        self.connect()
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def _producer(self):
        try:
            while not self._stop.is_set():
                frames = self.dev.read(self.buf)
                if frames is None or len(frames) == 0:
                    continue
                frames = np.asarray(frames)
                if frames.ndim != 2 or frames.shape[1] < len(self.channels):
                    logging.warning("Unexpected frame shape: %s", frames.shape)
                    continue
                A = frames[:, -len(self.channels):].T.astype(np.float32) * self.vpc  # [C,n]
                if self.q.full():
                    try: self.q.get_nowait()
                    except Empty: pass
                self.q.put_nowait(A)
        except Exception:
            logging.exception("Producer error")
            self._stop.set()

    def stop(self):
        self._stop.set()
        if self._th: self._th.join(timeout=1.0); self._th = None
        # flush queue
        while not self.q.empty():
            try: self.q.get_nowait()
            except Empty: break
        if self.dev:
            try: self.dev.stop()
            finally:
                self.dev.close(); self.dev = None
        logging.info("Acquisition stopped and device closed.")

    def stream(self, hop: int, timeout: float = 1.0):
        while not self._stop.is_set():
            try:
                yield self.read_block(hop, timeout=timeout)
            except Empty:
                if self._stop.is_set(): break
                continue
            
    
        
    # --- consumer API ---
    def read_block(self, n: int, timeout: float = 1.0) -> np.ndarray:
        """Ritorna un blocco [C, n] (accoda pezzi se serve)."""
        chunks = []
        got = 0
        while got < n:
            A = self.q.get(timeout=timeout)
            chunks.append(A); got += A.shape[1]
        X = np.concatenate(chunks, axis=1)
        if X.shape[1] > n:              # taglia se abbiamo preso troppo
            X = X[:, :n]
        return X

    # ===== Helper e default per usare facilmente l'acquisizione live =====
import os

# Imposta qui MAC e canale di default (A4 = indice 3 su BITalino: A1..A6 -> 0..5)
DEFAULT_MAC       = os.getenv("BITALINO_MAC", "BC:33:AC:AB:AF:54")
DEFAULT_CHANNELS  = [3]   # A4
DEFAULT_FS        = 100
DEFAULT_VPC       = 1.0   # scala ADC -> volt, metti il fattore giusto se ti serve

# Singleton semplice per riutilizzare la connessione
_live_singleton = None

def _get_live(fs: int = DEFAULT_FS, channels = DEFAULT_CHANNELS, mac: str = DEFAULT_MAC, vpc: float = DEFAULT_VPC) -> EEGLiveAcquisition:
    global _live_singleton
    if _live_singleton is None:
        _live_singleton = EEGLiveAcquisition(
            mac_address=mac,
            sampling_rate=fs,
            channels=channels,
            buffer_size=max(1, int(0.25 * fs)),
            volts_per_count=vpc
        )
        _live_singleton.connect()
        _live_singleton.start()
    return _live_singleton

def acquire_chunk_seconds(seconds: float, fs: int) -> np.ndarray:
    """
    Ritorna un blocco [C, T] acquisito in ~tempo reale per 'seconds' secondi a 'fs' Hz.
    Usata da calibrate_guided_live(...).
    """
    n = int(seconds * fs)
    live = _get_live(fs=fs)
    return live.read_block(n)

class LiveSource:
    """
    Sorgente compatibile con run_inference_stream(...).
    Emette blocchi [C, hop] da EEGLiveAcquisition.
    """
    def __init__(self, fs: int = DEFAULT_FS, channels = DEFAULT_CHANNELS, mac: str = DEFAULT_MAC):
        self.fs = fs
        self._live = _get_live(fs=fs, channels=channels, mac=mac)

    def stream(self, hop: int, timeout: float = 1.0):
        if DEBUG:
            log.debug(f"LiveSource stream called with hop={hop}, timeout={timeout}")
        yield from self._live.stream(hop, timeout=timeout)

    def __del__(self):
        # opzionale: non chiudiamo aggressivamente per riuso
        pass