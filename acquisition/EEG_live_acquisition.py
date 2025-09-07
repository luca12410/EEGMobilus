import numpy as np
import threading, logging
from queue import Queue, Empty
from bitalino import BITalino
from acquisition import Acquisition

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

