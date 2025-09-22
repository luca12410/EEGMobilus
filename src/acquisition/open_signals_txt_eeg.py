# acquisition/open_signals_txt_eeg.py
from __future__ import annotations
import numpy as np
from .acquisition import Acquisition

class OpenSignalsTxtEEG(Acquisition):
    """
    Loader for OpenSignals .txt EEG files.
    Usage:
        src = OpenSignalsTxtEEG(path="data.txt", channels=["A4"], fs
    """
    def __init__(self, path: str, channels=("A4",), fs: int | None = None, volts_per_count: float = 1.0):
        self.path = path
        self.channels = list(channels)
        self.fs = fs                      
        self.vpc = float(volts_per_count)

        self._X = None                 
        self._pos = 0
        self._eof = False

        self._load_file()

    # --- lifecycle (compat no-op) ---
    def connect(self): ...
    def start(self): ...
    def stop(self): ...

    # --- core ---
    def _load_file(self):
        import json

        header_lines = []
        data_start_pos = 0
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.startswith("#"):
                    data_start_pos = pos
                    break
                header_lines.append(line.rstrip("\n"))

        header_json = None
        for h in header_lines:
            if h.lstrip().startswith("# {"):
                header_json = json.loads(h.lstrip()[2:].strip())
                break
        if not header_json:
            raise ValueError("Header JSON OpenSignals not found")

        dev_key = next(iter(header_json))
        meta = header_json[dev_key]
        fs_file = int(meta.get("sampling rate", 500))
        col_names = meta.get("column", [])
        name_to_idx = {name: i for i, name in enumerate(col_names)}

        try:
            idxs = [name_to_idx[n] for n in self.channels]
        except KeyError as e:
            missing = [n for n in self.channels if n not in name_to_idx]
            raise KeyError(f"Required columns not found: {missing}. Found: {list(name_to_idx.keys())}") from e

        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(data_start_pos)
            data = np.genfromtxt(f, delimiter="\t", dtype=np.float32, autostrip=True)

        if data.ndim == 1:
            data = data[None, :]

        X = data[:, idxs].T.astype(np.float32) * self.vpc 

        if self.fs is None:
            self.fs = fs_file

        self._X = X
        self._pos = 0
        self._eof = False

    # --- consumer API ---
    def read_block(self, n: int, timeout: float = 1.0) -> np.ndarray:
        if self._eof:
            from queue import Empty
            raise Empty  
        end = self._pos + n
        if end >= self._X.shape[1]:
            end = self._X.shape[1]
            self._eof = True
        chunk = self._X[:, self._pos:end]
        self._pos = end
        if chunk.shape[1] < n:
            pad = np.zeros((chunk.shape[0], n - chunk.shape[1]), dtype=chunk.dtype)
            if chunk.shape[1] > 0:
                pad[:] = chunk[:, -1][:, None]
            chunk = np.concatenate([chunk, pad], axis=1)
        return chunk

    def stream(self, hop: int, timeout: float = 1.0):
        from queue import Empty
        while True:
            try:
                yield self.read_block(hop, timeout=timeout)
                if self._eof:
                    break
            except Empty:
                break
