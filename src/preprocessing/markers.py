# markers.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import json
import numpy as np

@dataclass
class Marker:
    t_ms: float        
    label: str         

class MarkerStream:
    def __init__(self, fs: int):
        self.fs = int(fs)
        self._markers: List[Marker] = []

    # --- add ---
    def add_ms(self, t_ms: float, label: str) -> None:
        self._markers.append(Marker(float(t_ms), label))

    def add_sample(self, t_sample: int, label: str) -> None:
        self.add_ms(self.samples_to_ms(t_sample), label)

    # --- converters ---
    def ms_to_samples(self, t_ms: float) -> int:
        return int(round(t_ms * self.fs / 1000.0))

    def samples_to_ms(self, t_samples: int) -> float:
        return (t_samples * 1000.0) / self.fs

    # --- access ---
    def markers(self) -> List[Marker]:
        return list(self._markers)

    # --- extract trials ---
    def trials(self,
               window_sec: Tuple[float, float] = (0.5, 3.5),
               labels_whitelist: Optional[List[str]] = None) -> List[Tuple[int, int, str]]:
        """
        Returns list of (start_sample, end_sample, label) for each marker.
        window_sec: (pre, post) in seconds relative to marker time.
        """
        pre_s, post_s = window_sec
        trials = []
        for m in self._markers:
            if labels_whitelist and m.label not in labels_whitelist:
                continue
            t0 = self.ms_to_samples(m.t_ms + pre_s * 1000.0)
            t1 = self.ms_to_samples(m.t_ms + post_s * 1000.0)
            if t1 > t0:
                trials.append((t0, t1, m.label))
        return trials

    # --- IO (JSON) ---
    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"fs": self.fs,
                       "markers": [asdict(m) for m in self._markers]}, f, indent=2)

    @staticmethod
    def load_json(path: str) -> "MarkerStream":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        ms = MarkerStream(obj["fs"])
        for m in obj.get("markers", []):
            ms.add_ms(m["t_ms"], m["label"])
        return ms

    # --- parsing from digital columns (0/1) ---
    @staticmethod
    def from_digital_columns(fs: int,
                             digital_matrix,                    
                             col_map: Dict[str, int],
                             label_map: Dict[str, str],
                             rising: bool = True,
                             min_separation_ms: float = 250.0) -> "MarkerStream":
        """
        Extract markers from digital columns (0/1) matrix.
        Params:
        - col_map: {'O1': idx, 'O2': idx}
        - label_map: {'O1': 'left', 'O2': 'right'}
        """
        X = np.asarray(digital_matrix)
        if X.ndim != 2:
            raise ValueError("digital_matrix must be 2D")
        if X.shape[0] < X.shape[1] and max(col_map.values()) < X.shape[1]:
            X = X.T

        ms = MarkerStream(fs)
        last_onset_ms: Dict[str, float] = {k: -1e9 for k in col_map.keys()}
        for ch_name, ch_idx in col_map.items():
            sig = (X[ch_idx] > 0).astype(int)
            diff = np.diff(sig, prepend=sig[0])
            edges = np.where(diff == (1 if rising else -1))[0]
            for t_samp in edges:
                t_ms = ms.samples_to_ms(int(t_samp))
                if t_ms - last_onset_ms[ch_name] >= min_separation_ms:
                    lbl = label_map.get(ch_name, ch_name)
                    ms.add_ms(t_ms, lbl)
                    last_onset_ms[ch_name] = t_ms
        return ms


# -------- Helper for OpenSignals file. --------
def read_opensignals_digital(path: str, wanted: list[str]):
    """
    Extracts digital columns from OpenSignals .txt file.
    Returns: digital_matrix [cols, T], col_map {name: idx}, fs
    """
    import json
    import numpy as np

    # --- HEADER READ ---
    header_lines = []
    data_start_pos = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            pos = f.tell()            
            line = f.readline()
            if not line:              
                break
            if not line.startswith("#"):
                data_start_pos = pos  
                break
            header_lines.append(line.rstrip("\n"))

    # --- parse header JSON OpenSignals ---
    header_json = None
    for h in header_lines:
        if h.lstrip().startswith("# {"):
            header_json = json.loads(h.lstrip()[2:].strip())
            break
    if not header_json:
        raise ValueError("Header JSON OpenSignals non trovato")

    dev_key = next(iter(header_json))
    meta = header_json[dev_key]
    fs = int(meta.get("sampling rate", 500))
    col_names = meta.get("column", [])
    name_to_idx = {name: i for i, name in enumerate(col_names)}

    try:
        idxs = [name_to_idx[n] for n in wanted]
    except KeyError as e:
        missing = [n for n in wanted if n not in name_to_idx]
        raise KeyError(f"Colonne richieste non trovate nel file: {missing}. "
                       f"Presenti: {list(name_to_idx.keys())}") from e

    # --- LOAD DATA ---
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(data_start_pos)
        data = np.genfromtxt(f, delimiter="\t", dtype=np.float32, autostrip=True)

    if data.ndim == 1:
        data = data[None, :]

    digital = data[:, idxs].T
    col_map = {n: i for i, n in enumerate(wanted)}
    return digital, col_map, fs
