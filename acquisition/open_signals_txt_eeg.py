import json
import re
import numpy as np
from typing import List, Union, Optional
from queue import Empty
from acquisition import Acquisition  # stessa ABC della live

class OpenSignalsTxtEEG(Acquisition):
    """
    Sorgente EEG offline da file .txt OpenSignals (BITalino).
    Mantiene la stessa API della live:
      - read_block(n) -> np.ndarray [C, n]
      - stream(hop)
    """

    def __init__(
        self,
        path: str,
        channels: Union[List[str], List[int]] = ("A1","A2","A3"),  # es. C3,Cz,C4
        volts_per_count: float = 1.0,
        fs: Optional[int] = None,   # se None, prova a leggere dall'header
    ):
        self.path = path
        self.requested = list(channels)
        self.vpc = float(volts_per_count)
        self._fs = fs
        self._X = None     # [C_all, T] (solo le colonne richieste)
        self._idx = 0      # cursore per read_block/stream
        self.meta = {}     # info header utili

        self._load_file()

    @property
    def fs(self) -> int:
        return self._fs

    def _load_file(self):
        # --- 1) leggi header (righe che iniziano con '#')
        header_lines = []
        data_start_pos = 0
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            pos = 0
            for line in f:
                if not line.startswith("#"):
                    # prima riga di dati
                    data_start_pos = pos
                    break
                header_lines.append(line.rstrip("\n"))
                pos = f.tell()

        # estrai il JSON dell’header (riga che inizia con '# {')
        header_json = None
        for h in header_lines:
            if h.lstrip().startswith("# {"):
                try:
                    header_json = json.loads(h.lstrip()[2:].strip())
                except Exception:
                    pass
                break

        if header_json:
            # Il JSON ha come chiave il MAC → prendi il primo oggetto
            dev_key = next(iter(header_json))
            meta = header_json[dev_key]
            self.meta = meta
            # sampling rate
            if self._fs is None:
                sr = meta.get("sampling rate", None)
                if isinstance(sr, int):
                    self._fs = sr
        if self._fs is None:
            # fallback sicuro
            self._fs = 500

        # --- 2) individua indici colonne richieste
        # Esempio header["column"] = ["nSeq","I1","I2","O1","O2","A1","A2","A3","A4","A5","A6"]
        col_names = []
        if header_json:
            dev_key = next(iter(header_json))
            col_names = header_json[dev_key].get("column", []) or []

        # mapping: se l'utente ha passato nomi ("A1","A2","A3"), converti in indici;
        # se ha passato indici, prendili così come sono.
        if col_names and isinstance(self.requested[0], str):
            name_to_idx = {name: i for i, name in enumerate(col_names)}
            try:
                col_idx = [name_to_idx[name] for name in self.requested]
            except KeyError as e:
                raise ValueError(f"Canale non presente nel file: {e}")
        else:
            col_idx = list(map(int, self.requested))

        # --- 3) carica i dati numerici (dalla prima riga non-commento)
        # Il separatore è TAB; alcune righe hanno TAB finali → usa genfromtxt robusto
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(data_start_pos)
            data = np.genfromtxt(
                f,
                delimiter="\t",
                dtype=np.float32,
                autostrip=True
            )
        if data.ndim == 1:
            data = data[None, :]  # 1 sola riga

        # --- 4) estrai solo le colonne richieste e trasponi in [C, T]
        # NB: data ha shape [T, cols]
        try:
            X = data[:, col_idx].T  # [C, T]
        except Exception as e:
            raise ValueError(f"Indice colonne non valido: {e}")

        # --- 5) converti in µV (o scala desiderata) se necessario
        X = X.astype(np.float32) * self.vpc

        # --- 6) salva in memoria e reset cursore
        self._X = X
        self._idx = 0

    # --- consumer API ---
    def read_block(self, n: int, timeout: float = 1.0) -> np.ndarray:
        """
        Ritorna [C, n] dal file, avanzando il cursore.
        Se finiti i dati, solleva queue.Empty per coerenza con la live.
        """
        end = min(self._idx + n, self._X.shape[1])
        if end <= self._idx:
            raise Empty
        blk = self._X[:, self._idx:end]
        self._idx = end
        # pad (raro): se vuoi garantire sempre n campioni
        if blk.shape[1] < n:
            # qui preferisco sollevare Empty per segnalare EOF
            raise Empty
        return blk

    def stream(self, hop: int, timeout: float = 1.0):
        """Generatore di blocchi [C, hop] fino a EOF."""
        while True:
            try:
                yield self.read_block(hop, timeout=timeout)
            except Empty:
                break

    # per simmetria con la live (no-op)
    def connect(self): pass
    def start(self): pass
    def stop(self): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): pass
