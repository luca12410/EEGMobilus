# robot_control/decision.py
import collections, json
import numpy as np
from typing import Optional, List, Dict
from robot_control.commands import RobotCommand, CMD_MOVE, CMD_TURN, CMD_STOP

class DecisionSmoother:
    """
    Media mobile + coerenza (min_votes) + margine + isteresi + fast-path.
    Se debug=True stampa il motivo per cui NON spara.
    """
    def __init__(self,
                 win: int = 3,
                 thr: float = 0.6,
                 margin: float = 0.05,
                 min_votes: int = 2,
                 refractory_ms: int = 250,
                 hysteresis: float = 0.10,
                 debug: bool = True):
        import collections
        self.buf = collections.deque(maxlen=int(win))
        self.thr = float(thr)
        self.margin = float(margin)
        self.min_votes = int(min_votes)
        self.refractory_ms = int(refractory_ms)
        self.hysteresis = float(hysteresis)
        self.last_ms = -1e12
        self.last_idx = None
        self.debug = debug

    def reset(self):
        self.buf.clear()
        self.last_ms = -1e12
        self.last_idx = None

    def step(self, probs: np.ndarray, now_ms: float) -> Optional[int]:
        x = np.asarray(probs, dtype=float).ravel()
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            if self.debug: print("[smooth] invalid input")
            return None

        # refrattario
        if (now_ms - self.last_ms) < self.refractory_ms:
            if self.debug: print("[smooth] refractory")
            self.buf.append(x)  # mantieni comunque il buffer aggiornato
            return None

        self.buf.append(x)
        if len(self.buf) < max(2, self.min_votes):
            if self.debug: print("[smooth] warmup")
            return None

        M = np.stack(self.buf, axis=0)        # [win, n_cls]
        p = M.mean(axis=0)
        k1 = int(np.argmax(p)); s1 = float(p[k1])
        if p.size > 1:
            pp = p.copy(); pp[k1] = -np.inf
            s2 = float(np.max(pp))
        else:
            s2 = 0.0
        votes = int(np.sum(np.argmax(M, axis=1) == k1))
        thr_eff = self.thr - (self.hysteresis if self.last_idx == k1 else 0.0)

        # fast-path: confidenze altissime → spara subito
        if s1 >= 0.90 and (s1 - s2) >= 0.10 and votes >= 1:
            self.last_idx = k1; self.last_ms = now_ms
            if self.debug: print(f"[smooth] FAST k={k1} s1={s1:.3f} s2={s2:.3f} votes={votes}")
            return k1

        # regole normali
        if s1 < thr_eff:
            if self.debug: print(f"[smooth] below thr s1={s1:.3f} < {thr_eff:.3f}")
            return None
        if (s1 - s2) < self.margin:
            if self.debug: print(f"[smooth] low margin {s1-s2:.3f} < {self.margin:.3f}")
            return None
        if votes < self.min_votes:
            if self.debug: print(f"[smooth] low votes {votes} < {self.min_votes}")
            return None

        self.last_idx = k1
        self.last_ms = now_ms
        if self.debug: print(f"[smooth] OK k={k1} s1={s1:.3f} s2={s2:.3f} votes={votes}")
        return k1

