import collections, json
import numpy as np
from typing import Optional, List, Dict
from robot_control.commands import RobotCommand, CMD_MOVE, CMD_TURN, CMD_STOP

class DecisionSmoother:
    """
    Robot decision smoother. Given a stream of class probabilities (n_cls),
    it applies temporal smoothing, hysteresis, thresholding, voting, and refractory period
    to produce a stable sequence of class indices (or None).
    It can be configured to consider only certain classes as "mapped" (i.e., having associated commands).
    """
    def __init__(self,
                 win: int = 3,
                 thr: float = 0.55,
                 margin: float = 0.05,
                 min_votes: int = 3,
                 refractory_ms: int = 250,  
                 hysteresis: float = 0.10,
                 debug: bool = False,
                 is_mapped=None   
                 ):
        import collections
        self.buf = collections.deque(maxlen=int(win))
        self.thr = float(thr)
        self.margin = float(margin)
        self.min_votes = int(min_votes)
        self.refractory_ms = int(refractory_ms)
        self.hysteresis = float(hysteresis)
        self.debug = debug

        self.last_switch_ms = -1e12
        self.current_idx = None      

        self.is_mapped = (lambda _: True) if is_mapped is None else is_mapped

    def reset(self):
        self.buf.clear()
        self.last_switch_ms = -1e12
        self.current_idx = None

    def _pick_candidate(self, xwin: np.ndarray):
        M = np.stack(xwin, axis=0)        # [win, n_cls]
        p = M.mean(axis=0)
        k1 = int(np.argmax(p)); s1 = float(p[k1])
        if p.size > 1:
            pp = p.copy(); pp[k1] = -np.inf
            s2 = float(np.max(pp))
        else:
            s2 = 0.0
        votes = int(np.sum(np.argmax(M, axis=1) == k1))

        thr_eff = self.thr - (self.hysteresis if self.current_idx == k1 else 0.0)

        if s1 >= 0.90 and (s1 - s2) >= 0.10 and votes >= 1:
            return k1, f"FAST k={k1} s1={s1:.3f} s2={s2:.3f} v={votes}"

        if s1 >= thr_eff and (s1 - s2) >= self.margin and votes >= self.min_votes:
            return k1, f"OK k={k1} s1={s1:.3f} s2={s2:.3f} v={votes}"

        if self.debug:
            if s1 < thr_eff: print(f"[SMOOTH] below thr s1={s1:.3f} < {thr_eff:.3f}")
            elif (s1 - s2) < self.margin: print(f"[SMOOTH] low margin {s1-s2:.3f} < {self.margin:.3f}")
            elif votes < self.min_votes: print(f"[SMOOTH] low votes {votes} < {self.min_votes}")
        return None, None

    def step(self, probs: np.ndarray, now_ms: float) -> Optional[int]:
        x = np.asarray(probs, dtype=float).ravel()
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            if self.debug: print("[SMOOTH] invalid input")
            return self.current_idx  

        self.buf.append(x)
        if len(self.buf) < max(2, self.min_votes):
            if self.debug: print("[SMOOTH] warmup")
            return self.current_idx 

        cand, reason = self._pick_candidate(self.buf)

        if cand is None:
            if self.current_idx is not None and self.debug:
                print(f"[SMOOTH] HOLD (no candidate) k={self.current_idx}")
            return self.current_idx

        if cand == self.current_idx:
            if self.debug: print(f"[SMOOTH] HOLD (same) k={self.current_idx} {reason}")
            return self.current_idx

        if not self.is_mapped(cand):
            if self.current_idx is not None and self.debug:
                print(f"[SMOOTH] IGNORE unmapped cand={cand} -> keep k={self.current_idx}")
            return self.current_idx

        if (now_ms - self.last_switch_ms) < self.refractory_ms:
            if self.debug: print("[SMOOTH] refractory (switch)")
            return self.current_idx

        self.current_idx = cand
        self.last_switch_ms = now_ms
        if self.debug: print(f"[SMOOTH] SWITCH→ k={cand} {reason}")
        return self.current_idx

