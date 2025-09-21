# robot_control/decision.py
import collections, json
import numpy as np
from typing import Optional, List, Dict
from robot_control.commands import RobotCommand, CMD_MOVE, CMD_TURN, CMD_STOP

class DecisionSmoother:
    """
    Media mobile + coerenza (min_votes) + margine + isteresi + fast-path
    + EDGE/LATCH con gap: emette solo sul fronte di salita, mantiene il comando
      finché non arriva una nuova decisione (classe diversa) oppure la classe
      scompare per almeno min_gap_ms e poi riappare (anche la stessa).
    """
    def __init__(self,
                 win: int = 3,
                 thr: float = 0.6,
                 margin: float = 0.05,
                 min_votes: int = 2,
                 refractory_ms: int = 250,
                 hysteresis: float = 0.10,
                 min_gap_ms: int = 500,      # tempo di "silenzio" per chiudere il batch
                 debug: bool = True):
        import collections
        self.buf = collections.deque(maxlen=int(win))
        self.thr = float(thr)
        self.margin = float(margin)
        self.min_votes = int(min_votes)
        self.refractory_ms = int(refractory_ms)
        self.hysteresis = float(hysteresis)
        self.min_gap_ms = int(min_gap_ms)
        self.last_ms = -1e12
        self.last_idx = None
        self.debug = debug

        # Latch edge-based
        self.latched_idx = None
        self.last_emit_ms = -1e12
        self.last_cand_seen_ms = -1e12  # ultimo istante in cui c’era un candidato valido

    def reset(self):
        self.buf.clear()
        self.last_ms = -1e12
        self.last_idx = None
        self.latched_idx = None
        self.last_emit_ms = -1e12
        self.last_cand_seen_ms = -1e12

    def step(self, probs: np.ndarray, now_ms: float) -> Optional[int]:
        x = np.asarray(probs, dtype=float).ravel()
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            if self.debug: print("[smooth] invalid input")
            return None

        # Refractory: aggiorna buffer ma non decide
        if (now_ms - self.last_ms) < self.refractory_ms:
            if self.debug: print("[smooth] refractory")
            self.buf.append(x)
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

        # Valuta candidato (come prima)
        candidate = None
        reason = None
        if s1 >= 0.90 and (s1 - s2) >= 0.10 and votes >= 1:
            candidate = k1
            reason = f"FAST k={k1} s1={s1:.3f} s2={s2:.3f} votes={votes}"
        elif s1 >= thr_eff and (s1 - s2) >= self.margin and votes >= self.min_votes:
            candidate = k1
            reason = f"OK k={k1} s1={s1:.3f} s2={s2:.3f} votes={votes}"
        else:
            if self.debug:
                if s1 < thr_eff: print(f"[smooth] below thr s1={s1:.3f} < {thr_eff:.3f}")
                elif (s1 - s2) < self.margin: print(f"[smooth] low margin {s1-s2:.3f} < {self.margin:.3f}")
                elif votes < self.min_votes: print(f"[smooth] low votes {votes} < {self.min_votes}")

        # ---- EDGE/LATCH con gap ----
        if self.latched_idx is None:
            # Nessun batch attivo → emetti SOLO su fronte di salita
            if candidate is not None:
                self.last_idx = candidate
                self.last_ms = now_ms
                self.last_emit_ms = now_ms
                self.latched_idx = candidate
                self.last_cand_seen_ms = now_ms
                if self.debug: print(f"[smooth] {reason} LATCH")
                return candidate
            return None

        # Batch attivo: mantieni finché non arriva un nuovo comando
        if candidate is not None:
            self.last_cand_seen_ms = now_ms
            if candidate == self.latched_idx:
                # stessa classe → mantiene, non ri-emette
                if self.debug: print(f"[smooth] hold k={self.latched_idx}")
                return None
            # classe diversa → SWITCH immediato (nuovo comando)
            self.last_idx = candidate
            self.last_ms = now_ms
            self.last_emit_ms = now_ms
            self.latched_idx = candidate
            if self.debug: print(f"[smooth] {reason} SWITCH")
            return candidate

        # Nessun candidato: se silenzio abbastanza lungo, rilascia
        gap = now_ms - self.last_cand_seen_ms
        if gap >= self.min_gap_ms:
            if self.debug: print(f"[smooth] release (gap {gap:.0f} ms) k={self.latched_idx}")
            self.latched_idx = None
        else:
            if self.debug: print(f"[smooth] hold (no-cand) gap={gap:.0f} ms < {self.min_gap_ms} ms")
        return None

