# control/decision.py
import collections
import numpy as np
from typing import Optional, List

class DecisionSmoother:
    def __init__(self, win:int=3, thr:float=0.6, refractory_ms:int=300):
        self.buf = collections.deque(maxlen=win)
        self.thr = thr
        self.last_ms = -1e9
        self.refractory_ms = refractory_ms

    def step(self, probs:np.ndarray, now_ms:float) -> Optional[int]:
        self.buf.append(probs)
        if (now_ms - self.last_ms) < self.refractory_ms:
            return None
        p = np.mean(self.buf, axis=0)
        cls = int(np.argmax(p))
        if p[cls] < self.thr:
            return None
        self.last_ms = now_ms
        return cls

def map_class_to_cmd(cls:int, classes:List[str]) -> str:
    lbl = classes[cls]
    return {"left":"MOVE_LEFT", "right":"MOVE_RIGHT"}.get(lbl, lbl.upper())
