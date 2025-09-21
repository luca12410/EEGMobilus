# control/bus.py
from dataclasses import dataclass
from queue import Queue
import numpy as np

@dataclass
class ProbsEvent:
    probs: np.ndarray  # shape [n_classes]
    t_ms: float

label_bus: Queue[ProbsEvent] = Queue(maxsize=256)
