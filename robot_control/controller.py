# control/controller.py
import json, os
from typing import Optional
from robot_control.decision import DecisionSmoother          # già nel repo :contentReference[oaicite:3]{index=3}
from robot_control.commands import RobotCommand
from robot_control.router import RobotRouter
import numpy as np

def _load_mapping(mapping_path: Optional[str], profile_dir: Optional[str]) -> dict:
    # 1) mapping esplicito; 2) mapping nel profilo; 3) fallback vuoto (STOP)
    for p in [mapping_path,
              os.path.join(profile_dir, "labels_to_commands.json") if profile_dir else None]:
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}

def _label_to_cmd(label: str, mapping: dict) -> RobotCommand:
    spec   = mapping.get(label, {"name": "STOP", "params": {}})
    return RobotCommand(name=spec.get("name","STOP"),
                        params=spec.get("params",{}),
                        target=spec.get("target"))

def run_controller(bus, classes, router: RobotRouter,
                   default_robot: Optional[str] = None,
                   mapping_path: Optional[str] = "config/labels_to_commands.json",
                   profile_dir: Optional[str] = None,
                   win:int=3, thr:float=0.6, refractory_ms:int=200):
    """
    SINGLE ENTRYPOINT: consuma eventi, smoother → label → mapping → router.
    """
    mapping = _load_mapping(mapping_path, profile_dir)
    smoother = DecisionSmoother(win=win, thr=thr, refractory_ms=refractory_ms)

    while True:
        ev = bus.get()                  # blocca finché arriva un evento
        idx = smoother.step(ev.probs, ev.t_ms)
        # DEBUG:
        print(f"[ctrl] event t={ev.t_ms:.1f} ms | probs={np.round(ev.probs,3)}")
        if idx is None:
            continue
        label = classes[idx]
        cmd = _label_to_cmd(label, mapping)
        if cmd is None:
            print(f"[ctrl] label '{label}' NON mappata -> skip")
            continue
        print(f"[ctrl] decision label={label} -> {cmd.name} {cmd.params}")
        router.dispatch(cmd, default_robot=default_robot)
