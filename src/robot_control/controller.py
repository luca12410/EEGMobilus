import json, os
from typing import Optional
from collections import defaultdict
import numpy as np

from robot_control.decision import DecisionSmoother
from robot_control.commands import RobotCommand
from robot_control.router import RobotRouter

_RR = defaultdict(int)

def _load_mapping(mapping_path: Optional[str], profile_dir: Optional[str]) -> dict:
    for p in [mapping_path,
              os.path.join(profile_dir, "labels_to_commands.json") if profile_dir else None]:
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}

def _is_mapped(label: str, mapping: dict) -> bool:
    if not mapping:
        return False
    spec = mapping.get(label, None)
    if spec is None:
        return False
    if isinstance(spec, list):
        return len(spec) > 0
    if isinstance(spec, dict):
        return True
    return False

def _pick_cmd_on_edge(label: str, mapping: dict) -> Optional[RobotCommand]:
    spec = mapping.get(label, None)
    if spec is None:
        return None

    if isinstance(spec, dict):
        name   = spec.get("name", "STOP")
        params = spec.get("params", {}) or {}
        target = spec.get("target")
        return RobotCommand(name=name, params=params, target=target)

    if isinstance(spec, list) and len(spec) > 0:
        i = _RR[label] % len(spec)
        _RR[label] += 1
        entry = spec[i] if isinstance(spec[i], dict) else None
        if entry is None:
            return None
        name   = entry.get("name", "STOP")
        params = entry.get("params", {}) or {}
        target = entry.get("target")
        return RobotCommand(name=name, params=params, target=target)

    return None

def run_controller(bus, classes, router: RobotRouter,
                   default_robot: Optional[str] = None,
                   mapping_path: Optional[str] = "config/labels_to_commands.json",
                   profile_dir: Optional[str] = None,
                   win:int=3, thr:float=0.6, refractory_ms:int=200):
    mapping = _load_mapping(mapping_path, profile_dir)
    smoother = DecisionSmoother(win=win, thr=thr, refractory_ms=refractory_ms)

    active_cmd: Optional[RobotCommand] = None     
    last_seen_mapped_label: Optional[str] = None     
    prev_is_mapped: bool = False                      

    while True:
        ev = bus.get() 
        idx = smoother.step(ev.probs, ev.t_ms)
        print(f"[CONTROL] event t={ev.t_ms:.1f} ms | probs={np.round(ev.probs,3)}")

        if idx is not None:
            label = classes[idx]
            curr_is_mapped = _is_mapped(label, mapping)

            is_edge = curr_is_mapped and (not prev_is_mapped or label != last_seen_mapped_label)

            if is_edge:
                new_cmd = _pick_cmd_on_edge(label, mapping)
                if new_cmd is not None:
                    active_cmd = new_cmd
                    last_seen_mapped_label = label
                    print(f"[CONTROL] EDGE MAPPED label=[{label}] -> [{active_cmd.name} {active_cmd.params}]")
                else:
                    print(f"[CONTROL] EDGE label=[{label}] mapping invalid -> [HOLD last]")
            else:
                if curr_is_mapped:
                    print(f"[CONTROL] HOLD mapped label=[{label}] -> keep [{active_cmd.name if active_cmd else 'None'}]")
                else:
                    print(f"[CONTROL] HOLD unmapped label=[{label}] -> keep [{active_cmd.name if active_cmd else 'None'}]")

            prev_is_mapped = curr_is_mapped
        else:
            print(f"[CONTROL] no label -> keep [{active_cmd.name if active_cmd else 'None'}]")

        if active_cmd is not None:
            router.dispatch(active_cmd, default_robot=default_robot)
