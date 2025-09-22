# EEG MOBILUS ENTRYPOINT
# Author: Luca Tocci, 2025
# Version: 1.0.0
import os, json, threading, numpy as np
import string, time

import model_interaction.inference as inference                     
import model_interaction.profiles  as profiles              

import acquisition.EEG_live_acquisition as live_src     
import acquisition.open_signals_txt_eeg as file_src     

import robot_control.controller as ctrl                     
import robot_control.bus as event_bus                  
import robot_control.router as rtr                     

DEBUG = True
DEFAULT_PROFILE = "profile_store/test_subject"
DEFAULT_TEST_FILE = "model_interaction/files/test_luca_eeg_blink_movement.txt"
DEFAULT_ROBOT = "tb3_1"
DEFAULT_TOPIC = "/cmd_vel"



# ---------------- utilities ----------------
def introduction():
    introduction_message = "EEG MOBILUS - BCI to Robot Control\n"
    msg = ""
    
    for c in introduction_message:
        for cc in string.printable:
            if cc == c:
                msg += cc
                print(msg, flush=True)
                time.sleep(0.002)
                break
            else:
                print(msg + cc, flush=True)
                time.sleep(0.002)
    os.system('cls' if os.name == 'nt' else 'clear')
    print(introduction_message)
    print("Intelligent Systems and Robotics Laboratory exam - Luca Tocci, 288585\n")
    
def _ask(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default

def _ask_float(prompt: str, default: float) -> float:
    s = _ask(prompt, str(default)).replace(",", ".")
    try: return float(s)
    except ValueError: return float(default)
    
def _ask_int(prompt: str, default: int) -> int:
    s = _ask(prompt, str(default)).strip()
    try: return int(s)
    except ValueError: return int(default)

def _configure_mapping(profile_dir: str, classes: list, default_path: str | None = None) -> str:
    """
    Wizard: configure mapping labels->commands.
    Returns the path of the JSON file.
    More than one command may be associated to a label. If so, they will be sent in round-robin.
    If no command is associated to a label, the label is ignored.
    """
    path = default_path or os.path.join(profile_dir, "labels_to_commands.json")
    print(f"\n[*]  Entering movement configuration for profile: {profile_dir}")
    print("    Classi:", classes)

    if os.path.exists(path):
        use = _ask("A mapping already exists. Do you wish to use it? (Y/n)", "Y").lower()
        if use in ("", "y", "yes"):
            print(f"[CFG] Using existing mapping: {path}\n")
            return path

    def _ask_cmd_entry():
        t = _ask("  Command type (STOP/MOVE/TURN/CUSTOM)", "STOP").strip().upper()
        params = {}
        if t == "MOVE":
            vx = _ask_float("    vx (m/s)", 0.25)
            vy = _ask_float("    vy (m/s)", 0.0)
            wz = _ask_float("    wz (rad/s)", 0.0)
            params = {"vx": vx}
            if abs(vy) > 0: params["vy"] = vy
            if abs(wz) > 0: params["wz"] = wz
        elif t == "TURN":
            wz = _ask_float("    wz (rad/s)", 0.8)
            params = {"wz": wz}
        elif t == "CUSTOM":
            raw = _ask("    params JSON (optional)", "{}")
            try: params = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                print("    [WARNING] Invalid JSON, using {}"); params = {}
        else:
            t, params = "STOP", {}

        tgt = _ask("    target robot-id (empty=default/broadcast)", "").strip()
        entry = {"name": t, "params": params}
        if tgt: entry["target"] = tgt
        return entry

    print("\nFor each class enter the number of commands to associate:")
    print("  0 = no action (ignore label)")
    print("  1 = single command")
    print("  N>=2 = round-robin for multiple commands\n")

    mapping: dict = {}
    for lab in classes:
        n = _ask_int(f"How many commands do you wish to associate to '{lab}'? (0..8)", 1)
        n = max(0, min(n, 8))
        if n == 0:
            # ----> In this case, the label is ignored and no key is added.
            continue
        entries = []
        for i in range(1, n+1):
            print(f"  [{lab}] Configure cmd{i}:")
            entries.append(_ask_cmd_entry())
        mapping[lab] = entries[0] if len(entries) == 1 else entries

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"[CFG] Mapping saved in: {path}\n")
    return path

def _build_router():
    """
    Target the correct robot backend by
    configuring the RobotRouter.
    
    If more backends are added, instead of expanding this function,
    if possible, consider adding a UDP listener on port 9999
    which receives JSON-serialized commands and forwards them to the router,
    like in robot_control/backends/udp.py
    (this would also allow remote control over the network).
    """
    router = rtr.RobotRouter()

    # 1) If a UDP target is specified, use it.
    target = os.getenv("EEG_UDP_TARGET")  # default: "127.0.0.1:9999"
    if target:
        host, port = (target.split(":", 1) + ["9999"])[:2]
        from robot_control.backends.udp import UDPClient
        router.register(DEFAULT_ROBOT, UDPClient(host, int(port)))
        print(f"[ROUTER] UDP backend -> {host}:{port}")
        return router

    # 2) Otherwise, try directly writing to ROS2 (if available).
    try:    
        import robot_control.backends.ros2 as ros2_backend
        client = ros2_backend.ROS2CmdVelClient(DEFAULT_ROBOT, topic=DEFAULT_TOPIC)
        router.register(DEFAULT_ROBOT, client)
        print(f"[ROUTER] ROS2 backend su {DEFAULT_TOPIC}")
    except Exception as e:
    # 3) If no other backend is available, use a mock which only prints the selected command.
        class _PrintBackend:
            def send(self, cmd): print(f"[MOCK] {cmd.name} {cmd.params} -> {cmd.target or DEFAULT_ROBOT}")
        router.register(DEFAULT_ROBOT, _PrintBackend())
        print(f"[ROUTER] Activated  (ROS2 unavailable): {e})")
    return router

# ---------------- ENTRYPOINT ----------------
if __name__ == "__main__":
    introduction()
    
    action = _ask("Please, choose an action (inference/calibration)", "calibration").lower()

    if action.startswith("c"):
        # EXECUTE CALIBRATION WIZARD
        import model_interaction.calibrate as calibrate
        calibrate.run_calibration_interactive() 
    else:
        # --- Setup profile and orchestrate inference ---
        profile_dir = _ask("Choose a profile (path or profile name): ", DEFAULT_PROFILE)
        eng = inference.InferenceEngine(profile_dir)                     
        fs = int(eng.meta["fs"])
        win_sec = eng.meta["samples"] / fs
        hop_sec = max(0.05, win_sec / 4)

        # --- Source selection  ---
        mode = _ask("Please choose an inference source (file/live)", "file").lower()
        if mode.startswith("l"):
            source = live_src.LiveSource(fs)  
        else:
            test_file = _ask("Please indicate an OpenSignals file", DEFAULT_TEST_FILE)
            source = file_src.OpenSignalsTxtEEG(test_file, channels=("A4",), fs=fs)  

        # --- Mapping wizard -> labels to commands ---
        mapping_path = _configure_mapping(getattr(eng, "profile_dir", profile_dir), eng.meta["classes"])

        # --- Router + controller (single entrypoint) ---
        router = _build_router()
        th = threading.Thread(
            target=ctrl.run_controller,
            args=(event_bus.label_bus, eng.meta["classes"], router),
            kwargs={
                "default_robot": DEFAULT_ROBOT,
                "mapping_path": mapping_path,
                "profile_dir": getattr(eng, "profile_dir", None),
                "win": 3, "thr": 0.6, "refractory_ms": 200
            },
            daemon=True
        )
        th.start()
        print("[CTRL] - - -  CONTROLLER STARTED! - - -")

        # --- Inference callback ---
        def on_probs(p, t_ms):
            if DEBUG:
                print(f"[raw] {t_ms:8.1f} ms | probs={np.round(p,3)}")
            try:
                event_bus.label_bus.put_nowait(event_bus.ProbsEvent(probs=p, t_ms=t_ms))  # :contentReference[oaicite:11]{index=11}
            except:
                if DEBUG: print("[WARNING] !!! Queue is full, events are being dropped !!!")

        # --- Start streaming inference ---
        inference.run_inference_stream(
            source, eng,
            fs=fs, win_sec=win_sec, hop_sec=hop_sec,
            on_probs=on_probs,
            profile_dir=getattr(eng, "profile_dir", None),
            DEBUG=DEBUG
        )
