# main.py
import os, json, threading, numpy as np

import model_interaction.inference as inference                       # InferenceEngine, run_inference_stream  
import model_interaction.profiles  as profiles                         # load_profile / indice profili           

import acquisition.EEG_live_acquisition as live_src   # acquisition/EEG_live_acquisition.py    
import acquisition.open_signals_txt_eeg as file_src   # acquisition/open_signals_txt_eeg.py    

import robot_control.controller as ctrl                 # run_controller (single entrypoint)     
import robot_control.bus as event_bus                   # ProbsEvent, label_bus                   :contentReference[oaicite:5]{index=5}
import robot_control.router as rtr                      # RobotRouter                             :contentReference[oaicite:6]{index=6}

DEBUG = True
DEFAULT_PROFILE = "profile_store/test_subject"
DEFAULT_TEST_FILE = "model_interaction/files/test_luca_eeg_blink_movement.txt"
DEFAULT_ROBOT = "tb3_1"
DEFAULT_TOPIC = "/cmd_vel"



# ---------------- utilities ----------------
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
    """Wizard: salva labels_to_commands.json per il profilo e ritorna il path.
       Per ogni label puoi associare 0..N comandi. Se N>1, verranno alternati."""
    path = default_path or os.path.join(profile_dir, "labels_to_commands.json")
    print(f"\n[*] Configurazione movimenti per il profilo: {profile_dir}")
    print("    Classi:", classes)

    if os.path.exists(path):
        use = _ask("Trovato mapping esistente. Usarlo? (Y/n)", "Y").lower()
        if use in ("", "y", "yes"):
            print(f"[cfg] Uso mapping esistente: {path}\n")
            return path

    def _ask_cmd_entry():
        t = _ask("  Tipo comando (STOP/MOVE/TURN/CUSTOM)", "STOP").strip().upper()
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
            raw = _ask("    params JSON (opzionale)", "{}")
            try: params = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                print("    [warn] JSON non valido, uso {}"); params = {}
        else:
            t, params = "STOP", {}

        tgt = _ask("    target robot-id (vuoto=default/broadcast)", "").strip()
        entry = {"name": t, "params": params}
        if tgt: entry["target"] = tgt
        return entry

    print("\nPer ogni classe indica QUANTI comandi associare:")
    print("  0 = nessuna azione (label ignorata)")
    print("  1 = singolo comando")
    print("  N>=2 = alterna i N comandi in round-robin\n")

    mapping: dict = {}
    for lab in classes:
        n = _ask_int(f"Quanti comandi associare a '{lab}'? (0..8)", 1)
        n = max(0, min(n, 8))
        if n == 0:
            # Non aggiungo la chiave -> label ignorata
            continue
        entries = []
        for i in range(1, n+1):
            print(f"  [{lab}] Configura cmd{i}:")
            entries.append(_ask_cmd_entry())
        mapping[lab] = entries[0] if len(entries) == 1 else entries

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"[cfg] Salvato mapping in: {path}\n")
    return path

def _build_router():
    router = rtr.RobotRouter()

    # 1) Se definito, usa backend UDP (Windows->WSL bridge)
    target = os.getenv("EEG_UDP_TARGET")  # es. "127.0.0.1:9999"
    if target:
        host, port = (target.split(":", 1) + ["9999"])[:2]
        from robot_control.backends.udp import UDPClient
        router.register(DEFAULT_ROBOT, UDPClient(host, int(port)))
        print(f"[router] UDP backend -> {host}:{port}")
        return router

    # 2) altrimenti tieni il tuo flusso (ROS2 o MOCK)
    try:
        import robot_control.backends.ros2 as ros2_backend
        client = ros2_backend.ROS2CmdVelClient(DEFAULT_ROBOT, topic=DEFAULT_TOPIC)
        router.register(DEFAULT_ROBOT, client)
        print(f"[router] ROS2 backend su {DEFAULT_TOPIC}")
    except Exception as e:
        class _PrintBackend:
            def send(self, cmd): print(f"[MOCK] {cmd.name} {cmd.params} -> {cmd.target or DEFAULT_ROBOT}")
        router.register(DEFAULT_ROBOT, _PrintBackend())
        print(f"[router] Backend MOCK attivo (ROS2 non disponibile: {e})")
    return router

# ---------------- main orchestration ----------------
if __name__ == "__main__":
    action = _ask("Scegli azione (calibrazione/inferenza)", "calibrazione").lower()

    if action.startswith("c"):
        import model_interaction.calibrate as calibrate
        calibrate.run_calibration_interactive()  # resta tutto nel modulo calibrate  
    else:
        # --- Profilo / Engine (solo moduli, nessun import di classi dal main) ---
        profile_dir = _ask("Path profilo (cartella o nome registrato)", DEFAULT_PROFILE)
        eng = inference.InferenceEngine(profile_dir)                       # usa la tua classe (via modulo)  
        fs = int(eng.meta["fs"])
        win_sec = eng.meta["samples"] / fs
        hop_sec = max(0.05, win_sec / 4)

        # --- Sorgente: file/live ---
        mode = _ask("Sorgente inferenza (file/live)", "file").lower()
        if mode.startswith("l"):
            source = live_src.LiveSource(fs)                               # via modulo, nessun import classe  
        else:
            test_file = _ask("Percorso file OpenSignals", DEFAULT_TEST_FILE)
            source = file_src.OpenSignalsTxtEEG(test_file, channels=("A4",), fs=fs)  

        # --- Wizard mapping per questo profilo ---
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
        print("[ctrl] controller thread started")

        # --- Producer: on_probs → coda (main non decide nulla) ---
        def on_probs(p, t_ms):
            if DEBUG:
                print(f"[raw] {t_ms:8.1f} ms | probs={np.round(p,3)}")
            try:
                event_bus.label_bus.put_nowait(event_bus.ProbsEvent(probs=p, t_ms=t_ms))  # :contentReference[oaicite:11]{index=11}
            except:
                if DEBUG: print("[warn] coda piena, evento droppato")

        # --- Avvia inferenza streaming (solo orchestrazione) ---
        inference.run_inference_stream(
            source, eng,
            fs=fs, win_sec=win_sec, hop_sec=hop_sec,
            on_probs=on_probs,
            profile_dir=getattr(eng, "profile_dir", None),
            DEBUG=DEBUG
        )
