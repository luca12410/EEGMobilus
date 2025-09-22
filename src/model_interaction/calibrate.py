# train/calibrate.py
import os, json
import numpy as np
from typing import List, Tuple
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model_interaction.base_model.EEGModels import EEGNet
from preprocessing.preprocess import preprocess_pipeline
from model_interaction.profiles import save_profile, ProfileMeta
from preprocessing.markers import MarkerStream, read_opensignals_digital
from acquisition.open_signals_txt_eeg import OpenSignalsTxtEEG
from model_interaction.base_model.classic import train_and_save_classic_profile


def _windows_from_trials(X_raw: np.ndarray,
                         trials: List[Tuple[int,int,str]],
                         classes: List[str],
                         samples: int,
                         hop: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapped windows for each triel.
    """
    C, T = X_raw.shape
    hop = hop or max(1, samples // 4)
    X_list, y_list = [], []
    for (t0, t1, lbl) in trials:
        L = t1 - t0
        if L < samples:
            continue
        start = t0
        while start + samples <= t1:
            seg = X_raw[:, start:start+samples]  # [C,S]
            X_list.append(seg)
            y_list.append(classes.index(lbl))
            start += hop
    if not X_list:
        raise RuntimeError("No valid window was created from the trials.")
    X = np.stack(X_list, axis=0)   # [N,C,S]
    y = np.asarray(y_list, dtype=int)
    return X, y

def calibrate_from_txt(
    subject_name: str,
    txt_path: str,
    analog_channels: List[str],          # es. ["A4","A5","A6"]
    digital_cols: List[str],             # es. ["O1","O2"]
    label_map: dict,                     # {"O1":"left","O2":"right"}
    fs_fallback: int = 500,
    band=(8,30), notch=50.0,
    samples: int = 500,                  # 1 s @ 500 Hz
    classes: List[str] = ("left","right"),
    epochs:int=5, batch_size:int=64,
    win_sec=1.0
) -> str:
    """
    Create markers from digital columns, preprocess analog signals, extract windows, train EEGNet and save profile.
    Returns the path of the saved profile.
    """
    # 1) digital → markers
    digital, col_map, fs = read_opensignals_digital(txt_path, wanted=digital_cols)
    fs = fs or fs_fallback
    if samples is None:
        samples = int(fs * win_sec)  
    print(f"[CALIBRATION] fs={fs}, samples(win)={samples}")

    mk = MarkerStream.from_digital_columns(
        fs=fs, digital_matrix=digital, col_map=col_map, label_map=label_map,
        rising=True, min_separation_ms=300.0
    )
    trials = mk.trials(window_sec=(0.5, 3.5), labels_whitelist=list(classes))

    # 2) load analog signals (raw)
    src = OpenSignalsTxtEEG(txt_path, channels=analog_channels, fs=fs, volts_per_count=1.0)
    X_raw_full = src._X  # [C,T] già estratti

    # 3) - - - PREPROCESS - - -
    X_filt = preprocess_pipeline(X_raw_full, fs)

    # 4) dataset da trials → finestre [N,C,S] + y
    X, y = _windows_from_trials(X_filt, trials, list(classes), samples)
    
    # --- Specialized calibration for BLINK ---
    if set(map(str.lower, classes)) == {"blink", "stare"}:
        blink_idx = classes.index("blink")
        stare_idx = classes.index("stare")

        z = (X[:,0,:] - X[:,0,:].mean(axis=1, keepdims=True)) / (X[:,0,:].std(axis=1, keepdims=True) + 1e-8)
        zpk = np.max(np.abs(z), axis=1)

        z_thr = 3.0 
        pos_mask = (y == blink_idx) & (zpk >= z_thr)
        neg_mask = (y == stare_idx) | ((y == blink_idx) & (zpk < z_thr)) 

        X_pos = X[pos_mask];  y_pos = np.full(X_pos.shape[0], blink_idx, dtype=int)
        X_neg = X[neg_mask];  y_neg = np.full(X_neg.shape[0], stare_idx, dtype=int)

        n = min(len(X_pos), len(X_neg))
        if n < 20:
            raise RuntimeError("No significant blinks were read.")

        idxp = np.random.choice(len(X_pos), n, replace=False)
        idxn = np.random.choice(len(X_neg), n, replace=False)
        X = np.concatenate([X_pos[idxp], X_neg[idxn]], axis=0)
        y = np.concatenate([y_pos[idxp], y_neg[idxn]], axis=0)
    

    X_keras = X[..., None]
    y_cat = to_categorical(y, num_classes=len(classes))

    # 5) Load model EEGNet
    chans = X.shape[1]
    model = EEGNet(nb_classes=len(classes), Chans=chans, Samples=samples, kernLength=max(8, samples//2))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

     # 6) Train EEGNet
    callbacks = [EarlyStopping(monitor="val_acc", mode="max", patience=2, restore_best_weights=True)]
    n = X_keras.shape[0]
    idx = np.arange(n); np.random.shuffle(idx)
    cut = int(0.8*n)
    tr, va = idx[:cut], idx[cut:]
    model.fit(X_keras[tr], y_cat[tr],
              validation_data=(X_keras[va], y_cat[va]),
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)

    # 7) Meta + save profile for EEGNet
    chans = X.shape[1]
    meta = ProfileMeta(
        fs=fs, chans=chans, samples=samples,
        classes=list(classes), band=band, notch=notch,
        notes=f"Calibrated from {os.path.basename(txt_path)}"
    )
    profile_path = save_profile(subject_name, model, meta, scaler=None)

    # 8) TRAIN AND SAVE CLASSIC MODEL
    train_and_save_classic_profile(profile_path, X, y, fs, classes)
    print("[CALIBRATION] SVM + RF Baseline was correctly saved.")

    # 9) Save markers
    with open(os.path.join(profile_path, "markers.json"), "w", encoding="utf-8") as f:
        json.dump({"trials": trials}, f, indent=2)

    return profile_path



def calibrate_guided_live(
    subject_name: str,
    acquire_chunk_fn,               
    labels=("blink","stare"),       
    fs=100,
    chans=1,                        
    trial_sec=3.0, rest_sec=2.0,    
    reps_per_label=8,
    band=(0.5,12), notch=50.0,
    win_sec=1.0,
    epochs:int=5, batch_size:int=64,
):
    """
    Guided live calibration (no file, no marker).
    acquire_chunk_fn(sec, fs) -> np.ndarray [C,T]
    Returns the path of the saved profile.
    """
    import time, os, json
    import numpy as np
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    classes = list(labels)
    samples = int(fs * win_sec)
    print(f"[CALIBRATION] fs={fs}, chans={chans}, win={samples} samp ({win_sec:.2f}s)")

    X_blocks = []
    trials = []   # lista di (t0, t1, label) in campioni sull’array concatenato
    t_cursor = 0

    def _cue(msg, sec):
        print(f"=== {msg.upper()} ({sec:.0f}s) ===")
        for s in range(int(sec), 0, -1):
            print(f"{s}…", end="\r", flush=True); time.sleep(1)
        print(" " * 10, end="\r")

    # baseline
    _cue("rest", rest_sec)
    _ = acquire_chunk_fn(rest_sec, fs)   # scarto, solo stabilizzazione

    for lab in classes:
        for r in range(reps_per_label):
            _cue(f"prepare: {lab}", 2)
            _cue(lab, trial_sec)
            xi = acquire_chunk_fn(trial_sec, fs)      # atteso [C, T]
            assert xi.ndim == 2 and xi.shape[0] == chans, f"acquire_chunk_fn deve restituire [{chans}, T], got {xi.shape}"
            X_blocks.append(xi)
            n = xi.shape[1]
            trials.append((t_cursor, t_cursor + n, lab))
            t_cursor += n
            _cue("riposo", rest_sec)
            _ = acquire_chunk_fn(rest_sec, fs)

    if not X_blocks:
        raise RuntimeError("No data was read during the acquisition.")
    X_raw_full = np.concatenate(X_blocks, axis=1)

    # - - - PREPROCESS - - -
    X_filt = preprocess_pipeline(X_raw_full, fs)

    # Windows
    X, y = _windows_from_trials(X_filt, trials, classes, samples)
    X_keras = X[..., None]
    y_cat = to_categorical(y, num_classes=len(classes))

    chans_detected = X.shape[1]
    model = EEGNet(nb_classes=len(classes), Chans=chans_detected, Samples=samples, kernLength=max(8, samples//2))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    n = X_keras.shape[0]
    idx = np.arange(n); np.random.shuffle(idx)
    cut = max(1, int(0.8*n))
    if n < 2:
        tr = va = idx[:1]
    else:
        tr = idx[:cut]
        va = idx[cut:] if cut < n else idx[:1]
    model.fit(X_keras[tr], y_cat[tr],
              validation_data=(X_keras[va], y_cat[va]),
              epochs=epochs, batch_size=batch_size, verbose=1,
              callbacks=[EarlyStopping(monitor="val_acc", mode="max", patience=2, restore_best_weights=True)])

    meta = ProfileMeta(
        fs=fs, chans=chans_detected, samples=samples,
        classes=classes, band=band, notch=notch, notes="Calibrated LIVE (no markers)"
    )
    profile_path = save_profile(subject_name, model, meta, scaler=None)

    # Aggiungi SVM+RF nella stessa cartella profilo
    train_and_save_classic_profile(profile_path, X, y, fs, classes)
    print("[CALIBRATION] Classic model baseline was correctly saved.")

    with open(os.path.join(profile_path, "markers.json"), "w", encoding="utf-8") as f:
        json.dump({"trials": trials, "mode": "guided_live"}, f, indent=2)

    print(f"[CALIBRATION] Profile correctly saved in path: {profile_path}")
    return profile_path


def _ask(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default

def _ask_int(prompt: str, default: int) -> int:
    s = _ask(prompt, str(default))
    try: return int(s)
    except ValueError: return default

def _ask_float(prompt: str, default: float) -> float:
    s = _ask(prompt, str(default))
    try: return float(s.replace(",", "."))
    except ValueError: return default

def run_calibration_interactive(subject_name: str = "test_subject", fs_default: int = 100, win_sec_default: float = 1.0):
    """
    Interactive wizard for guided live calibration (no file, no marker).
    Returns the path of the saved profile.
    """
    from acquisition.EEG_live_acquisition import acquire_chunk_seconds

    print("[*] LIVE CALIBRATION INITIATED")
    labels_str = _ask("Please specify the classes (labels) to be registered", "blink,stare")
    labels = tuple([x.strip() for x in labels_str.split(",") if x.strip()]) or ("stare",)

    fs          = _ask_int("Sampling frequency (Hz)", fs_default)
    trial_sec   = _ask_float("Trial time per action (s)", 3.0)
    rest_sec    = _ask_float("Rest time per action (s)", 2.0)
    reps        = _ask_int("Number of trials per class", 8)
    epochs      = _ask_int("Number of training epoch", 5)
    batch_size  = _ask_int("Batch size", 64)
    win_sec     = _ask_float("Feature window (s)", win_sec_default)
    subject_name = _ask("Profile name", subject_name)

    print("\nCalibration summary:")
    print("  Classes:", labels)
    print(f"  fs={fs} Hz | trial={trial_sec}s | rest={rest_sec}s | reps/cls={reps}")
    print(f"  win={win_sec}s | epochs={epochs} | batch={batch_size}")
    _ = _ask("Press ENTER to continue (Ctrl+C to discard)", "")

    profile_dir = calibrate_guided_live(
        subject_name=subject_name,
        acquire_chunk_fn=acquire_chunk_seconds,
        labels=labels,
        fs=fs,
        chans=1,
        trial_sec=trial_sec,
        rest_sec=rest_sec,
        reps_per_label=reps,
        win_sec=win_sec,
        epochs=epochs,
        batch_size=batch_size
    )
    print(f"[✓] Profile was correctly created and saved in: {profile_dir}")
    return profile_dir

