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

def _windows_from_trials(X_raw: np.ndarray,
                         trials: List[Tuple[int,int,str]],
                         classes: List[str],
                         samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea dataset allineato alle finestre da training.
    Ritorna X [N,C,S] e y int [N].
    """
    C, T = X_raw.shape
    X_list, y_list = [], []
    for (t0, t1, lbl) in trials:
        # prendi la porzione centrale di lunghezza 'samples'
        if (t1 - t0) < samples:
            continue
        start = t0 + ( (t1 - t0) - samples ) // 2
        seg = X_raw[:, start:start+samples]   # [C,S]
        X_list.append(seg)
        y_list.append(classes.index(lbl))
    if not X_list:
        raise RuntimeError("Nessuna finestra valida creata dai trials.")
    X = np.stack(X_list, axis=0)   # [N,C,S]
    y = np.array(y_list, dtype=int)
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
    Crea marker dai digitali, costruisce dataset, allena EEGNet, salva profilo.
    Ritorna il path del profilo salvato.
    """
    # 1) digital → markers
    digital, col_map, fs = read_opensignals_digital(txt_path, wanted=digital_cols)
    fs = fs or fs_fallback
    if samples is None:
        samples = int(fs * win_sec)   # es. 1.0 s -> 100 campioni a fs=100
    print(f"[cal] fs={fs}, samples(win)={samples}")

    mk = MarkerStream.from_digital_columns(
        fs=fs, digital_matrix=digital, col_map=col_map, label_map=label_map,
        rising=True, min_separation_ms=300.0
    )
    trials = mk.trials(window_sec=(0.5, 3.5), labels_whitelist=list(classes))

    # 2) carica analogici (grezzi) coerenti con i canali richiesti
    src = OpenSignalsTxtEEG(txt_path, channels=analog_channels, fs=fs, volts_per_count=1.0)
    X_raw_full = src._X  # [C,T] già estratti

    # 3) preprocess (filtri + z-score)
    X_filt = preprocess_pipeline(X_raw_full, fs)

    # 4) dataset da trials → finestre [N,C,S] + y
    X, y = _windows_from_trials(X_filt, trials, list(classes), samples)
    # reshape per Keras: [N,C,S,1]
    X_keras = X[..., None]
    y_cat = to_categorical(y, num_classes=len(classes))

    # 5) modello (EEGNet) scalato per fs/samples/chans
    chans = X.shape[1]
    model = EEGNet(nb_classes=len(classes), Chans=chans, Samples=samples, kernLength=max(8, samples//2))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    # 6) training rapido
    callbacks = [
        EarlyStopping(monitor="val_acc", mode="max", patience=2, restore_best_weights=True),
    ]
    # split semplice 80/20
    n = X_keras.shape[0]
    idx = np.arange(n); np.random.shuffle(idx)
    cut = int(0.8*n)
    tr, va = idx[:cut], idx[cut:]
    model.fit(X_keras[tr], y_cat[tr],
              validation_data=(X_keras[va], y_cat[va]),
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)

    # 7) meta + salvataggio profilo
    meta = ProfileMeta(
        fs=fs, chans=chans, samples=samples,
        classes=list(classes), band=band, notch=notch, notes=f"Calibrated from {os.path.basename(txt_path)}"
    )
    profile_path = save_profile(subject_name, model, meta, scaler=None)
    # salva anche i marker per tracciabilità (facoltativo)
    with open(os.path.join(profile_path, "markers.json"), "w", encoding="utf-8") as f:
        json.dump({"trials": trials}, f, indent=2)

    return profile_path



def calibrate_guided_live(
    subject_name: str,
    acquire_chunk_fn,                 # callable: acquire_chunk_fn(seconds, fs) -> np.ndarray [C, T]
    labels=("mano","braccio"),        # etichette/cue da mostrare
    fs=100,
    chans=1,                          # numero di canali EEG che l'acquisitore fornisce
    trial_sec=3.0, rest_sec=2.0,      # durata cue attivo e riposo
    reps_per_label=8,
    band=(8,30), notch=50.0,
    win_sec=1.0,
    epochs:int=5, batch_size:int=64,
):
    """
    Mostra cue testuali, acquisisce in diretta senza marker, costruisce trials e allena EEGNet.
    Ritorna il path del profilo salvato.
    """
    import time, os, json
    import numpy as np
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    classes = list(labels)
    samples = int(fs * win_sec)
    print(f"[cal-live] fs={fs}, chans={chans}, win={samples} samp ({win_sec:.2f}s)")

    X_blocks = []
    trials = []   # lista di (t0, t1, label) in campioni sull’array concatenato
    t_cursor = 0

    def _cue(msg, sec):
        print(f"=== {msg.upper()} ({sec:.0f}s) ===")
        for s in range(int(sec), 0, -1):
            print(f"{s}…", end="\r", flush=True); time.sleep(1)
        print(" " * 10, end="\r")

    # baseline
    _cue("riposo", rest_sec)
    _ = acquire_chunk_fn(rest_sec, fs)   # scarto, solo stabilizzazione

    for lab in classes:
        for r in range(reps_per_label):
            _cue(f"preparati: {lab}", 2)
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
        raise RuntimeError("Nessun dato acquisito in calibrazione live.")
    X_raw_full = np.concatenate(X_blocks, axis=1)

    # Preprocess identico alla calibrazione da file
    X_filt = preprocess_pipeline(X_raw_full, fs)

    # Dataset da trials → finestre
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

    with open(os.path.join(profile_path, "markers.json"), "w", encoding="utf-8") as f:
        json.dump({"trials": trials, "mode": "guided_live"}, f, indent=2)

    print(f"[cal-live] Profilo salvato in: {profile_path}")
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
    Orchestrazione INTERATTIVA della calibrazione (cue a schermo, no marker).
    Tutta l'interazione resta qui, il main si limita a chiamare questa funzione.
    """
    from acquisition.EEG_live_acquisition import acquire_chunk_seconds

    print("[*] Calibrazione live guidata (no file / no marker)")
    labels_str = _ask("Classi (etichette) separate da virgola", "mano,braccio")
    labels = tuple([x.strip() for x in labels_str.split(",") if x.strip()]) or ("mano",)

    fs          = _ask_int("Frequenza di campionamento (Hz)", fs_default)
    trial_sec   = _ask_float("Durata azione per trial (s)", 3.0)
    rest_sec    = _ask_float("Durata riposo tra trial (s)", 2.0)
    reps        = _ask_int("Ripetizioni per classe (trial/etichetta)", 8)
    epochs      = _ask_int("Epoch di training", 5)
    batch_size  = _ask_int("Batch size", 64)
    win_sec     = _ask_float("Finestra feature (s)", win_sec_default)
    subject_name = _ask("Nome soggetto (usato per salvare il profilo)", subject_name)

    print("\nRiepilogo calibrazione:")
    print("  Classi:", labels)
    print(f"  fs={fs} Hz | trial={trial_sec}s | rest={rest_sec}s | reps/cls={reps}")
    print(f"  win={win_sec}s | epochs={epochs} | batch={batch_size}")
    _ = _ask("Invio per iniziare (Ctrl+C per annullare)", "")

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
    print(f"[✓] Profilo creato: {profile_dir}")
    return profile_dir

