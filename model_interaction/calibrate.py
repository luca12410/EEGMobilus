# train/calibrate.py
import os, json
import numpy as np
from typing import List, Tuple
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from base_model.EEGModels import EEGNet
from preprocessing.preprocess import preprocess_pipeline  # usa i tuoi filtri
from profiles import save_profile, ProfileMeta
from markers import MarkerStream, read_opensignals_digital
from open_signals_txt_eeg import OpenSignalsTxtEEG

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
    epochs:int=5, batch_size:int=64
) -> str:
    """
    Crea marker dai digitali, costruisce dataset, allena EEGNet, salva profilo.
    Ritorna il path del profilo salvato.
    """
    # 1) digital → markers
    digital, col_map, fs = read_opensignals_digital(txt_path, wanted=digital_cols)
    fs = fs or fs_fallback

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
