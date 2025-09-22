# classic.py
# Ensemble SVM + RF with standard ML features (log-bandpower + RMS)
# Using scikit-learn

import os, json
import numpy as np
from dataclasses import dataclass
from joblib import dump, load
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def _butter_band(x, fs, lo, hi, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band', analog=False)
    return filtfilt(b, a, x, axis=-1)

def _log_bandpower(x, fs, bands=((0.5,4),(4,8),(8,12))):
    """
    x: [N,C,S] o [C,S]
    Returns: [N,F] or [F], where F = len(bands)
    """
    x = np.asarray(x)
    squeeze = False
    if x.ndim == 2:  # [C,S] -> [1,C,S]
        x = x[None, ...]
        squeeze = True
    assert x.shape[1] == 1, "Assuming one channel (C=1)."

    feats = []
    for (lo, hi) in bands:
        xb = _butter_band(x[:,0,:], fs, lo, hi)
        v = np.mean(xb*xb, axis=-1) + 1e-8
        feats.append(np.log(v))
    F = np.stack(feats, axis=-1)  # [N,F]
    return F[0] if squeeze else F

def _rms(x):
    # x: [N,C,S] o [C,S] -> [N,] o []
    x = np.asarray(x)
    squeeze = False
    if x.ndim == 2:
        x = x[None, ...]
        squeeze = True
    r = np.sqrt(np.mean(x[:,0,:]**2, axis=-1) + 1e-12)
    return r[0] if squeeze else r

def extract_features(x, fs, add_rms=True):
    """
    x: [N,1,S] o [1,S]
    Returns: [N, F] o [F], where F = len(bands) + (1 if add_rms)
    """
    bp = _log_bandpower(x, fs)             # [N,Fbp] o [Fbp]
    if add_rms:
        rms = _rms(x)                      # [N] o []
        if bp.ndim == 1:
            return np.concatenate([bp, [rms]], axis=-1)
        else:
            return np.concatenate([bp, rms[:,None]], axis=-1)
    return bp

# ---------------------------------------------------------------------
# Model + profile
# ---------------------------------------------------------------------

@dataclass
class ClassicMeta:
    fs: int
    samples: int
    classes: list
    features: str = "log-bandpower[1-4,4-8,8-12,12-30]+RMS"
    model_file: str = "classic_model.joblib"
    meta_file: str = "meta_classic.json"
    chans: int = 1

class ClassicModel:
    """
    Pipeline: StandardScaler -> Voting(SVM_RBF, RandomForest), soft vote.
    API:
        fit(X, y, fs, classes)
        save(profile_dir)
        load(profile_dir)
        predict_window(X_win)  # X_win: [1,S] o [C=1,S]
    """
    def __init__(self):
        svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
        rf  = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)
        self.ensemble = VotingClassifier(
            estimators=[("svm", svm), ("rf", rf)],
            voting="soft"
        )
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", self.ensemble)
        ])
        self.meta = None

    # -------- training --------
    def fit(self, X_win, y, fs, classes):
        """
        X_win: [N, C=1, S] 
        y:     [N] in {0..K-1}
        """
        X_win = np.asarray(X_win)
        assert X_win.ndim == 3 and X_win.shape[1] == 1, "Expecting X=[N,1,S]"
        S = X_win.shape[-1]
        Xf = extract_features(X_win, fs, add_rms=True)  # [N,F]
        self.pipeline.fit(Xf, y)
        self.meta = ClassicMeta(fs=int(fs), samples=int(S), classes=list(classes))
        return self

    # -------- inference on a single window. --------
    def predict_window(self, X_win):
        """
        X_win: [C=1, S] or [1, S]
        Returns: probs [K]
        """
        assert self.meta is not None, "Model is not loaded/trained."
        X_win = np.asarray(X_win)
        if X_win.ndim == 1:
            X_win = X_win[None, ...]   
        if X_win.shape[0] != 1:
            X_win = X_win[:1, :]
        assert X_win.shape[-1] == self.meta.samples, f"Window size {X_win.shape[-1]} != {self.meta.samples}"
        Xf = extract_features(X_win, self.meta.fs, add_rms=True)[None, ...]  # [1,F]
        probs = self.pipeline.predict_proba(Xf)[0]
        return probs

    # -------- saving profile --------
    def save(self, profile_dir):
        os.makedirs(profile_dir, exist_ok=True)
        # model
        dump(self.pipeline, os.path.join(profile_dir, self.meta.model_file))
        # meta
        with open(os.path.join(profile_dir, self.meta.meta_file), "w", encoding="utf-8") as f:
            json.dump(self.meta.__dict__, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(profile_dir):
        with open(os.path.join(profile_dir, "meta_classic.json"), "r", encoding="utf-8") as f:
            d = json.load(f)
        meta = ClassicMeta(**d)

        pipeline = load(os.path.join(profile_dir, meta.model_file))
        m = ClassicModel()
        m.pipeline = pipeline
        m.meta = meta
        return m

# ---------------------------------------------------------------------
# Training wizard
# ---------------------------------------------------------------------

def train_and_save_classic_profile(profile_dir, X_win, y, fs, classes):
    """
    Trains and saves a classic model (SVM + RF) profile.
    X_win: [N, C=1, S]
    """
    model = ClassicModel().fit(X_win, y, fs, classes)
    model.save(profile_dir)
    return profile_dir

def classic_predict_window(profile_dir, X_win):
    import json, os, numpy as np
    from joblib import load
    from preprocessing.preprocess import preprocess_pipeline  

    # meta classico
    meta_c = json.load(open(os.path.join(profile_dir, "meta_classic.json"), "r"))
    fs = meta_c["fs"]; samples = meta_c["samples"]
    assert X_win.shape[-1] == samples

    # load model
    clf = load(os.path.join(profile_dir, meta_c["model_file"]))

    # --- PREPROCESS ---
    Xp = preprocess_pipeline(np.asarray(X_win), fs)   

    Xf = extract_features(Xp[None, :, :], fs, add_rms=True)  # [1,F]

    probs = clf.predict_proba(Xf)[0]
    
    z = (Xp[0]-Xp[0].mean())/(Xp[0].std()+1e-8)
    zpk = float(np.max(np.abs(z)))
    if "blink" in meta_c["classes"]:
        b = meta_c["classes"].index("blink")
        if zpk < 3.0:          
            probs[b] *= 0.1
            probs = probs / probs.sum()
    
    return probs, meta_c["classes"]
