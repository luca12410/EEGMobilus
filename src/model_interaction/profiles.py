import os, json, pickle, shutil
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, Union

INDEX_FILE = "profiles_index.json"   
DEFAULT_ROOT = "profile_store"          

# ---------- Meta ----------
@dataclass
class ProfileMeta:
    fs: int
    chans: int
    samples: int
    classes: list          
    band: Tuple[int,int]   
    notch: float          
    notes: str = ""      

# ---------- Index helpers ----------
def _load_index() -> Dict[str,str]:
    if not os.path.exists(INDEX_FILE):
        return {}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_index(idx: Dict[str,str]) -> None:
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)

def register_profile(name: str, dir_path: str) -> None:
    idx = _load_index()
    idx[name] = dir_path
    _save_index(idx)

def get_profile_path(name: str) -> Optional[str]:
    return _load_index().get(name)

def list_profiles() -> Dict[str,str]:
    return _load_index()

def remove_profile(name: str, delete_files: bool = False) -> None:
    idx = _load_index()
    path = idx.pop(name, None)
    _save_index(idx)
    if delete_files and path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

# ---------- Save / Load ----------
def save_profile(
    name: str,
    model,                          
    meta: ProfileMeta,
    scaler: Optional[object] = None,  
    root: str = DEFAULT_ROOT,
    overwrite: bool = True,
) -> str:
    """
    Save: model (Keras), meta (ProfileMeta), optional scaler (e.g. StandardScaler)
    """
    dir_path = os.path.join(root, name)
    os.makedirs(dir_path, exist_ok=True)

    model_path = os.path.join(dir_path, "model.keras")
    if os.path.exists(model_path) and not overwrite:
        raise FileExistsError(f"{model_path} already exists. Use overwrite=True to replace.")
    model.save(model_path)

    with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    if scaler is not None:
        with open(os.path.join(dir_path, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    register_profile(name, dir_path)
    return dir_path

def load_profile(profile_dir: str):
    model_path  = os.path.join(profile_dir, "model.h5")
    meta_path   = os.path.join(profile_dir, "meta.json")
    meta_class  = os.path.join(profile_dir, "meta_classic.json")
    scaler_path = os.path.join(profile_dir, "scaler.pkl")

    # SVM + RF
    if os.path.exists(meta_class):
        with open(meta_class, "r", encoding="utf-8") as f:
            meta = json.load(f)
        scaler = None
        if os.path.exists(scaler_path):
            import pickle
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        model = None  
        return model, scaler, meta

    # --- DL PROFILE ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Manca {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Manca {meta_path}")

    model = _load_tf_model(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scaler = None
    if os.path.exists(scaler_path):
        import pickle
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return model, scaler, meta

def _load_tf_model(path: str):
    try:
        from tensorflow.keras.models import load_model as tf_load_model
    except ModuleNotFoundError:
        raise RuntimeError(
            "This profile requires TensorFlow but it is not installed. "
            "Use a classic profile or install TensorFlow."
        )
    return tf_load_model(path, compile=False)