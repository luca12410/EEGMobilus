# profiles.py  — Keras/TensorFlow
import os, json, pickle, shutil
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, Union

from tensorflow.keras.models import load_model as tf_load_model

INDEX_FILE = "profiles_index.json"   # file { "nome": "profiles/nome", ... }
DEFAULT_ROOT = "profile_store"            # cartella dove creare i profili

# ---------- Meta ----------
@dataclass
class ProfileMeta:
    fs: int
    chans: int
    samples: int
    classes: list            # es. ["left","right"] (ordine = output)
    band: Tuple[int,int]     # es. (8,30)
    notch: float             # es. 50.0
    notes: str = ""          # opzionale

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
    model,                            # keras.Model già compilato o no
    meta: ProfileMeta,
    scaler: Optional[object] = None,  # es. dict con mean/var per canale
    root: str = DEFAULT_ROOT,
    overwrite: bool = True,
) -> str:
    """
    Salva: model.keras, meta.json, scaler.pkl in profiles/<name>/  e aggiorna l'indice.
    Ritorna il percorso del profilo.
    """
    dir_path = os.path.join(root, name)
    os.makedirs(dir_path, exist_ok=True)

    # modello (architettura + pesi)
    model_path = os.path.join(dir_path, "model.keras")
    if os.path.exists(model_path) and not overwrite:
        raise FileExistsError(f"{model_path} esiste già. Imposta overwrite=True o cambia nome profilo.")
    model.save(model_path)

    # meta
    with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    # scaler opzionale
    if scaler is not None:
        with open(os.path.join(dir_path, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    # indice
    register_profile(name, dir_path)
    return dir_path

def load_profile(profile: Union[str, os.PathLike]):
    """
    Carica (model, scaler, meta) da:
      - nome profilo registrato nell'indice, oppure
      - percorso cartella profilo (contente model.keras/meta.json/scaler.pkl?).
    """
    # risolvi nome → cartella (se necessario)
    profile = str(profile)
    dir_path = profile if os.path.isdir(profile) else get_profile_path(profile)
    if not dir_path:
        raise FileNotFoundError(f"Profilo '{profile}' non trovato (né cartella, né nel {INDEX_FILE}).")

    model_path = os.path.join(dir_path, "model.keras")
    meta_path  = os.path.join(dir_path, "meta.json")
    scaler_path= os.path.join(dir_path, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Manca {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Manca {meta_path}")

    model  = tf_load_model(model_path, compile=False)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return model, scaler, meta
