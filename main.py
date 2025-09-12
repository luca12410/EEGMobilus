# main.py
from model_interaction.calibrate import run_calibration_interactive
from model_interaction.inference import run_inference_interactive

DEBUG = True

def _ask(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default

if __name__ == "__main__":
    action = _ask("Scegli azione (calibrazione/inferenza)", "calibrazione").lower()
    if action.startswith("c"):
        run_calibration_interactive(subject_name="test_subject", fs_default=100, win_sec_default=1.0)
    else:
        run_inference_interactive(
            default_profile_dir="profile_store/test_subject",
            default_mode="file",
            test_file="model_interaction/files/test_luca_eeg_blink_movement.txt",
            analog_channels=("A4",),
            win_sec=1.0,
            hop_sec=0.5
        )