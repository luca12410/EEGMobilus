# main.py
from model_interaction.calibrate import run_calibration_interactive
from model_interaction.inference import run_inference_interactive

def _ask(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default

if __name__ == "__main__":
    action = _ask("Scegli azione (calibrazione/inferenza)", "calibrazione").lower()
    if action.startswith("c"):
        run_calibration_interactive(subject_name="test_subject", fs_default=100, win_sec_default=1.0)
    else:
        run_inference_interactive(
            default_profile_dir="profiles/latest",
            default_mode="file",
            test_file="model_interaction/files/campione090824_test.txt",
            analog_channels=("A4",),
            win_sec=1.0,
            hop_sec=0.5
        )