from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
LABELED_DATASET_DIR = DATASET_DIR / "final_labeled"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

RUNTIME_DIR = PROJECT_ROOT / "runtime"
TEXTURE_OUTPUT_DIR = RUNTIME_DIR / "texture"
ORIENTATION_OUTPUT_DIR = RUNTIME_DIR / "orientation"
WEB_INPUT_DIR = RUNTIME_DIR / "web_inputs"
WEB_OUTPUT_DIR = RUNTIME_DIR / "web_outputs"
REPORT_OUTPUT_DIR = RUNTIME_DIR / "reports"
HEATMAP_OUTPUT_DIR = RUNTIME_DIR / "heatmaps"
RESULTS_DIR = RUNTIME_DIR / "results"
FINAL_RESULTS_DIR = RUNTIME_DIR / "final_results"
FINAL_OVERLAY_DIR = RUNTIME_DIR / "final_overlays"
TRAINING_OUTPUT_DIR = RUNTIME_DIR / "training"
CONFIG_DIR = RUNTIME_DIR / "config"
DISPLAY_SETTINGS_PATH = CONFIG_DIR / "web_demo_display_settings.json"

DEFAULT_MODEL_NAME = "best_trans_unet_model_20250614_122913.pth"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME


def ensure_runtime_dirs() -> None:
    for path in (
        TEXTURE_OUTPUT_DIR,
        ORIENTATION_OUTPUT_DIR,
        WEB_INPUT_DIR,
        WEB_OUTPUT_DIR,
        REPORT_OUTPUT_DIR,
        HEATMAP_OUTPUT_DIR,
        RESULTS_DIR,
        FINAL_RESULTS_DIR,
        FINAL_OVERLAY_DIR,
        TRAINING_OUTPUT_DIR,
        CONFIG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_project_path(value: str | Path) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_model_path(value: str | Path = DEFAULT_MODEL_NAME) -> Path:
    """Resolve new models/ paths while accepting historical bare filenames."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate

    model_candidate = (MODELS_DIR / path.name).resolve()
    if model_candidate.exists() or path.parent == Path("."):
        return model_candidate

    return project_candidate
