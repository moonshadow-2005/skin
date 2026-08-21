from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

try:
    import cv2
    _CV2_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV2_IMPORT_ERROR = exc
import numpy as np
import streamlit as st
import torch
import matplotlib.cm as cm

from src.texture_extraction import analyze_skin_texture
from src.orientation_analysis import analyze_texture_orientation
from src.local_score_heatmap import (
    build_disk_kernel,
    build_effective_texture_region_mask,
    compute_orientations,
    dynamic_radius_from_size,
    make_heatmap_image,
    normalize_on_mask,
    overlay_heatmap,
    predict_mask,
)
from src.run_one_full_pipeline import overlay_images_unicode
from src.worst_box_direction import (
    extract_connected_high_area,
    find_worst_box,
    scaled_box_size_from_shape,
)
import src.worst_box_direction as worst_box_direction_module
from src.project_paths import (
    DEFAULT_MODEL_NAME,
    DISPLAY_SETTINGS_PATH,
    ORIENTATION_OUTPUT_DIR,
    PROJECT_ROOT,
    TEXTURE_OUTPUT_DIR,
    WEB_INPUT_DIR,
    WEB_OUTPUT_DIR,
    resolve_model_path,
)


INPUT_DIR = WEB_INPUT_DIR
OUTPUT_DIR = WEB_OUTPUT_DIR

PARAM_DEFAULTS = {
    "model_rel": f"models/{DEFAULT_MODEL_NAME}",
    "target_class": 1,
    "radius_mode": "Dynamic",
    "fixed_radius": 40,
    "density_weight": 0.70,
    "consistency_weight": 0.30,
    "heat_alpha": 0.55,
    "n_bins": 2,
    "presence_mode": "quantile",
    "presence_cuts": [80.0],
    "box_mode": "Fixed",
    "box_size": 80,
    "num_boxes": 2,
    "min_overlap": 0.30,
    "area_percentile": 80.0,
    "local_direction_radius": 40,
}

DISPLAY_SETTINGS_DEFAULTS = {
    "parameters": {
        "model_file": True,
        "target_class": True,
        "radius_mode": True,
        "heatmap_radius": True,
        "density_weight": True,
        "consistency_weight": True,
        "severity_overlay_opacity": True,
        "presence_levels": True,
        "presence_split_mode": True,
        "presence_cutoffs": True,
        "box_size_mode": True,
        "box_size": True,
        "number_of_boxes": True,
        "minimum_mask_coverage": True,
        "worst_area_percentile": True,
        "local_direction_radius": True,
        "parameter_guide": True,
    },
    "images": {
        "original_image": True,
        "segmentation_overlay": True,
        "segmentation_filled": True,
        "texture_lines": True,
        "texture_comparison": True,
        "pixel_texture_axis_map": True,
        "texture_axis_overlay": True,
        "severity_map": True,
        "presence_map": True,
        "severity_map_overlay": True,
        "presence_level_overlay": True,
        "worst_area_search_boxes": True,
        "presence_map_overlay": True,
        "regional_mean_direction_map": True,
        "local_direction_map": True,
        "worst_area_parameters": True,
    },
}

PARAMETER_SETTING_LABELS = {
    "model_file": "Model File",
    "target_class": "Target Class",
    "radius_mode": "Radius Mode",
    "heatmap_radius": "Heatmap Radius",
    "density_weight": "Local Texture Density Weight",
    "consistency_weight": "Local Orientation Consistency Weight",
    "severity_overlay_opacity": "Severity Map Overlay Opacity",
    "presence_levels": "Number of Presence Levels",
    "presence_split_mode": "Presence Split Mode",
    "presence_cutoffs": "Presence Thresholds / Quantiles",
    "box_size_mode": "Box Size Mode",
    "box_size": "Box Size",
    "number_of_boxes": "Number of Boxes",
    "minimum_mask_coverage": "Minimum Mask Coverage",
    "worst_area_percentile": "Worst Area Severity Percentile (q)",
    "local_direction_radius": "Local Direction Radius",
    "parameter_guide": "Parameter Guide",
}

IMAGE_SETTING_LABELS = {
    "original_image": "Original Image (Output Copy)",
    "segmentation_overlay": "Three-Class Segmentation Overlay",
    "segmentation_filled": "Three-Class Segmentation",
    "texture_lines": "Texture Lines",
    "texture_comparison": "Texture Line Overlay",
    "pixel_texture_axis_map": "Pixel-Level Texture-Axis Map",
    "texture_axis_overlay": "Texture-Axis Overlay",
    "severity_map": "Severity Map",
    "presence_map": "Presence Level",
    "severity_map_overlay": "Severity Map Overlay",
    "presence_level_overlay": "Presence Level Overlay",
    "worst_area_search_boxes": "Worst Area Search Boxes",
    "presence_map_overlay": "Presence Map Overlay",
    "regional_mean_direction_map": "Regional Mean Direction Map",
    "local_direction_map": "Local Direction Map",
    "worst_area_parameters": "Worst Area Parameters",
}

PARAMETER_SETTING_GROUPS = (
    ("Input and Model", ("model_file", "target_class")),
    (
        "Severity Analysis",
        ("radius_mode", "heatmap_radius", "density_weight", "consistency_weight", "severity_overlay_opacity"),
    ),
    (
        "Presence Parameters",
        ("presence_levels", "presence_split_mode", "presence_cutoffs", "worst_area_percentile"),
    ),
    (
        "Worst Area Detection",
        ("box_size_mode", "box_size", "number_of_boxes", "minimum_mask_coverage"),
    ),
    ("Direction Analysis", ("local_direction_radius",)),
    ("Help", ("parameter_guide",)),
)

IMAGE_SETTING_GROUPS = (
    ("Input", ("original_image",)),
    ("Three-Class Segmentation", ("segmentation_overlay", "segmentation_filled")),
    ("Texture Extraction", ("texture_lines", "texture_comparison")),
    ("Texture-Axis Estimation", ("pixel_texture_axis_map", "texture_axis_overlay")),
    ("Severity Analysis", ("severity_map", "severity_map_overlay")),
    ("Presence Level", ("presence_map", "presence_level_overlay")),
    ("Worst Area Detection", ("worst_area_search_boxes", "presence_map_overlay", "worst_area_parameters")),
    ("Direction Analysis", ("regional_mean_direction_map", "local_direction_map")),
)


def normalize_display_settings(raw_data: dict | None) -> dict:
    normalized = {
        group: dict(defaults)
        for group, defaults in DISPLAY_SETTINGS_DEFAULTS.items()
    }
    if not isinstance(raw_data, dict):
        return normalized

    for group, defaults in DISPLAY_SETTINGS_DEFAULTS.items():
        raw_group = raw_data.get(group)
        if not isinstance(raw_group, dict):
            continue
        for key in defaults:
            value = raw_group.get(key)
            if isinstance(value, bool):
                normalized[group][key] = value
    return normalized


def load_display_settings() -> dict:
    if not DISPLAY_SETTINGS_PATH.exists():
        return normalize_display_settings(None)
    try:
        raw = json.loads(DISPLAY_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return normalize_display_settings(None)
    return normalize_display_settings(raw)


def save_display_settings(settings: dict) -> None:
    normalized = normalize_display_settings(settings)
    DISPLAY_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = DISPLAY_SETTINGS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(normalized, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(DISPLAY_SETTINGS_PATH)


def display_enabled(settings: dict, group: str, key: str) -> bool:
    return bool(settings.get(group, {}).get(key, DISPLAY_SETTINGS_DEFAULTS[group][key]))


def render_display_settings_page(settings: dict) -> None:
    st.title("Display Settings")
    st.caption(
        "Choose which parameter controls and result images are visible on the Analysis page. "
        "Saved settings persist across Demo restarts."
    )

    with st.form("display_settings_form"):
        parameter_tab, image_tab = st.tabs(["Parameter Controls", "Result Images"])
        selected = {"parameters": {}, "images": {}}

        with parameter_tab:
            st.markdown("Select the controls to show in the Analysis sidebar.")
            for group_title, keys in PARAMETER_SETTING_GROUPS:
                st.markdown(f"**{group_title}**")
                columns = st.columns(2)
                for index, key in enumerate(keys):
                    with columns[index % 2]:
                        selected["parameters"][key] = st.checkbox(
                            PARAMETER_SETTING_LABELS[key],
                            value=display_enabled(settings, "parameters", key),
                            key=f"setting_parameter_{key}",
                        )

        with image_tab:
            st.markdown("Select the intermediate results and output images to show.")
            for group_title, keys in IMAGE_SETTING_GROUPS:
                st.markdown(f"**{group_title}**")
                columns = st.columns(2)
                for index, key in enumerate(keys):
                    with columns[index % 2]:
                        selected["images"][key] = st.checkbox(
                            IMAGE_SETTING_LABELS[key],
                            value=display_enabled(settings, "images", key),
                            key=f"setting_image_{key}",
                        )

        save_clicked = st.form_submit_button("Save Display Settings", type="primary")

    reset_clicked = st.button("Reset All to Visible")
    if save_clicked:
        try:
            save_display_settings(selected)
        except OSError as exc:
            st.error(f"Could not save display settings: {exc}")
        else:
            st.success(f"Display settings saved to {DISPLAY_SETTINGS_PATH.name}.")
            st.rerun()
    if reset_clicked:
        try:
            save_display_settings(DISPLAY_SETTINGS_DEFAULTS)
        except OSError as exc:
            st.error(f"Could not reset display settings: {exc}")
        else:
            for key in list(st.session_state):
                if key.startswith("setting_parameter_") or key.startswith("setting_image_"):
                    del st.session_state[key]
            st.success("All parameter controls and result images are visible again.")
            st.rerun()


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _clamp_float(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(v))))


def _normalize_presence_cuts(cuts: list[float], n_bins: int, mode: str) -> list[float]:
    needed = max(0, int(n_bins) - 1)
    cleaned = []
    for c in cuts:
        if mode == "threshold":
            cleaned.append(_clamp_float(c, 0.0, 1.0))
        else:
            cleaned.append(_clamp_float(c, 0.0, 100.0))
    cleaned = sorted(cleaned)
    if len(cleaned) >= needed:
        return cleaned[:needed]

    for i in range(len(cleaned), needed):
        if mode == "threshold":
            cleaned.append(float((i + 1) / max(1, n_bins)))
        else:
            cleaned.append(float((i + 1) * 100.0 / max(1, n_bins)))
    return sorted(cleaned)


def normalize_params_dict(raw_data: dict) -> dict:
    data = dict(PARAM_DEFAULTS)
    if isinstance(raw_data, dict):
        data.update(raw_data)

    data["target_class"] = 1 if int(data.get("target_class", 1)) != 2 else 2
    legacy_fixed = "\u56fa\u5b9a"
    data["radius_mode"] = "Fixed" if data.get("radius_mode") in ("Fixed", legacy_fixed) else "Dynamic"
    data["box_mode"] = "Fixed" if data.get("box_mode") in ("Fixed", legacy_fixed) else "Dynamic"
    data["presence_mode"] = "quantile" if data.get("presence_mode") == "quantile" else "threshold"
    data["fixed_radius"] = _clamp_int(int(data.get("fixed_radius", 40)), 8, 120)
    data["density_weight"] = _clamp_float(float(data.get("density_weight", 0.70)), 0.0, 1.0)
    data["consistency_weight"] = _clamp_float(float(data.get("consistency_weight", 0.30)), 0.0, 1.0)
    data["heat_alpha"] = _clamp_float(float(data.get("heat_alpha", 0.55)), 0.1, 0.9)
    data["n_bins"] = _clamp_int(int(data.get("n_bins", 4)), 2, 6)
    data["box_size"] = _clamp_int(int(data.get("box_size", 80)), 24, 240)
    data["num_boxes"] = _clamp_int(int(data.get("num_boxes", 1)), 1, 5)
    data["min_overlap"] = _clamp_float(float(data.get("min_overlap", 0.30)), 0.0, 0.95)
    data["area_percentile"] = _clamp_float(float(data.get("area_percentile", 80.0)), 0.0, 100.0)
    data["local_direction_radius"] = _clamp_int(int(data.get("local_direction_radius", 40)), 1, 240)

    raw_cuts = data.get("presence_cuts", PARAM_DEFAULTS["presence_cuts"])
    if not isinstance(raw_cuts, list):
        raw_cuts = PARAM_DEFAULTS["presence_cuts"]
    data["presence_cuts"] = _normalize_presence_cuts(raw_cuts, data["n_bins"], data["presence_mode"])
    return data


def apply_params_to_session(params: dict) -> None:
    st.session_state["p_model_rel"] = str(params["model_rel"])
    st.session_state["p_target_class"] = int(params["target_class"])
    st.session_state["p_radius_mode"] = str(params["radius_mode"])
    st.session_state["p_fixed_radius"] = int(params["fixed_radius"])
    st.session_state["p_density_weight"] = float(params["density_weight"])
    st.session_state["p_consistency_weight"] = float(params["consistency_weight"])
    st.session_state["p_heat_alpha"] = float(params["heat_alpha"])
    st.session_state["p_n_bins"] = int(params["n_bins"])
    st.session_state["p_presence_mode"] = str(params["presence_mode"])
    st.session_state["p_box_mode"] = str(params["box_mode"])
    st.session_state["p_box_size"] = int(params["box_size"])
    st.session_state["p_num_boxes"] = int(params["num_boxes"])
    st.session_state["p_min_overlap"] = float(params["min_overlap"])
    st.session_state["p_area_percentile"] = float(params["area_percentile"])
    st.session_state["p_local_direction_radius"] = int(params["local_direction_radius"])

    cuts = params["presence_cuts"]
    for i in range(5):
        key = f"p_presence_cut_{i}"
        if i < len(cuts):
            st.session_state[key] = float(cuts[i])
        elif key not in st.session_state:
            st.session_state[key] = 0.0


def load_params_from_output_dir(out_dir: Path) -> dict | None:
    params_path = out_dir / "run_params.json"
    if not params_path.exists():
        return None
    try:
        raw = json.loads(params_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return normalize_params_dict(raw)
    except Exception:
        return None


def parameter_session_defaults() -> dict:
    params = normalize_params_dict(PARAM_DEFAULTS)
    defaults = {
        "p_model_rel": str(params["model_rel"]),
        "p_target_class": int(params["target_class"]),
        "p_radius_mode": str(params["radius_mode"]),
        "p_fixed_radius": int(params["fixed_radius"]),
        "p_density_weight": float(params["density_weight"]),
        "p_consistency_weight": float(params["consistency_weight"]),
        "p_heat_alpha": float(params["heat_alpha"]),
        "p_n_bins": int(params["n_bins"]),
        "p_presence_mode": str(params["presence_mode"]),
        "p_box_mode": str(params["box_mode"]),
        "p_box_size": int(params["box_size"]),
        "p_num_boxes": int(params["num_boxes"]),
        "p_min_overlap": float(params["min_overlap"]),
        "p_area_percentile": float(params["area_percentile"]),
        "p_local_direction_radius": int(params["local_direction_radius"]),
    }
    cuts = params["presence_cuts"]
    for i in range(5):
        defaults[f"p_presence_cut_{i}"] = float(cuts[i]) if i < len(cuts) else 0.0
    return defaults


def cache_parameter_session_state() -> None:
    cache = dict(st.session_state.get("_parameter_state_cache", {}))
    for key in parameter_session_defaults():
        if key in st.session_state:
            cache[key] = st.session_state[key]
    st.session_state["_parameter_state_cache"] = cache


def restore_cached_parameter_session_state() -> None:
    cached_values = st.session_state.get("_parameter_state_cache", {})
    if not isinstance(cached_values, dict):
        return
    for key in parameter_session_defaults():
        if key in cached_values:
            st.session_state[key] = cached_values[key]


def init_params_state_once() -> None:
    cached_values = st.session_state.get("_parameter_state_cache", {})
    if not isinstance(cached_values, dict):
        cached_values = {}
    for key, value in parameter_session_defaults().items():
        if key not in st.session_state:
            st.session_state[key] = cached_values.get(key, value)

    legacy_fixed = "\u56fa\u5b9a"
    st.session_state["p_radius_mode"] = (
        "Fixed" if st.session_state.get("p_radius_mode") in ("Fixed", legacy_fixed) else "Dynamic"
    )
    st.session_state["p_box_mode"] = (
        "Fixed" if st.session_state.get("p_box_mode") in ("Fixed", legacy_fixed) else "Dynamic"
    )
    cache_parameter_session_state()
    st.session_state["_params_initialized"] = True


def build_current_params_snapshot(
    model_rel: str,
    target_class: int,
    radius_mode: str,
    fixed_radius: int | None,
    density_weight: float,
    consistency_weight: float,
    heat_alpha: float,
    n_bins: int,
    presence_mode: str,
    presence_cuts: list[float],
    box_mode: str,
    box_size: int | None,
    num_boxes: int,
    min_overlap: float,
    area_percentile: float,
    local_direction_radius: int,
) -> dict:
    snapshot = {
        "model_rel": str(model_rel),
        "target_class": int(target_class),
        "radius_mode": str(radius_mode),
        "fixed_radius": int(fixed_radius) if fixed_radius is not None else int(st.session_state.get("p_fixed_radius", 40)),
        "density_weight": float(density_weight),
        "consistency_weight": float(consistency_weight),
        "heat_alpha": float(heat_alpha),
        "n_bins": int(n_bins),
        "presence_mode": str(presence_mode),
        "presence_cuts": [float(v) for v in presence_cuts],
        "box_mode": str(box_mode),
        "box_size": int(box_size) if box_size is not None else int(st.session_state.get("p_box_size", 80)),
        "num_boxes": int(num_boxes),
        "min_overlap": float(min_overlap),
        "area_percentile": float(area_percentile),
        "local_direction_radius": int(local_direction_radius),
    }
    snapshot["presence_cuts"] = _normalize_presence_cuts(snapshot["presence_cuts"], snapshot["n_bins"], snapshot["presence_mode"])
    return snapshot


def save_params_to_output_dir(
    out_dir: Path,
    input_image_path: Path,
    params_snapshot: dict,
    heat_info: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(params_snapshot)
    payload["input_image"] = str(input_image_path)
    payload["output_dir"] = str(out_dir)
    payload["radius_used"] = int(heat_info.get("radius", payload.get("fixed_radius", 0)))
    payload["box_size_used"] = int(heat_info.get("box_size", payload.get("box_size", 0)))
    payload["presence_mode_used"] = str(heat_info.get("presence_mode", payload.get("presence_mode", "threshold")))
    payload["presence_cuts_used"] = [float(v) for v in heat_info.get("presence_cuts", payload.get("presence_cuts", []))]
    payload["presence_thresholds_used"] = [float(v) for v in heat_info.get("presence_thresholds", [])]
    if "area_percentile" in heat_info:
        payload["area_percentile_used"] = float(heat_info["area_percentile"])
    if "area_threshold" in heat_info:
        payload["area_threshold_used"] = float(heat_info["area_threshold"])
    if "local_direction_radius" in heat_info:
        payload["local_direction_radius_used"] = int(heat_info["local_direction_radius"])

    out_path = out_dir / "run_params.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()
    return cleaned or "uploaded_image"


def imread_unicode(image_path: Path | None, flags=cv2.IMREAD_COLOR):
    if image_path is None:
        return None
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def save_uploaded_file(uploaded) -> tuple[Path, str, str, str]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    content = uploaded.getvalue()
    digest = hashlib.md5(content).hexdigest()[:10]
    original_name = Path(uploaded.name).name
    original_stem = sanitize_name(Path(original_name).stem)
    ext = Path(uploaded.name).suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        ext = ".jpg"
    case_id = f"web_{digest}"
    output_folder = f"{original_stem}__{case_id}"
    image_path = INPUT_DIR / f"{case_id}{ext}"
    image_path.write_bytes(content)
    return image_path, case_id, original_name, output_folder


def new_batch_id() -> str:
    return f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_batch_manifest(batch_dir: Path, items: list[dict]) -> Path:
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "batch_id": batch_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_items": len(items),
        "items": items,
    }
    out = batch_dir / "batch_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def copy_original_to_output(image_path: Path, out_dir: Path, original_name: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if original_name:
        safe_name = sanitize_name(Path(original_name).stem)
    else:
        safe_name = sanitize_name(image_path.stem)

    dst = out_dir / f"00_original__{safe_name}.png"
    img = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read the original image for PNG conversion: {image_path}")
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Cannot encode the original image as PNG: {image_path}")
    encoded.tofile(str(dst))
    return dst


def pick_existing_path(candidates: List[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_predict_output(case_id: str, prefix: str) -> Path:
    """Resolve predict output file with compatibility for IDs that contain underscores.

    predict.py currently derives output suffix from the last underscore-separated token,
    so uploaded IDs like web_abcd may produce files named with suffix abcd only.
    """
    predict_dir = ORIENTATION_OUTPUT_DIR
    suffix = case_id.split("_")[-1]
    exact = predict_dir / f"{prefix}{case_id}.png"
    fallback = predict_dir / f"{prefix}{suffix}.png"
    matched = pick_existing_path([exact, fallback])
    if matched is not None:
        return matched

    wildcard = sorted(predict_dir.glob(f"{prefix}*{suffix}.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if wildcard:
        return wildcard[0]

    raise FileNotFoundError(f"Prediction output not found: {prefix}{case_id}.png (or compatible suffix {suffix})")


def make_segmentation_visuals(
    original_bgr: np.ndarray,
    pred: np.ndarray,
    region_boundary_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = original_bgr.shape[:2]
    if pred.shape[:2] != (h, w):
        pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    filled = np.zeros_like(rgb, dtype=np.uint8)
    # class 0/1/2 colors in RGB
    bg_color = [70, 130, 255]
    outer_color = [80, 220, 120]
    inner_color = [255, 110, 90]
    filled[:, :] = bg_color

    boundary = np.zeros_like(rgb, dtype=np.uint8)
    mask_u8 = None
    hole_mask = np.zeros((h, w), dtype=np.uint8)
    if region_boundary_mask is not None:
        mask_u8 = (region_boundary_mask > 0).astype(np.uint8)
        if mask_u8.shape[:2] != (h, w):
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        # Use region-mask contour hierarchy so boundary rings match severity_map exactly:
        # outer boundary -> green, inner hole boundaries -> red.
        contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            hier = hierarchy[0]
            for i, cnt in enumerate(contours):
                parent = hier[i][3]
                color = (0, 255, 0) if parent == -1 else (255, 0, 0)
                cv2.drawContours(boundary, [cnt], -1, color, 2)
                if parent != -1:
                    cv2.drawContours(hole_mask, [cnt], -1, 1, thickness=-1)
        else:
            # Fallback: if no hierarchy is available, draw only outer boundary.
            contours_ext, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(boundary, contours_ext, -1, (0, 255, 0), 2)

        # Fill segmentation by region-mask geometry (not raw pred class1/class2).
        filled[mask_u8 > 0] = outer_color
        filled[hole_mask > 0] = inner_color
    else:
        filled[pred == 0] = bg_color
        filled[pred == 1] = outer_color
        filled[pred == 2] = inner_color
        for class_id, color in [(2, (255, 0, 0)), (1, (0, 255, 0))]:
            binary = (pred == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(boundary, contours, -1, color, 2)

    alpha_map = np.zeros((h, w), dtype=np.float32)
    if mask_u8 is not None:
        alpha_map[:, :] = 0.12
        alpha_map[mask_u8 > 0] = 0.28
        alpha_map[hole_mask > 0] = 0.40
    else:
        alpha_map[pred == 0] = 0.12
        alpha_map[pred == 1] = 0.28
        alpha_map[pred == 2] = 0.40
    overlay = rgb.astype(np.float32).copy()
    for c in range(3):
        overlay[:, :, c] = (1.0 - alpha_map) * overlay[:, :, c] + alpha_map * filled[:, :, c]

    # Apply the same boundary styling to the filled view for visual consistency.
    filled = cv2.addWeighted(filled.astype(np.uint8), 1.0, boundary, 0.85, 0)
    overlay = cv2.addWeighted(overlay.astype(np.uint8), 1.0, boundary, 0.75, 0)
    return filled, boundary, overlay


def compute_score_map_custom(
    image_path: Path,
    model_path: Path,
    texture_path: Path,
    target_class: int,
    fixed_radius: int | None,
    density_weight: float,
    consistency_weight: float,
    device: str,
):
    pred = predict_mask(str(image_path), str(model_path), device)
    raw_region_mask = (pred == target_class).astype(np.uint8)
    if target_class == 1:
        region_mask = build_effective_texture_region_mask(raw_region_mask)
    else:
        region_mask = raw_region_mask

    if np.sum(region_mask) == 0:
        raise RuntimeError("The effective intertidal zone is empty. Change Target Class or check the image quality.")

    tex = imread_unicode(texture_path, cv2.IMREAD_GRAYSCALE)
    if tex is None:
        raise FileNotFoundError(f"Texture image not found: {texture_path}")
    if tex.shape != pred.shape:
        tex = cv2.resize(tex, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    h, w = tex.shape
    radius = dynamic_radius_from_size(h, w) if fixed_radius is None else int(fixed_radius)
    kernel = build_disk_kernel(radius)

    texture_binary = ((tex > 0) & (region_mask > 0)).astype(np.float32)
    region_mask_f = region_mask.astype(np.float32)

    valid_count = cv2.filter2D(region_mask_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    texture_count = cv2.filter2D(texture_binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    density = np.zeros_like(texture_binary, dtype=np.float32)
    valid_local = valid_count > 1e-6
    density[valid_local] = texture_count[valid_local] / valid_count[valid_local]

    orientations = compute_orientations(tex, region_mask)
    orient_valid = (~np.isnan(orientations) & (region_mask > 0)).astype(np.float32)
    theta = np.nan_to_num(orientations, nan=0.0)

    cos2 = np.cos(2.0 * theta).astype(np.float32) * orient_valid
    sin2 = np.sin(2.0 * theta).astype(np.float32) * orient_valid

    cnt = cv2.filter2D(orient_valid, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    sum_cos = cv2.filter2D(cos2, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    sum_sin = cv2.filter2D(sin2, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    consistency = np.zeros_like(texture_binary, dtype=np.float32)
    ok = cnt > 1e-6
    mean_cos = np.zeros_like(texture_binary, dtype=np.float32)
    mean_sin = np.zeros_like(texture_binary, dtype=np.float32)
    mean_cos[ok] = sum_cos[ok] / cnt[ok]
    mean_sin[ok] = sum_sin[ok] / cnt[ok]
    consistency[ok] = np.sqrt(mean_cos[ok] ** 2 + mean_sin[ok] ** 2)

    total = max(1e-6, density_weight + consistency_weight)
    dw = density_weight / total
    cw = consistency_weight / total

    score = dw * density + cw * consistency
    score[region_mask == 0] = 0.0
    score_norm = normalize_on_mask(score, region_mask)
    return {
        "pred": pred,
        "region_mask": region_mask,
        "orientations": orientations,
        "density": density,
        "consistency": consistency,
        "score_norm": score_norm,
        "radius": radius,
    }


def add_vertical_colorbar_bgr(
    image_bgr: np.ndarray,
    bar_width: int = 28,
    pad: int = 10,
    margin: int = 12,
) -> np.ndarray:
    """Append a right-side vertical colorbar for normalized severity [0, 1]."""
    h, w = image_bgr.shape[:2]
    canvas_w = w + pad + bar_width + 56
    canvas = np.full((h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[:, :w] = image_bgr

    y = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    cmap_rgb = (cv2.applyColorMap((y * 255).astype(np.uint8), cv2.COLORMAP_JET))
    bar = np.repeat(cmap_rgb, bar_width, axis=1)

    x0 = w + pad
    x1 = x0 + bar_width
    canvas[:, x0:x1] = bar
    cv2.rectangle(canvas, (x0, 0), (x1 - 1, h - 1), (0, 0, 0), 1)

    ticks = [1.0, 0.75, 0.50, 0.25, 0.0]
    for t in ticks:
        yy = int(round((1.0 - t) * (h - 1)))
        cv2.line(canvas, (x1 + 2, yy), (x1 + 10, yy), (0, 0, 0), 1)
        label = f"{t:.2f}" if t not in (1.0, 0.0) else f"{t:.1f}"
        cv2.putText(
            canvas,
            label,
            (x1 + 14, min(max(yy + 4, margin), h - margin)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(canvas, "Severity", (x0 - 2, margin), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def build_presence_map_custom(
    score_norm: np.ndarray,
    region_mask: np.ndarray,
    thresholds: List[float],
    colors_bgr: List[Tuple[int, int, int]],
) -> np.ndarray:
    if len(colors_bgr) != len(thresholds) + 1:
        raise ValueError("The number of colors must equal the number of thresholds plus one.")

    out = np.zeros((score_norm.shape[0], score_norm.shape[1], 3), dtype=np.uint8)
    inside = region_mask > 0
    if not np.any(inside):
        return out

    bins = [-np.inf] + thresholds + [np.inf]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            m = inside & (score_norm <= hi)
        elif i == len(bins) - 2:
            m = inside & (score_norm > lo)
        else:
            m = inside & (score_norm > lo) & (score_norm <= hi)
        out[m] = colors_bgr[i]
    return out


def get_presence_colors(num_bins: int) -> List[Tuple[int, int, int]]:
    """Return low->high colors in BGR, always ending with red for highest severity."""
    if num_bins < 2:
        num_bins = 2

    # Low -> high severity anchors (BGR): blue, cyan, green, yellow, orange, red
    anchors = [
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 165, 255),
        (0, 0, 255),
    ]

    if num_bins >= len(anchors):
        return anchors[:num_bins]

    if num_bins == 2:
        return [anchors[0], anchors[-1]]

    last = len(anchors) - 1
    idxs = sorted({int(round(i * last / (num_bins - 1))) for i in range(num_bins)})

    # Ensure index count exactly equals num_bins while keeping last color as red.
    while len(idxs) < num_bins:
        for candidate in range(last):
            if candidate not in idxs:
                idxs.insert(-1, candidate)
                if len(idxs) == num_bins:
                    break

    idxs = idxs[: num_bins - 1] + [last]
    return [anchors[i] for i in idxs]


def resolve_presence_thresholds(
    score_norm: np.ndarray,
    region_mask: np.ndarray,
    mode: str,
    cuts: List[float],
) -> tuple[List[float], List[float]]:
    """Resolve Presence split lines to actual score thresholds.

    Returns:
        thresholds: actual score thresholds in [0, 1]
        used_cuts: configured cut values (threshold mode: 0-1, quantile mode: 0-100)
    """
    inside = region_mask > 0
    if not np.any(inside):
        return [], []

    if mode == "quantile":
        used_cuts = sorted([float(np.clip(v, 0.0, 100.0)) for v in cuts])
        vals = score_norm[inside]
        thresholds = [float(np.quantile(vals, q / 100.0)) for q in used_cuts]
        return thresholds, used_cuts

    used_cuts = sorted([float(np.clip(v, 0.0, 1.0)) for v in cuts])
    thresholds = used_cuts
    return thresholds, used_cuts


def add_colorbar_right_bgr(
    image_bgr: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
    bar_width: int = 26,
    pad: int = 10,
) -> np.ndarray:
    """Append a vertical jet color bar to the right side of an image (BGR)."""
    h, w = image_bgr.shape[:2]

    # Build top(high)->bottom(low) gradient and convert from RGB to BGR.
    grad = np.linspace(vmax, vmin, h, dtype=np.float32)[:, None]
    bar_rgb = (cm.get_cmap("jet")(grad)[:, :, :3] * 255).astype(np.uint8)
    bar_rgb = np.repeat(bar_rgb, bar_width, axis=1)
    bar_bgr = cv2.cvtColor(bar_rgb, cv2.COLOR_RGB2BGR)

    canvas_w = w + pad * 3 + bar_width + 44
    canvas = np.full((h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[:, :w] = image_bgr

    x0 = w + pad
    x1 = x0 + bar_width
    canvas[:, x0:x1] = bar_bgr

    # Draw border and ticks.
    cv2.rectangle(canvas, (x0, 0), (x1 - 1, h - 1), (0, 0, 0), 1)
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t in ticks:
        y = int(round((1.0 - t) * (h - 1)))
        cv2.line(canvas, (x1 + 2, y), (x1 + 8, y), (0, 0, 0), 1)
        cv2.putText(
            canvas,
            f"{t:.2f}",
            (x1 + 12, max(10, min(h - 4, y + 4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


def run_heatmap_and_worst(
    image_path: Path,
    case_id: str,
    output_folder: str,
    original_name: str | None,
    model_path: Path,
    target_class: int,
    fixed_radius: int | None,
    density_weight: float,
    consistency_weight: float,
    heat_alpha: float,
    presence_mode: str,
    presence_cuts: List[float],
    box_size: int | None,
    num_boxes: int,
    min_overlap: float,
    area_percentile: float,
    local_direction_radius: int,
    device: str,
) -> dict:
    local_direction_radius = _clamp_int(int(local_direction_radius), 1, 240)
    out_dir = OUTPUT_DIR / output_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    original_copy = copy_original_to_output(image_path, out_dir, original_name)

    texture_path = TEXTURE_OUTPUT_DIR / f"only_texture_line_{case_id}.png"
    if not texture_path.exists():
        raise FileNotFoundError("Run the full pipeline first; the texture image has not been generated.")

    result = compute_score_map_custom(
        image_path=image_path,
        model_path=model_path,
        texture_path=texture_path,
        target_class=target_class,
        fixed_radius=fixed_radius,
        density_weight=density_weight,
        consistency_weight=consistency_weight,
        device=device,
    )

    original = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"Cannot read the original image: {image_path}")
    if original.shape[:2] != result["score_norm"].shape:
        original = cv2.resize(original, (result["score_norm"].shape[1], result["score_norm"].shape[0]))

    heatmap_rgb = make_heatmap_image(result["score_norm"], result["region_mask"])
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    severity_overlay = overlay_heatmap(original, heatmap_bgr, result["region_mask"], alpha=heat_alpha)

    # Add right-side color bars for readability.
    heatmap_bgr_with_bar = add_colorbar_right_bgr(heatmap_bgr, vmin=0.0, vmax=1.0)
    severity_overlay_with_bar = add_colorbar_right_bgr(severity_overlay, vmin=0.0, vmax=1.0)

    presence_thresholds, used_cuts = resolve_presence_thresholds(
        result["score_norm"],
        result["region_mask"],
        mode=presence_mode,
        cuts=presence_cuts,
    )

    # Fixed palette low->high: blue, cyan, green, yellow, orange, red
    colors = get_presence_colors(len(presence_thresholds) + 1)
    presence_map = build_presence_map_custom(result["score_norm"], result["region_mask"], presence_thresholds, colors)
    presence_overlay = original.copy().astype(np.float32)
    inside = result["region_mask"] > 0
    for c in range(3):
        presence_overlay[:, :, c][inside] = 0.50 * presence_overlay[:, :, c][inside] + 0.50 * presence_map[:, :, c][inside]
    presence_overlay = np.clip(presence_overlay, 0, 255).astype(np.uint8)

    cv2.imwrite(str(out_dir / "severity_map.png"), heatmap_bgr_with_bar)
    cv2.imwrite(str(out_dir / "severity_overlay.png"), severity_overlay_with_bar)
    cv2.imwrite(str(out_dir / "presence_map.png"), presence_map)
    cv2.imwrite(str(out_dir / "presence_overlay.png"), presence_overlay)

    if box_size is None:
        used_box_size = scaled_box_size_from_shape(result["region_mask"].shape[0], result["region_mask"].shape[1])
    else:
        used_box_size = int(box_size)

    boxes = []
    forbidden = []
    area_masks = []
    seed_points = []
    forbidden_area = np.zeros_like(result["region_mask"], dtype=np.uint8)
    vals_inside = result["score_norm"][result["region_mask"] > 0]
    area_percentile = _clamp_float(float(area_percentile), 0.0, 100.0)
    area_threshold = float(np.quantile(vals_inside, area_percentile / 100.0)) if vals_inside.size > 0 else 0.0
    for _ in range(num_boxes):
        try:
            box = find_worst_box(
                result["score_norm"],
                result["region_mask"],
                box_size=used_box_size,
                min_overlap_ratio=min_overlap,
                forbidden_boxes=forbidden,
                forbidden_area_mask=forbidden_area,
            )
        except RuntimeError:
            break
        boxes.append(box)
        forbidden.append(box)
        area_mask, seed_pt, _ = extract_connected_high_area(
            result["score_norm"],
            result["region_mask"],
            box,
            area_threshold,
        )
        area_masks.append(area_mask)
        seed_points.append(seed_pt)
        forbidden_area[area_mask > 0] = 1

    if len(boxes) == 0:
        raise RuntimeError("No worst box satisfies the connected-area constraint. Adjust the parameters and retry.")

    # Remove stale worst-box artifacts so UI always reflects the current run.
    for stale in out_dir.glob(f"{image_path.stem}_worst*_box.png"):
        stale.unlink(missing_ok=True)
    for stale in out_dir.glob(f"{image_path.stem}_worst*_direction.png"):
        stale.unlink(missing_ok=True)
    for stale in out_dir.glob(f"{image_path.stem}_worst*_area.png"):
        stale.unlink(missing_ok=True)
    for stale in out_dir.glob(f"{image_path.stem}_worst*_info.txt"):
        stale.unlink(missing_ok=True)

    importlib.reload(worst_box_direction_module)
    worst_box_direction_module.draw_outputs(
        image_path=image_path,
        out_dir=out_dir,
        box_size=used_box_size,
        boxes=boxes,
        pred=result["pred"],
        region_mask=result["region_mask"],
        area_masks=area_masks,
        seed_points=seed_points,
        area_threshold=area_threshold,
        area_percentile=area_percentile,
        orientations=result["orientations"],
        density_map=result["density"],
        consistency_map=result["consistency"],
        local_direction_radius=local_direction_radius,
    )

    return {
        "radius": result["radius"],
        "box_size": used_box_size,
        "out_dir": out_dir,
        "original_copy": original_copy,
        "presence_mode": presence_mode,
        "presence_cuts": used_cuts,
        "presence_thresholds": presence_thresholds,
        "area_percentile": area_percentile,
        "area_threshold": area_threshold,
        "local_direction_radius": local_direction_radius,
    }


def run_full_pipeline(
    image_path: Path,
    case_id: str,
    output_folder: str,
    original_name: str | None,
    model_path: Path,
    device: str,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUT_DIR / output_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    original_copy = copy_original_to_output(image_path, out_dir, original_name)

    pred = predict_mask(str(image_path), str(model_path), device)
    original = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"Cannot read the original image: {image_path}")
    region_mask_for_boundary = build_effective_texture_region_mask((pred == 1).astype(np.uint8))
    filled, boundary, seg_overlay = make_segmentation_visuals(
        original,
        pred,
        region_boundary_mask=region_mask_for_boundary,
    )
    cv2.imwrite(str(out_dir / "segmentation_filled.png"), cv2.cvtColor(filled, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "segmentation_boundary.png"), cv2.cvtColor(boundary, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "segmentation_overlay.png"), cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR))

    analyze_skin_texture(str(image_path), model_path=str(model_path), device=device)
    texture_path = TEXTURE_OUTPUT_DIR / f"only_texture_line_{case_id}.png"
    if not texture_path.exists():
        raise FileNotFoundError(f"Texture image was not generated: {texture_path}")

    analyze_texture_orientation(str(texture_path), include_sector_analysis=False)
    orientation_local = resolve_predict_output(case_id, "orientation_local_texture_line_")
    orientation_full = resolve_predict_output(case_id, "orientation_texture_line_")

    texture_direction_overlay = out_dir / "texture_direction_overlay.png"
    overlay_images_unicode(str(image_path), str(orientation_local), str(texture_direction_overlay))

    return {
        "out_dir": out_dir,
        "original_copy": original_copy,
        "texture_only": texture_path,
        "texture_compare": TEXTURE_OUTPUT_DIR / f"texture_overlay_{case_id}.png",
        "orientation_local": orientation_local,
        "orientation_full": orientation_full,
        "texture_direction_overlay": texture_direction_overlay,
    }


def show_image_if_exists(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Not found: {path.name}")


def show_image_with_explain(path: Path, title: str, explain: str):
    st.markdown(f"**{title}**")
    if path.exists():
        st.image(str(path), use_container_width=True)
        st.caption(explain)
    else:
        st.warning(f"Not found: {path.name}")
        st.caption(explain)


def latest_result_file(out_dir: Path, pattern: str) -> Path | None:
    matches = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def resolve_texture_overlay(case_id: str, output_folder: str | None = None) -> Path:
    overlay_path = TEXTURE_OUTPUT_DIR / f"texture_overlay_{case_id}.png"
    texture_path = TEXTURE_OUTPUT_DIR / f"only_texture_line_{case_id}.png"
    out_dir = OUTPUT_DIR / output_folder if output_folder else None
    original_path = latest_result_file(out_dir, "00_original__*") if out_dir is not None else None
    if original_path is None:
        input_candidates = sorted(INPUT_DIR.glob(f"{case_id}.*"), key=lambda p: p.stat().st_mtime)
        original_path = input_candidates[-1] if input_candidates else None

    original = imread_unicode(original_path, cv2.IMREAD_COLOR) if original_path is not None else None
    texture_lines = imread_unicode(texture_path, cv2.IMREAD_GRAYSCALE)
    if original is not None and texture_lines is not None:
        if texture_lines.shape != original.shape[:2]:
            texture_lines = cv2.resize(
                texture_lines,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        line_mask = texture_lines > 0
        red_layer = np.zeros_like(original)
        red_layer[line_mask] = (0, 0, 255)
        overlay = original.copy()
        blended = cv2.addWeighted(original, 0.70, red_layer, 0.30, 0)
        overlay[line_mask] = blended[line_mask]

        ok, encoded = cv2.imencode(".png", overlay)
        if ok:
            safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
            cache_paths = (overlay_path, Path("/tmp") / f"ketas_texture_overlay_{safe_case_id}.png")
            for cache_path in cache_paths:
                try:
                    encoded.tofile(str(cache_path))
                except OSError:
                    continue
                return cache_path

    if overlay_path.exists():
        return overlay_path
    comparison_path = TEXTURE_OUTPUT_DIR / f"texture_line_{case_id}.png"
    return comparison_path if comparison_path.exists() else overlay_path


def case_result_images(case_id: str, output_folder: str) -> dict[str, tuple[Path | None, str, str]]:
    out_dir = OUTPUT_DIR / output_folder
    try:
        orientation_full = resolve_predict_output(case_id, "orientation_texture_line_")
    except Exception:
        orientation_full = ORIENTATION_OUTPUT_DIR / f"orientation_texture_line_{case_id}.png"

    return {
        "original_image": (
            latest_result_file(out_dir, "00_original__*"),
            "Original Image (Output Copy)",
            "Input image copied into the case output directory.",
        ),
        "segmentation_overlay": (
            out_dir / "segmentation_overlay.png",
            "Three-Class Segmentation Overlay",
            "Class boundaries overlaid on the original image.",
        ),
        "segmentation_filled": (
            out_dir / "segmentation_filled.png",
            "Three-Class Segmentation",
            "Filled normal-skin, intertidal-zone, and keloid-body classes.",
        ),
        "texture_lines": (
            TEXTURE_OUTPUT_DIR / f"only_texture_line_{case_id}.png",
            "Texture Lines",
            "Extracted texture-line signal used by subsequent calculations.",
        ),
        "texture_comparison": (
            resolve_texture_overlay(case_id, output_folder),
            "Texture Line Overlay",
            "Extracted texture lines overlaid on the original image.",
        ),
        "pixel_texture_axis_map": (
            orientation_full,
            "Pixel-Level Texture-Axis Map",
            "Local texture axes estimated from the structure tensor.",
        ),
        "texture_axis_overlay": (
            out_dir / "texture_direction_overlay.png",
            "Texture-Axis Overlay",
            "Texture lines and local texture axes overlaid on the input image.",
        ),
        "severity_map": (
            out_dir / "severity_map.png",
            "Severity Map",
            "Final continuous severity output from local density and orientation consistency.",
        ),
        "severity_map_overlay": (
            out_dir / "severity_overlay.png",
            "Severity Map Overlay",
            "Severity Map overlaid on the original image.",
        ),
        "presence_map": (
            out_dir / "presence_map.png",
            "Presence Level",
            "Discrete presence levels derived from the Severity Map.",
        ),
        "presence_level_overlay": (
            out_dir / "presence_overlay.png",
            "Presence Level Overlay",
            "Discrete Presence Levels overlaid on the original image.",
        ),
        "worst_area_search_boxes": (
            latest_result_file(out_dir, "*_worst*_box.png"),
            "Worst Area Search Boxes",
            "Candidate search regions selected from high-severity locations.",
        ),
        "presence_map_overlay": (
            latest_result_file(out_dir, "*_worst*_area.png"),
            "Presence Map Overlay",
            "Closed high-severity connected regions overlaid on the original image.",
        ),
        "regional_mean_direction_map": (
            latest_result_file(out_dir, "*_worst*_area_direction.png"),
            "Regional Mean Direction Map",
            "Direction arrows computed from each complete selected region.",
        ),
        "local_direction_map": (
            latest_result_file(out_dir, "*_worst*_area_local_direction.png"),
            "Local Direction Map",
            "Final direction output computed around each constrained arrow origin.",
        ),
    }


def render_analysis_flow_stage(
    stage_number: int,
    title: str,
    keys: tuple[str, ...],
    images: dict[str, tuple[Path | None, str, str]],
    display_settings: dict,
    final_output_key: str | None = None,
) -> bool:
    visible_keys = [key for key in keys if display_enabled(display_settings, "images", key)]
    if not visible_keys:
        return False

    final_label = " · Final Output" if final_output_key in visible_keys else ""
    st.markdown(f"### {stage_number}. {title}{final_label}")
    columns = st.columns(min(3, len(visible_keys)))
    for index, key in enumerate(visible_keys):
        path, image_title, explanation = images[key]
        with columns[index % len(columns)]:
            st.markdown(f"**{image_title}**")
            if path is not None and path.exists():
                st.image(str(path), width="stretch")
            else:
                missing_name = path.name if path is not None else image_title
                st.warning(f"Not found: {missing_name}")
            st.caption(explanation)
    return True


def selected_direction_result_key(display_settings: dict) -> str | None:
    if display_enabled(display_settings, "images", "local_direction_map"):
        return "local_direction_map"
    if display_enabled(display_settings, "images", "regional_mean_direction_map"):
        return "regional_mean_direction_map"
    return None


def render_analysis_overview(
    case_id: str,
    output_folder: str,
    display_settings: dict,
    image_size_text: str | None = None,
) -> None:
    out_dir = OUTPUT_DIR / output_folder
    st.subheader("Analysis Results")
    if image_size_text:
        st.caption(f"Input image size: {image_size_text}")
    st.caption(f"Output directory: runtime/web_outputs/{output_folder}")

    images = case_result_images(case_id, output_folder)
    direction_result_key = selected_direction_result_key(display_settings)
    stages = (
        (1, "Input", ("original_image",), None),
        (2, "Three-Class Segmentation", ("segmentation_overlay", "segmentation_filled"), None),
        (3, "Texture Extraction", ("texture_lines", "texture_comparison"), None),
        (4, "Texture-Axis Estimation", ("pixel_texture_axis_map", "texture_axis_overlay"), None),
        (5, "Severity Map", ("severity_map", "severity_map_overlay"), "severity_map_overlay"),
        (
            6,
            "Presence Level",
            ("presence_map", "presence_level_overlay", "worst_area_search_boxes", "presence_map_overlay"),
            "presence_map_overlay",
        ),
        (7, "Direction Map", ("regional_mean_direction_map", "local_direction_map"), direction_result_key),
    )

    rendered_any = False
    for stage_number, title, keys, final_output_key in stages:
        stage_visible = any(display_enabled(display_settings, "images", key) for key in keys)
        if rendered_any and stage_visible:
            st.markdown("<div style='text-align:center;font-size:2rem;line-height:1'>↓</div>", unsafe_allow_html=True)
        rendered_any = render_analysis_flow_stage(
            stage_number,
            title,
            keys,
            images,
            display_settings,
            final_output_key,
        ) or rendered_any

    if display_enabled(display_settings, "images", "worst_area_parameters"):
        info_file = latest_result_file(out_dir, "*_worst*_info.txt")
        st.markdown("### Supporting Data")
        st.markdown("**Worst Area Parameters**")
        if info_file is not None:
            st.code(info_file.read_text(encoding="utf-8"), language="text")
        else:
            st.warning("Worst Area Parameters were not found.")

    if not rendered_any:
        st.info("No result images are enabled. Select images on the Display Settings page.")


OVERVIEW_IMAGE_LABELS = {
    "original_image": "Original Image",
    "segmentation_overlay": "Segmentation Overlay",
    "segmentation_filled": "Segmentation Mask",
    "texture_lines": "Texture Lines",
    "texture_comparison": "Texture Line Overlay",
    "pixel_texture_axis_map": "Pixel Texture Axis",
    "texture_axis_overlay": "Texture-Axis Overlay",
    "severity_map": "Severity Map",
    "severity_map_overlay": "Severity Overlay",
    "presence_map": "Presence Level",
    "presence_level_overlay": "Presence-Level Overlay",
    "worst_area_search_boxes": "Worst Area Search",
    "presence_map_overlay": "Presence Area Overlay",
    "regional_mean_direction_map": "Regional Direction",
    "local_direction_map": "Direction Map",
}


def draw_fitted_text(
    canvas: np.ndarray,
    text: str,
    center_x: int,
    baseline_y: int,
    max_width: int,
    initial_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    scale = initial_scale
    while scale > 0.28:
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if width <= max_width:
            break
        scale -= 0.03
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        canvas,
        text,
        (int(center_x - width / 2), baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def paste_overview_thumbnail(canvas: np.ndarray, path: Path | None, box: tuple[int, int, int, int]) -> None:
    x, y, width, height = box
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (218, 222, 226), 1, cv2.LINE_AA)
    image = imread_unicode(path, cv2.IMREAD_COLOR) if path is not None and path.exists() else None
    if image is None:
        draw_fitted_text(canvas, "Image not found", x + width // 2, y + height // 2, width - 12, 0.42, (110, 110, 110))
        return

    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    left = x + (width - resized_width) // 2
    top = y + (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized


def draw_overview_card(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    stage_number: int,
    title: str,
    keys: tuple[str, ...],
    images: dict[str, tuple[Path | None, str, str]],
    display_settings: dict,
) -> None:
    x, y, width, height = box
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (122, 132, 143), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + width, y + 54), (242, 238, 232), -1)
    draw_fitted_text(canvas, f"{stage_number}. {title}", x + width // 2, y + 35, width - 18, 0.56, (45, 52, 60), 1)

    visible_keys = [key for key in keys if display_enabled(display_settings, "images", key)]
    if not visible_keys:
        draw_fitted_text(canvas, "No image selected", x + width // 2, y + height // 2, width - 24, 0.43, (130, 130, 130))
        return

    content_top = y + 64
    content_height = height - 74
    slot_height = content_height // len(visible_keys)
    for index, key in enumerate(visible_keys):
        slot_top = content_top + index * slot_height
        label_height = 24
        thumbnail_height = max(42, slot_height - label_height - 5)
        paste_overview_thumbnail(
            canvas,
            images[key][0],
            (x + 10, slot_top, width - 20, thumbnail_height),
        )
        draw_fitted_text(
            canvas,
            OVERVIEW_IMAGE_LABELS[key],
            x + width // 2,
            slot_top + thumbnail_height + 18,
            width - 18,
            0.42,
            (55, 60, 66),
        )


def compose_process_overview(
    case_id: str,
    output_folder: str,
    display_settings: dict,
) -> np.ndarray:
    canvas_width, canvas_height = 1920, 940
    canvas = np.full((canvas_height, canvas_width, 3), (247, 248, 250), dtype=np.uint8)
    images = case_result_images(case_id, output_folder)
    direction_result_key = selected_direction_result_key(display_settings)
    direction_process_keys = tuple(
        key
        for key in ("regional_mean_direction_map", "local_direction_map")
        if key != direction_result_key
    )
    stages = (
        ("Input", ("original_image",)),
        ("Segmentation", ("segmentation_overlay", "segmentation_filled")),
        ("Texture Extraction", ("texture_lines", "texture_comparison")),
        ("Texture Axis", ("pixel_texture_axis_map", "texture_axis_overlay")),
        ("Severity Analysis", ("severity_map",)),
        (
            "Presence Level",
            ("presence_map", "presence_level_overlay", "worst_area_search_boxes"),
        ),
        ("Direction Analysis", direction_process_keys),
    )
    result_specs = [
        ("severity_map_overlay", "SEVERITY MAP OVERLAY", (60, 110, 220), 4),
        ("presence_map_overlay", "PRESENCE MAP OVERLAY", (35, 165, 225), 5),
    ]
    if direction_result_key is not None:
        direction_title = (
            "LOCAL DIRECTION MAP"
            if direction_result_key == "local_direction_map"
            else "REGIONAL MEAN DIRECTION MAP"
        )
        result_specs.append((direction_result_key, direction_title, (175, 95, 145), 6))

    draw_fitted_text(canvas, "KeTAS IMAGE-GENERATION FLOW", canvas_width // 2, 38, 1000, 0.82, (37, 43, 51), 2)
    cv2.putText(canvas, "PROCESS", (22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 98, 108), 1, cv2.LINE_AA)

    margin_x, gap = 20, 24
    stage_y, stage_height = 78, 430
    stage_width = (canvas_width - 2 * margin_x - gap * (len(stages) - 1)) // len(stages)
    stage_boxes = [
        (margin_x + index * (stage_width + gap), stage_y, stage_width, stage_height)
        for index in range(len(stages))
    ]

    for left_box, right_box in zip(stage_boxes, stage_boxes[1:]):
        start = (left_box[0] + left_box[2] + 3, stage_y + stage_height // 2)
        end = (right_box[0] - 3, stage_y + stage_height // 2)
        cv2.arrowedLine(canvas, start, end, (92, 101, 112), 3, cv2.LINE_AA, tipLength=0.35)

    result_region = (210, 615, 1500, 300)
    rx, ry, rw, rh = result_region
    cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (238, 241, 246), -1)
    cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (126, 136, 150), 2, cv2.LINE_AA)
    draw_fitted_text(canvas, "RESULTS", canvas_width // 2, ry + 34, 300, 0.72, (42, 49, 58), 2)

    visible_results = [spec for spec in result_specs if display_enabled(display_settings, "images", spec[0])]
    result_boxes: list[tuple[int, int, int, int]] = []
    if visible_results:
        inner_margin, result_gap = 30, 30
        result_width = min(440, (rw - 2 * inner_margin - result_gap * (len(visible_results) - 1)) // len(visible_results))
        total_width = result_width * len(visible_results) + result_gap * (len(visible_results) - 1)
        result_start_x = rx + (rw - total_width) // 2
        result_boxes = [
            (result_start_x + index * (result_width + result_gap), ry + 50, result_width, rh - 65)
            for index in range(len(visible_results))
        ]

        for index, ((key, _, color, source_stage), result_box) in enumerate(zip(visible_results, result_boxes)):
            source_box = stage_boxes[source_stage]
            source = (source_box[0] + source_box[2] // 2, source_box[1] + source_box[3])
            target_x = result_box[0] + result_box[2] // 2
            lane_y = 535 + index * 22
            cv2.line(canvas, source, (source[0], lane_y), color, 4, cv2.LINE_AA)
            cv2.line(canvas, (source[0], lane_y), (target_x, lane_y), color, 4, cv2.LINE_AA)
            cv2.arrowedLine(
                canvas,
                (target_x, lane_y),
                (target_x, result_box[1] - 5),
                color,
                4,
                cv2.LINE_AA,
                tipLength=0.12,
            )
    else:
        draw_fitted_text(
            canvas,
            "Final outputs are hidden by Display Settings",
            canvas_width // 2,
            ry + rh // 2,
            rw - 80,
            0.62,
            (105, 110, 118),
        )

    for index, (title, keys) in enumerate(stages):
        draw_overview_card(canvas, stage_boxes[index], index + 1, title, keys, images, display_settings)

    for (key, title, color, _), box in zip(visible_results, result_boxes):
        x, y, width, height = box
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 4, cv2.LINE_AA)
        draw_fitted_text(canvas, title, x + width // 2, y + 31, width - 20, 0.62, color, 2)
        paste_overview_thumbnail(canvas, images[key][0], (x + 12, y + 43, width - 24, height - 55))

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def render_process_overview(
    case_id: str,
    output_folder: str,
    display_settings: dict,
    image_size_text: str | None = None,
) -> None:
    st.subheader("Process Overview")
    details = f"Input: {image_size_text} · " if image_size_text else ""
    st.caption(f"{details}Output: runtime/web_outputs/{output_folder} · Visibility follows Display Settings")
    overview = compose_process_overview(case_id, output_folder, display_settings)
    st.image(overview, width="stretch")


def render_case_results(
    case_id: str,
    output_folder: str,
    display_settings: dict,
    image_size_text: str | None = None,
):
    out_dir = OUTPUT_DIR / output_folder
    if image_size_text:
        st.caption(f"Input image size: {image_size_text}")
    st.caption(f"Output directory: runtime/web_outputs/{output_folder}")

    intermediate_keys = (
        "original_image",
        "segmentation_overlay",
        "segmentation_filled",
        "texture_lines",
        "texture_comparison",
        "pixel_texture_axis_map",
        "texture_axis_overlay",
    )
    if any(display_enabled(display_settings, "images", key) for key in intermediate_keys):
        st.subheader("Intermediate Results")

    if display_enabled(display_settings, "images", "original_image"):
        copied_original = sorted(out_dir.glob("00_original__*"), key=lambda p: p.stat().st_mtime)
        if copied_original:
            show_image_with_explain(
                copied_original[-1],
                "Original Image (Output Copy)",
                "A copy of the uploaded image stored in runtime/web_outputs for review and result comparison.",
            )
    if display_enabled(display_settings, "images", "segmentation_overlay"):
        show_image_with_explain(out_dir / "segmentation_overlay.png", "Three-Class Segmentation Overlay", "Overlays normal skin, intertidal zone, and keloid body boundaries on the original image.")
    if display_enabled(display_settings, "images", "segmentation_filled"):
        show_image_with_explain(out_dir / "segmentation_filled.png", "Three-Class Segmentation", "Fills normal skin, intertidal zone, and keloid body with distinct colors.")
    if display_enabled(display_settings, "images", "texture_lines"):
        show_image_with_explain(TEXTURE_OUTPUT_DIR / f"only_texture_line_{case_id}.png", "Texture Lines", "Contains only the extracted texture-line signal used for orientation analysis and local scoring.")
    if display_enabled(display_settings, "images", "texture_comparison"):
        show_image_with_explain(resolve_texture_overlay(case_id, output_folder), "Texture Line Overlay", "Overlays the extracted texture lines on the original image.")
    if display_enabled(display_settings, "images", "pixel_texture_axis_map"):
        try:
            orientation_full_show = resolve_predict_output(case_id, "orientation_texture_line_")
        except Exception:
            orientation_full_show = ORIENTATION_OUTPUT_DIR / f"orientation_texture_line_{case_id}.png"
        show_image_with_explain(orientation_full_show, "Pixel-Level Texture-Axis Map", "Shows structure-tensor texture axes in the image context.")
    if display_enabled(display_settings, "images", "texture_axis_overlay"):
        show_image_with_explain(
            out_dir / "texture_direction_overlay.png",
            "Texture-Axis Overlay",
            "Overlays semi-transparent red texture lines and blue local texture-axis markers on the original image.",
        )

    indicator_keys = (
        "severity_map",
        "presence_map",
        "severity_map_overlay",
        "presence_level_overlay",
        "worst_area_search_boxes",
        "presence_map_overlay",
        "regional_mean_direction_map",
        "local_direction_map",
        "worst_area_parameters",
    )
    if any(display_enabled(display_settings, "images", key) for key in indicator_keys):
        st.subheader("Quantitative Indicators and Worst Areas")

    if display_enabled(display_settings, "images", "severity_map"):
        show_image_with_explain(out_dir / "severity_map.png", "Severity Map", "Within-image relative severity computed from Local Texture Density and Local Orientation Consistency.")
    if display_enabled(display_settings, "images", "presence_map"):
        show_image_with_explain(out_dir / "presence_map.png", "Presence Level", "A discrete-level visualization of Severity Map values using score thresholds or within-image percentiles.")
    if display_enabled(display_settings, "images", "severity_map_overlay"):
        show_image_with_explain(out_dir / "severity_overlay.png", "Severity Map Overlay", "Overlays the within-image relative Severity Map on the original image.")
    if display_enabled(display_settings, "images", "presence_level_overlay"):
        show_image_with_explain(out_dir / "presence_overlay.png", "Presence Level Overlay", "Overlays the discrete Presence Levels on the original image.")

    if display_enabled(display_settings, "images", "worst_area_search_boxes"):
        worst_box_img = sorted(out_dir.glob("*_worst*_box.png"), key=lambda p: p.stat().st_mtime)
        if worst_box_img:
            show_image_with_explain(worst_box_img[-1], "Worst Area Search Boxes", "Candidate boxes used to seed 1 to 5 non-overlapping Worst Areas within the Effective Intertidal Zone.")
    if display_enabled(display_settings, "images", "presence_map_overlay"):
        worst_area_img = sorted(out_dir.glob("*_worst*_area.png"), key=lambda p: p.stat().st_mtime)
        if worst_area_img:
            show_image_with_explain(worst_area_img[-1], "Presence Map Overlay", "Overlays each closed and externally filled 8-connected high-severity region on the original image.")
    if display_enabled(display_settings, "images", "regional_mean_direction_map"):
        worst_area_dir_img = sorted(out_dir.glob("*_worst*_area_direction.png"), key=lambda p: p.stat().st_mtime)
        if worst_area_dir_img:
            show_image_with_explain(worst_area_dir_img[-1], "Regional Mean Direction Map", "Each arrow starts at the highest-severity point in the 10-pixel inner boundary band and uses the mean texture axis of the entire Worst Area.")
    if display_enabled(display_settings, "images", "local_direction_map"):
        worst_area_local_dir_img = sorted(
            out_dir.glob("*_worst*_area_local_direction.png"),
            key=lambda p: p.stat().st_mtime,
        )
        if worst_area_local_dir_img:
            show_image_with_explain(worst_area_local_dir_img[-1], "Local Direction Map", "Uses the same constrained arrow origin and averages the texture axis within the independently adjustable Local Direction Radius.")
    if display_enabled(display_settings, "images", "worst_area_parameters"):
        info_files = sorted(out_dir.glob("*_worst*_info.txt"))
        if info_files:
            st.markdown("**Worst Area Parameters**")
            st.code(info_files[-1].read_text(encoding="utf-8"), language="text")


def app():
    st.set_page_config(page_title="Keloid Texture Analysis System (KeTAS) Demo", layout="wide")

    st.markdown(
        """
        <style>
        div[data-testid="stSidebarUserContent"] .floating-actions {
            position: sticky;
            top: 0.5rem;
            z-index: 999;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Page",
        options=["Analysis", "Process Overview", "Display Settings"],
        key="demo_page",
    )
    display_settings = load_display_settings()
    previous_page = st.session_state.get("_active_demo_page")
    if page != "Analysis":
        cache_parameter_session_state()
        st.session_state["_active_demo_page"] = page
        if page == "Display Settings":
            render_display_settings_page(display_settings)
            return

        batch_records = st.session_state.get("batch_records", [])
        if batch_records:
            options = [f"{item['index']:02d}. {item['original_name']}" for item in batch_records]
            selected = st.sidebar.selectbox(
                "Select an image to view its process",
                options=options,
                key="batch_overview_select",
            )
            selected_item = batch_records[options.index(selected)]
            render_process_overview(
                case_id=selected_item["case_id"],
                output_folder=selected_item["output_folder"],
                display_settings=display_settings,
                image_size_text=selected_item.get("image_size_text"),
            )
        elif st.session_state.get("case_id") is not None:
            case_id = st.session_state["case_id"]
            render_process_overview(
                case_id=case_id,
                output_folder=st.session_state.get("output_folder") or case_id,
                display_settings=display_settings,
                image_size_text=st.session_state.get("image_size_text"),
            )
        else:
            st.title("Process Overview")
            st.info("Run an analysis first to view its complete generation flow.")
        return
    if previous_page not in (None, "Analysis"):
        restore_cached_parameter_session_state()
    st.session_state["_active_demo_page"] = page

    st.title("Keloid Texture Analysis System (KeTAS) Demo")
    st.caption("Three-class U-Net segmentation and Gabor-gradient-domain analysis for Severity Maps, Presence Levels, and Direction Maps")

    if cv2 is None:
        st.error("OpenCV is not available in the current Python environment, so the image-processing pipeline cannot run.")
        st.code(
            "python3 -m pip install --user opencv-python\n"
            "# If libGL.so.1 is missing, run:\n"
            "sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0"
        )
        if _CV2_IMPORT_ERROR is not None:
            st.text(f"OpenCV import error: {_CV2_IMPORT_ERROR}")
        st.stop()

    init_params_state_once()

    # Apply per-image params before widgets are instantiated to avoid Streamlit key mutation errors.
    pending_params = st.session_state.get("_pending_params_to_apply")
    if isinstance(pending_params, dict):
        apply_params_to_session(normalize_params_dict(pending_params))
        st.session_state["_pending_params_to_apply"] = None
        cache_parameter_session_state()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.markdown(f"**Device**: {device}")

    if display_enabled(display_settings, "parameters", "model_file"):
        model_rel = st.sidebar.text_input(
            "Model File",
            key="p_model_rel",
            help="Path to the segmentation model weights, relative to the project root.",
        )
    else:
        model_rel = str(st.session_state["p_model_rel"])
    model_path = resolve_model_path(model_rel)
    if not model_path.exists():
        st.sidebar.error("The model file does not exist. Enable Model File in Display Settings to change it.")

    uploaded = st.file_uploader(
        "Select an Image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="Upload a keloid image to generate three-class segmentation, texture-line, Severity Map, Presence Level, Worst Area, and Direction Map results.",
    )
    batch_uploaded = st.file_uploader(
        "Select Images for Batch Processing",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="batch_uploader",
        help="Select multiple images to process them with the same parameters and save them under one batch directory.",
    )

    if "image_path" not in st.session_state:
        st.session_state["image_path"] = None
    if "case_id" not in st.session_state:
        st.session_state["case_id"] = None
    if "original_name" not in st.session_state:
        st.session_state["original_name"] = None
    if "output_folder" not in st.session_state:
        st.session_state["output_folder"] = None
    if "image_size_text" not in st.session_state:
        st.session_state["image_size_text"] = None
    if "batch_records" not in st.session_state:
        st.session_state["batch_records"] = []
    if "batch_id" not in st.session_state:
        st.session_state["batch_id"] = None
    if "_params_loaded_for_output_folder" not in st.session_state:
        st.session_state["_params_loaded_for_output_folder"] = None
    if "_pending_params_to_apply" not in st.session_state:
        st.session_state["_pending_params_to_apply"] = None

    if uploaded is not None:
        image_path, case_id, original_name, output_folder = save_uploaded_file(uploaded)
        st.session_state["image_path"] = str(image_path)
        st.session_state["case_id"] = case_id
        st.session_state["original_name"] = original_name
        st.session_state["output_folder"] = output_folder

        original_for_size = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if original_for_size is not None:
            h, w = original_for_size.shape[:2]
            st.session_state["image_size_text"] = f"{w} x {h} pixels"
        else:
            st.session_state["image_size_text"] = "Read failed"

    current_output_folder = st.session_state.get("output_folder")
    if current_output_folder and st.session_state.get("_params_loaded_for_output_folder") != current_output_folder:
        out_dir = OUTPUT_DIR / current_output_folder
        loaded_params = load_params_from_output_dir(out_dir)
        if loaded_params is None:
            loaded_params = normalize_params_dict(PARAM_DEFAULTS)
        st.session_state["_pending_params_to_apply"] = loaded_params
        st.session_state["_params_loaded_for_output_folder"] = current_output_folder
        st.rerun()

    severity_parameter_keys = (
        "target_class",
        "radius_mode",
        "heatmap_radius",
        "density_weight",
        "consistency_weight",
        "severity_overlay_opacity",
    )
    if any(display_enabled(display_settings, "parameters", key) for key in severity_parameter_keys):
        st.sidebar.header("Severity Map Parameters")

    if display_enabled(display_settings, "parameters", "target_class"):
        target_class = st.sidebar.selectbox(
            "Target Class",
            options=[1, 2],
            key="p_target_class",
            help="Select the segmented class used for local scoring. Class 1 is the Intertidal Zone and class 2 is the Keloid Body.",
        )
    else:
        target_class = int(st.session_state["p_target_class"])

    if display_enabled(display_settings, "parameters", "radius_mode"):
        radius_mode = st.sidebar.selectbox(
            "Radius Mode",
            options=["Dynamic", "Fixed"],
            key="p_radius_mode",
            help="Dynamic scales the radius with image size; Fixed uses the specified pixel radius.",
        )
    else:
        radius_mode = str(st.session_state["p_radius_mode"])
    fixed_radius = None
    if radius_mode == "Fixed":
        if display_enabled(display_settings, "parameters", "heatmap_radius"):
            fixed_radius = st.sidebar.number_input(
                "Heatmap Radius",
                min_value=8,
                max_value=120,
                step=1,
                key="p_fixed_radius",
                help="Radius of the local statistics window in pixels. Larger values are smoother; smaller values are more sensitive.",
            )
        else:
            fixed_radius = int(st.session_state["p_fixed_radius"])

    if display_enabled(display_settings, "parameters", "density_weight"):
        density_weight = st.sidebar.number_input(
            "Local Texture Density Weight",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            key="p_density_weight",
            help="Weight of local texture density in the severity score.",
        )
    else:
        density_weight = float(st.session_state["p_density_weight"])

    if display_enabled(display_settings, "parameters", "consistency_weight"):
        consistency_weight = st.sidebar.number_input(
            "Local Orientation Consistency Weight",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            key="p_consistency_weight",
            help="Weight of local orientation consistency in the severity score.",
        )
    else:
        consistency_weight = float(st.session_state["p_consistency_weight"])

    if display_enabled(display_settings, "parameters", "severity_overlay_opacity"):
        heat_alpha = st.sidebar.number_input(
            "Severity Map Overlay Opacity",
            min_value=0.1,
            max_value=0.9,
            step=0.05,
            format="%.2f",
            key="p_heat_alpha",
            help="Strength of the Severity Map colors overlaid on the original image.",
        )
    else:
        heat_alpha = float(st.session_state["p_heat_alpha"])

    presence_parameter_keys = (
        "presence_levels",
        "presence_split_mode",
        "presence_cutoffs",
        "worst_area_percentile",
    )
    if any(display_enabled(display_settings, "parameters", key) for key in presence_parameter_keys):
        st.sidebar.header("Presence Parameters")

    if display_enabled(display_settings, "parameters", "presence_levels"):
        n_bins = st.sidebar.number_input(
            "Number of Presence Levels",
            min_value=2,
            max_value=6,
            step=1,
            key="p_n_bins",
            help="Number of color-coded Presence Levels.",
        )
    else:
        n_bins = int(st.session_state["p_n_bins"])
    n_bins = int(n_bins)
    if display_enabled(display_settings, "parameters", "presence_split_mode"):
        presence_mode = st.sidebar.selectbox(
            "Presence Split Mode",
            options=["threshold", "quantile"],
            key="p_presence_mode",
            help="threshold uses fixed score cutoffs; quantile uses percentiles of the score distribution.",
        )
    else:
        presence_mode = str(st.session_state["p_presence_mode"])
    presence_cuts = []
    show_presence_cutoffs = display_enabled(display_settings, "parameters", "presence_cutoffs")
    if presence_mode == "threshold" and show_presence_cutoffs:
        st.sidebar.caption("Threshold mode: each cutoff is in [0, 1] and directly splits the normalized score.")
        for i in range(n_bins - 1):
            v = st.sidebar.number_input(
                f"Threshold {i + 1}",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key=f"p_presence_cut_{i}",
                help=f"Threshold between levels {i + 1} and {i + 2}, in the range [0, 1].",
            )
            presence_cuts.append(v)
    elif presence_mode == "quantile" and show_presence_cutoffs:
        st.sidebar.caption("Quantile mode: each cutoff is a percentile in [0, 100] based on scores inside the mask.")
        for i in range(n_bins - 1):
            q = st.sidebar.number_input(
                f"Quantile {i + 1} (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.1f",
                key=f"p_presence_cut_{i}",
                help=f"Percentile boundary between levels {i + 1} and {i + 2}, in the range [0, 100].",
            )
            presence_cuts.append(q)
    else:
        presence_cuts = [
            float(st.session_state.get(f"p_presence_cut_{i}", 0.0))
            for i in range(n_bins - 1)
        ]
    presence_cuts = _normalize_presence_cuts([float(v) for v in presence_cuts], n_bins, presence_mode)

    if display_enabled(display_settings, "parameters", "worst_area_percentile"):
        area_percentile = st.sidebar.number_input(
            "Worst Area Severity Percentile (q)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            format="%.1f",
            key="p_area_percentile",
            help="Quantile threshold used to extract connected Worst Areas. The default of 80 retains approximately the highest-scoring 20%.",
        )
    else:
        area_percentile = float(st.session_state["p_area_percentile"])

    worst_area_parameter_keys = (
        "box_size_mode",
        "box_size",
        "number_of_boxes",
        "minimum_mask_coverage",
    )
    if any(display_enabled(display_settings, "parameters", key) for key in worst_area_parameter_keys):
        st.sidebar.header("Worst Area Parameters")

    if display_enabled(display_settings, "parameters", "box_size_mode"):
        box_mode = st.sidebar.selectbox(
            "Box Size Mode",
            options=["Dynamic", "Fixed"],
            key="p_box_mode",
            help="Dynamic scales the box with image size; Fixed uses the specified box size.",
        )
    else:
        box_mode = str(st.session_state["p_box_mode"])
    box_size = None
    if box_mode == "Fixed":
        if display_enabled(display_settings, "parameters", "box_size"):
            box_size = st.sidebar.number_input(
                "Box Size",
                min_value=24,
                max_value=240,
                step=2,
                key="p_box_size",
                help="Side length of each worst-area search box in pixels.",
            )
        else:
            box_size = int(st.session_state["p_box_size"])

    if display_enabled(display_settings, "parameters", "number_of_boxes"):
        num_boxes = st.sidebar.number_input(
            "Number of Boxes",
            min_value=1,
            max_value=5,
            step=1,
            key="p_num_boxes",
            help="Number of non-overlapping high-severity boxes to select, from 1 to 5.",
        )
    else:
        num_boxes = int(st.session_state["p_num_boxes"])

    if display_enabled(display_settings, "parameters", "minimum_mask_coverage"):
        min_overlap = st.sidebar.number_input(
            "Minimum Mask Coverage",
            min_value=0.0,
            max_value=0.95,
            step=0.01,
            format="%.2f",
            key="p_min_overlap",
            help="Minimum fraction of valid-region pixels inside a candidate box. Higher values keep boxes closer to the target region.",
        )
    else:
        min_overlap = float(st.session_state["p_min_overlap"])

    if display_enabled(display_settings, "parameters", "local_direction_radius"):
        st.sidebar.header("Direction Parameters")
        local_direction_radius = st.sidebar.number_input(
            "Local Direction Radius",
            min_value=1,
            max_value=240,
            step=1,
            key="p_local_direction_radius",
            help="Radius used by the Local Direction Map to average texture axes around the arrow origin. It is independent of Heatmap Radius.",
        )
    else:
        local_direction_radius = int(st.session_state["p_local_direction_radius"])
    num_boxes = int(num_boxes)
    local_direction_radius = int(local_direction_radius)
    if fixed_radius is not None:
        fixed_radius = int(fixed_radius)
    if box_size is not None:
        box_size = int(box_size)

    if display_enabled(display_settings, "parameters", "parameter_guide"):
        with st.sidebar.expander("Parameter Guide", expanded=False):
            st.markdown(
                "- Target Class: selects the segmented region used for Severity Map computation.\n"
                "- Heatmap Radius: sets the disk neighborhood for Local Texture Density and Local Orientation Consistency.\n"
                "- Texture Points: all nonzero pixels in the unresampled binary texture-line image are used directly.\n"
                "- Density/Consistency Weights: jointly determine within-image relative severity.\n"
                "- Presence Mode: threshold uses fixed scores; quantile uses within-mask score percentiles.\n"
                "- Worst Area Parameters: control seed-box size, region count, mask coverage, and the high-severity percentile.\n"
                "- Local Direction Radius: controls texture-axis averaging around the arrow origin independently of Heatmap Radius."
            )

    st.sidebar.markdown('<div class="floating-actions">', unsafe_allow_html=True)
    st.sidebar.subheader("Run")
    run_all = st.sidebar.button("1) Run Full Pipeline", help="Recommended for the first run; generates all intermediate results and visualizations.")
    rerun_part = st.sidebar.button("2) Recompute Indicators and Areas", help="Updates Severity Maps, Presence Levels, and Worst Areas without rerunning segmentation or texture-axis extraction.")
    run_batch = st.sidebar.button("3) Process Batch", help="Processes all batch-uploaded images once using the current parameters.")
    st.sidebar.caption("This action panel remains visible while scrolling.")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["case_id"] is not None:
        st.info(f"Current case ID: {st.session_state['case_id']}")
        st.caption(f"Input image size: {st.session_state['image_size_text']}")
        st.caption(f"Output directory: runtime/web_outputs/{st.session_state['output_folder']}")

    if run_all:
        if st.session_state["image_path"] is None:
            st.error("Upload an image first.")
            st.stop()
        if not model_path.exists():
            st.error("The model path is invalid.")
            st.stop()

        st.session_state["batch_records"] = []
        st.session_state["batch_id"] = None

        params_snapshot = build_current_params_snapshot(
            model_rel=model_rel,
            target_class=target_class,
            radius_mode=radius_mode,
            fixed_radius=fixed_radius,
            density_weight=density_weight,
            consistency_weight=consistency_weight,
            heat_alpha=heat_alpha,
            n_bins=n_bins,
            presence_mode=presence_mode,
            presence_cuts=presence_cuts,
            box_mode=box_mode,
            box_size=box_size,
            num_boxes=num_boxes,
            min_overlap=min_overlap,
            area_percentile=area_percentile,
            local_direction_radius=local_direction_radius,
        )
        with st.spinner("Generating full-pipeline intermediate results..."):
            image_path = Path(st.session_state["image_path"])
            case_id = st.session_state["case_id"]
            output_folder = st.session_state["output_folder"] or case_id
            original_name = st.session_state["original_name"]
            full_info = run_full_pipeline(image_path, case_id, output_folder, original_name, model_path, device)
            heat_info = run_heatmap_and_worst(
                image_path=image_path,
                case_id=case_id,
                output_folder=output_folder,
                original_name=original_name,
                model_path=model_path,
                target_class=target_class,
                fixed_radius=fixed_radius,
                density_weight=density_weight,
                consistency_weight=consistency_weight,
                heat_alpha=heat_alpha,
                presence_mode=presence_mode,
                presence_cuts=presence_cuts,
                box_size=box_size,
                num_boxes=num_boxes,
                min_overlap=min_overlap,
                area_percentile=area_percentile,
                local_direction_radius=local_direction_radius,
                device=device,
            )
        st.success(f"Completed. radius={heat_info['radius']}, box_size={heat_info['box_size']}")
        run_params_path = save_params_to_output_dir(
            out_dir=Path(heat_info["out_dir"]),
            input_image_path=image_path,
            params_snapshot=params_snapshot,
            heat_info=heat_info,
        )
        st.caption(f"Parameters saved: {run_params_path.name}")
        if heat_info.get("presence_mode") == "quantile":
            st.caption(
                f"Presence quantiles (%): {heat_info['presence_cuts']} -> effective thresholds: "
                f"{[round(v, 4) for v in heat_info['presence_thresholds']]}"
            )
        else:
            st.caption(f"Presence thresholds: {[round(v, 4) for v in heat_info['presence_thresholds']]}")

    if rerun_part:
        if st.session_state["image_path"] is None:
            st.error("Upload an image and run the full pipeline at least once.")
            st.stop()
        if not model_path.exists():
            st.error("The model path is invalid.")
            st.stop()

        st.session_state["batch_records"] = []
        st.session_state["batch_id"] = None

        params_snapshot = build_current_params_snapshot(
            model_rel=model_rel,
            target_class=target_class,
            radius_mode=radius_mode,
            fixed_radius=fixed_radius,
            density_weight=density_weight,
            consistency_weight=consistency_weight,
            heat_alpha=heat_alpha,
            n_bins=n_bins,
            presence_mode=presence_mode,
            presence_cuts=presence_cuts,
            box_mode=box_mode,
            box_size=box_size,
            num_boxes=num_boxes,
            min_overlap=min_overlap,
            area_percentile=area_percentile,
            local_direction_radius=local_direction_radius,
        )
        with st.spinner("Recomputing Severity Maps, Presence Levels, and Worst Areas with the current parameters..."):
            image_path = Path(st.session_state["image_path"])
            case_id = st.session_state["case_id"]
            output_folder = st.session_state["output_folder"] or case_id
            original_name = st.session_state["original_name"]
            heat_info = run_heatmap_and_worst(
                image_path=image_path,
                case_id=case_id,
                output_folder=output_folder,
                original_name=original_name,
                model_path=model_path,
                target_class=target_class,
                fixed_radius=fixed_radius,
                density_weight=density_weight,
                consistency_weight=consistency_weight,
                heat_alpha=heat_alpha,
                presence_mode=presence_mode,
                presence_cuts=presence_cuts,
                box_size=box_size,
                num_boxes=num_boxes,
                min_overlap=min_overlap,
                area_percentile=area_percentile,
                local_direction_radius=local_direction_radius,
                device=device,
            )
        st.success(f"Recomputation completed. radius={heat_info['radius']}, box_size={heat_info['box_size']}")
        run_params_path = save_params_to_output_dir(
            out_dir=Path(heat_info["out_dir"]),
            input_image_path=image_path,
            params_snapshot=params_snapshot,
            heat_info=heat_info,
        )
        st.caption(f"Parameters saved: {run_params_path.name}")
        if heat_info.get("presence_mode") == "quantile":
            st.caption(
                f"Presence quantiles (%): {heat_info['presence_cuts']} -> effective thresholds: "
                f"{[round(v, 4) for v in heat_info['presence_thresholds']]}"
            )
        else:
            st.caption(f"Presence thresholds: {[round(v, 4) for v in heat_info['presence_thresholds']]}")

    if run_batch:
        if not batch_uploaded:
            st.error("Select at least one image in the batch uploader first.")
            st.stop()
        if not model_path.exists():
            st.error("The model path is invalid.")
            st.stop()

        params_snapshot = build_current_params_snapshot(
            model_rel=model_rel,
            target_class=target_class,
            radius_mode=radius_mode,
            fixed_radius=fixed_radius,
            density_weight=density_weight,
            consistency_weight=consistency_weight,
            heat_alpha=heat_alpha,
            n_bins=n_bins,
            presence_mode=presence_mode,
            presence_cuts=presence_cuts,
            box_mode=box_mode,
            box_size=box_size,
            num_boxes=num_boxes,
            min_overlap=min_overlap,
            area_percentile=area_percentile,
            local_direction_radius=local_direction_radius,
        )

        batch_id = new_batch_id()
        batch_root = OUTPUT_DIR / batch_id
        batch_items = []
        failed = []

        with st.spinner(f"Processing {len(batch_uploaded)} images..."):
            for idx, one in enumerate(batch_uploaded, start=1):
                try:
                    image_path, case_id, original_name, output_folder = save_uploaded_file(one)
                    output_folder = f"{batch_id}/{output_folder}"

                    original_for_size = imread_unicode(image_path, cv2.IMREAD_COLOR)
                    if original_for_size is not None:
                        h, w = original_for_size.shape[:2]
                        image_size_text = f"{w} x {h} pixels"
                    else:
                        image_size_text = "Read failed"

                    run_full_pipeline(image_path, case_id, output_folder, original_name, model_path, device)
                    heat_info = run_heatmap_and_worst(
                        image_path=image_path,
                        case_id=case_id,
                        output_folder=output_folder,
                        original_name=original_name,
                        model_path=model_path,
                        target_class=target_class,
                        fixed_radius=fixed_radius,
                        density_weight=density_weight,
                        consistency_weight=consistency_weight,
                        heat_alpha=heat_alpha,
                        presence_mode=presence_mode,
                        presence_cuts=presence_cuts,
                        box_size=box_size,
                        num_boxes=num_boxes,
                        min_overlap=min_overlap,
                        area_percentile=area_percentile,
                        local_direction_radius=local_direction_radius,
                        device=device,
                    )
                    save_params_to_output_dir(
                        out_dir=Path(heat_info["out_dir"]),
                        input_image_path=image_path,
                        params_snapshot=params_snapshot,
                        heat_info=heat_info,
                    )
                    batch_items.append(
                        {
                            "index": idx,
                            "original_name": original_name,
                            "case_id": case_id,
                            "output_folder": output_folder,
                            "image_size_text": image_size_text,
                        }
                    )
                except Exception as exc:
                    failed.append(f"[{idx}] {getattr(one, 'name', 'unknown')}: {exc}")

        save_batch_manifest(batch_root, batch_items)
        st.session_state["batch_records"] = batch_items
        st.session_state["batch_id"] = batch_id

        if batch_items:
            st.success(f"Batch completed: {len(batch_items)} succeeded, {len(failed)} failed.")
            st.caption(f"Batch directory: runtime/web_outputs/{batch_id}")
        else:
            st.error("Batch processing failed; no usable results were generated.")
        if failed:
            st.warning("Some images failed:")
            st.code("\n".join(failed), language="text")

    batch_records = st.session_state.get("batch_records", [])
    if batch_records:
        st.subheader("Batch Results")
        st.caption(f"Current batch directory: runtime/web_outputs/{st.session_state.get('batch_id')}")
        options = [f"{x['index']:02d}. {x['original_name']}" for x in batch_records]
        selected = st.selectbox("Select an image to view its results", options=options, key="batch_viewer_select")
        sel_idx = options.index(selected)
        sel_item = batch_records[sel_idx]
        render_analysis_overview(
            case_id=sel_item["case_id"],
            output_folder=sel_item["output_folder"],
            display_settings=display_settings,
            image_size_text=sel_item.get("image_size_text"),
        )

    if st.session_state["case_id"] is not None and not st.session_state.get("batch_records"):
        case_id = st.session_state["case_id"]
        output_folder = st.session_state.get("output_folder") or case_id
        render_analysis_overview(
            case_id,
            output_folder,
            display_settings,
            st.session_state.get("image_size_text"),
        )


if __name__ == "__main__":
    app()
