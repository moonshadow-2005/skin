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

from skin import analyze_skin_texture
from predict import analyze_texture_orientation
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
    draw_outputs,
    extract_connected_high_area,
    find_worst_box,
    mean_orientation_degrees,
    scaled_box_size_from_shape,
)
import src.worst_box_direction as worst_box_direction_module


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "web_demo_inputs"
OUTPUT_DIR = PROJECT_ROOT / "web_demo_output"

PARAM_DEFAULTS = {
    "model_rel": "best_trans_unet_model_20250614_122913.pth",
    "target_class": 1,
    "radius_mode": "动态",
    "fixed_radius": 40,
    "texture_threshold": 0.40,
    "density_weight": 0.70,
    "consistency_weight": 0.30,
    "heat_alpha": 0.55,
    "n_bins": 2,
    "presence_mode": "quantile",
    "presence_cuts": [80.0],
    "box_mode": "固定",
    "box_size": 80,
    "num_boxes": 2,
    "min_overlap": 0.30,
}


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
    data["radius_mode"] = "固定" if data.get("radius_mode") == "固定" else "动态"
    data["box_mode"] = "固定" if data.get("box_mode") == "固定" else "动态"
    data["presence_mode"] = "quantile" if data.get("presence_mode") == "quantile" else "threshold"
    data["fixed_radius"] = _clamp_int(int(data.get("fixed_radius", 40)), 8, 120)
    data["texture_threshold"] = _clamp_float(float(data.get("texture_threshold", 0.40)), 0.05, 0.95)
    data["density_weight"] = _clamp_float(float(data.get("density_weight", 0.70)), 0.0, 1.0)
    data["consistency_weight"] = _clamp_float(float(data.get("consistency_weight", 0.30)), 0.0, 1.0)
    data["heat_alpha"] = _clamp_float(float(data.get("heat_alpha", 0.55)), 0.1, 0.9)
    data["n_bins"] = _clamp_int(int(data.get("n_bins", 4)), 2, 6)
    data["box_size"] = _clamp_int(int(data.get("box_size", 80)), 24, 240)
    data["num_boxes"] = _clamp_int(int(data.get("num_boxes", 1)), 1, 5)
    data["min_overlap"] = _clamp_float(float(data.get("min_overlap", 0.30)), 0.0, 0.95)

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
    st.session_state["p_texture_threshold"] = float(params["texture_threshold"])
    st.session_state["p_density_weight"] = float(params["density_weight"])
    st.session_state["p_consistency_weight"] = float(params["consistency_weight"])
    st.session_state["p_heat_alpha"] = float(params["heat_alpha"])
    st.session_state["p_n_bins"] = int(params["n_bins"])
    st.session_state["p_presence_mode"] = str(params["presence_mode"])
    st.session_state["p_box_mode"] = str(params["box_mode"])
    st.session_state["p_box_size"] = int(params["box_size"])
    st.session_state["p_num_boxes"] = int(params["num_boxes"])
    st.session_state["p_min_overlap"] = float(params["min_overlap"])

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


def init_params_state_once() -> None:
    if st.session_state.get("_params_initialized"):
        return

    params = normalize_params_dict(PARAM_DEFAULTS)
    apply_params_to_session(params)

    st.session_state["_params_initialized"] = True


def build_current_params_snapshot(
    model_rel: str,
    target_class: int,
    radius_mode: str,
    fixed_radius: int | None,
    texture_threshold: float,
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
) -> dict:
    snapshot = {
        "model_rel": str(model_rel),
        "target_class": int(target_class),
        "radius_mode": str(radius_mode),
        "fixed_radius": int(fixed_radius) if fixed_radius is not None else int(st.session_state.get("p_fixed_radius", 40)),
        "texture_threshold": float(texture_threshold),
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
    if "area_threshold_p80" in heat_info:
        payload["area_threshold_p80"] = float(heat_info["area_threshold_p80"])

    out_path = out_dir / "run_params.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()
    return cleaned or "uploaded_image"


def imread_unicode(image_path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(image_path), dtype=np.uint8)
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
        raise FileNotFoundError(f"无法读取原图用于PNG转换: {image_path}")
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"无法将原图编码为PNG: {image_path}")
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
    predict_dir = PROJECT_ROOT / "predict_output"
    suffix = case_id.split("_")[-1]
    exact = predict_dir / f"{prefix}{case_id}.png"
    fallback = predict_dir / f"{prefix}{suffix}.png"
    matched = pick_existing_path([exact, fallback])
    if matched is not None:
        return matched

    wildcard = sorted(predict_dir.glob(f"{prefix}*{suffix}.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if wildcard:
        return wildcard[0]

    raise FileNotFoundError(f"未找到预测输出: {prefix}{case_id}.png (或兼容后缀 {suffix})")


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
    texture_threshold: float,
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
        raise RuntimeError("目标区域为空，请切换 target_class 或检查图片质量。")

    tex = imread_unicode(texture_path, cv2.IMREAD_GRAYSCALE)
    if tex is None:
        raise FileNotFoundError(f"纹理图不存在: {texture_path}")
    if tex.shape != pred.shape:
        tex = cv2.resize(tex, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    h, w = tex.shape
    radius = dynamic_radius_from_size(h, w) if fixed_radius is None else int(fixed_radius)
    kernel = build_disk_kernel(radius)

    tex_norm = tex.astype(np.float32) / 255.0
    texture_binary = ((tex_norm > texture_threshold) & (region_mask > 0)).astype(np.float32)
    region_mask_f = region_mask.astype(np.float32)

    valid_count = cv2.filter2D(region_mask_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    texture_count = cv2.filter2D(texture_binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    density = np.zeros_like(tex_norm, dtype=np.float32)
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

    consistency = np.zeros_like(tex_norm, dtype=np.float32)
    ok = cnt > 1e-6
    mean_cos = np.zeros_like(tex_norm, dtype=np.float32)
    mean_sin = np.zeros_like(tex_norm, dtype=np.float32)
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
        raise ValueError("颜色数量必须等于阈值数量+1")

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
    texture_threshold: float,
    density_weight: float,
    consistency_weight: float,
    heat_alpha: float,
    presence_mode: str,
    presence_cuts: List[float],
    box_size: int | None,
    num_boxes: int,
    min_overlap: float,
    device: str,
) -> dict:
    out_dir = OUTPUT_DIR / output_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    original_copy = copy_original_to_output(image_path, out_dir, original_name)

    texture_path = PROJECT_ROOT / "skin_output" / f"only_texture_line_{case_id}.png"
    if not texture_path.exists():
        raise FileNotFoundError("请先生成全流程中间结果（纹理图尚未生成）。")

    result = compute_score_map_custom(
        image_path=image_path,
        model_path=model_path,
        texture_path=texture_path,
        target_class=target_class,
        fixed_radius=fixed_radius,
        texture_threshold=texture_threshold,
        density_weight=density_weight,
        consistency_weight=consistency_weight,
        device=device,
    )

    original = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"无法读取原图: {image_path}")
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
    means = []
    forbidden = []
    area_masks = []
    seed_points = []
    forbidden_area = np.zeros_like(result["region_mask"], dtype=np.uint8)
    vals_inside = result["score_norm"][result["region_mask"] > 0]
    area_threshold = float(np.quantile(vals_inside, 0.80)) if vals_inside.size > 0 else 0.0
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
        means.append(mean_orientation_degrees(result["orientations"], result["region_mask"], box))
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
        raise RuntimeError("在联通区域约束下未找到可用最严重框，请调整参数后重试。")

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
        mean_degs=means,
        pred=result["pred"],
        region_mask=result["region_mask"],
        area_masks=area_masks,
        seed_points=seed_points,
        area_threshold=area_threshold,
    )

    return {
        "radius": result["radius"],
        "box_size": used_box_size,
        "out_dir": out_dir,
        "original_copy": original_copy,
        "presence_mode": presence_mode,
        "presence_cuts": used_cuts,
        "presence_thresholds": presence_thresholds,
        "area_threshold_p80": area_threshold,
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
        raise FileNotFoundError(f"无法读取原图: {image_path}")
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
    texture_path = PROJECT_ROOT / "skin_output" / f"only_texture_line_{case_id}.png"
    if not texture_path.exists():
        raise FileNotFoundError(f"纹理图未生成: {texture_path}")

    analyze_texture_orientation(str(texture_path))
    orientation_only = resolve_predict_output(case_id, "orientation_only_texture_line_")
    orientation_full = resolve_predict_output(case_id, "orientation_texture_line_")
    sector_vis = resolve_predict_output(case_id, "spatial_sector_directions_")

    final_overlay = out_dir / "final_overlay.png"
    overlay_images_unicode(str(image_path), str(orientation_only), str(final_overlay))

    return {
        "out_dir": out_dir,
        "original_copy": original_copy,
        "texture_only": texture_path,
        "texture_compare": PROJECT_ROOT / "skin_output" / f"texture_line_{case_id}.png",
        "orientation_only": orientation_only,
        "orientation_full": orientation_full,
        "sector_vis": sector_vis,
        "final_overlay": final_overlay,
    }


def show_image_if_exists(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"未找到: {path.name}")


def show_image_with_explain(path: Path, title: str, explain: str):
    st.markdown(f"**{title}**")
    if path.exists():
        st.image(str(path), use_container_width=True)
        st.caption(explain)
    else:
        st.warning(f"未找到: {path.name}")
        st.caption(explain)


def render_case_results(case_id: str, output_folder: str, image_size_text: str | None = None):
    out_dir = OUTPUT_DIR / output_folder
    if image_size_text:
        st.caption(f"当前输入图片尺寸: {image_size_text}")
    st.caption(f"当前输出目录: web_demo_output/{output_folder}")
    try:
        orientation_only_show = resolve_predict_output(case_id, "orientation_only_texture_line_")
    except Exception:
        orientation_only_show = PROJECT_ROOT / "predict_output" / f"orientation_only_texture_line_{case_id}.png"
    try:
        orientation_full_show = resolve_predict_output(case_id, "orientation_texture_line_")
    except Exception:
        orientation_full_show = PROJECT_ROOT / "predict_output" / f"orientation_texture_line_{case_id}.png"
    try:
        sector_vis_show = resolve_predict_output(case_id, "spatial_sector_directions_")
    except Exception:
        sector_vis_show = PROJECT_ROOT / "predict_output" / f"spatial_sector_directions_{case_id}.png"

    st.subheader("中间结果总览")
    copied_original = sorted(out_dir.glob("00_original__*"), key=lambda p: p.stat().st_mtime)
    if copied_original:
        show_image_with_explain(
            copied_original[-1],
            "原图（已复制到输出目录）",
            "这是上传原图在 web_demo_output 中的副本，便于后续查阅与结果对照。",
        )
    show_image_with_explain(out_dir / "segmentation_overlay.png", "分割叠加图", "显示三分类分割结果叠加在原图上，用于快速检查分割边界是否合理。")
    show_image_with_explain(out_dir / "segmentation_filled.png", "分割填充图", "将各类别区域直接上色填充，便于查看面积分布与类别关系。")
    show_image_with_explain(PROJECT_ROOT / "skin_output" / f"only_texture_line_{case_id}.png", "纯纹理线条", "仅保留纹理线条信号，作为方向分析与局部评分的核心输入。")
    show_image_with_explain(PROJECT_ROOT / "skin_output" / f"texture_line_{case_id}.png", "纹理对比图", "展示原图、纹理线条和叠加效果，帮助判断纹理提取是否过强或过弱。")
    show_image_with_explain(orientation_only_show, "方向图（纯）", "在纹理线条上绘制局部方向与主方向，便于观察纹理走向。")
    show_image_with_explain(orientation_full_show, "方向图（含背景）", "在背景上下文中查看方向信息，更容易定位方向异常区域。")
    show_image_with_explain(out_dir / "final_overlay.png", "最终叠加图", "将主要方向结果叠加回原图，作为整体效果展示图。")
    show_image_with_explain(sector_vis_show, "8扇区分析", "把区域划分为8个扇区，比较各扇区纹理密度与方向一致性。")

    st.subheader("热图与最严重框")
    show_image_with_explain(out_dir / "severity_map.png", "Severity Map", "像素级严重度图，分数由纹理密度与方向一致性加权得到。")
    show_image_with_explain(out_dir / "presence_map.png", "Presence Map", "按你设置的阈值/分位把严重度分档着色，最高档固定为红色。")
    show_image_with_explain(out_dir / "severity_overlay.png", "Severity Overlay", "将 Severity Map 叠加到原图，方便观察高分区与真实组织位置关系。")
    show_image_with_explain(out_dir / "presence_overlay.png", "Presence Overlay", "将 Presence 分级结果叠加到原图，用于直观查看各档分布范围。")

    worst_box_img = sorted(out_dir.glob("*_worst*_box.png"), key=lambda p: p.stat().st_mtime)
    worst_dir_img = sorted(out_dir.glob("*_worst*_direction.png"), key=lambda p: p.stat().st_mtime)
    worst_area_img = sorted(out_dir.glob("*_worst*_area.png"), key=lambda p: p.stat().st_mtime)
    if worst_box_img:
        show_image_with_explain(worst_box_img[-1], "最严重框", "在目标区域内筛选出的高严重度框，支持1~5个不重叠框。")
    if worst_area_img:
        show_image_with_explain(worst_area_img[-1], "Worst Area 联通区域", "显示每个最严重框对应的高分联通区域（A=80分位阈值）以及种子点，后续框与这些区域不重叠。")
    if worst_dir_img:
        show_image_with_explain(worst_dir_img[-1], "框内方向", "在最严重框中心绘制主方向箭头，反映该区域主要纹理方向。")

    info_files = sorted(out_dir.glob("*_worst*_info.txt"))
    if info_files:
        st.markdown("**最严重框参数文本**")
        st.code(info_files[-1].read_text(encoding="utf-8"), language="text")


def app():
    st.set_page_config(page_title="皮肤纹理方向分析 Demo", layout="wide")

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

    st.title("皮肤纹理方向分析 Demo")
    st.caption("上传图片 -> 生成全流程中间结果 -> 调参重算热图/Presence/最严重框")

    if cv2 is None:
        st.error("当前 Python 环境缺少 OpenCV，无法运行图像处理流程。")
        st.code(
            "python3 -m pip install --user opencv-python\n"
            "# 如果提示 libGL.so.1 缺失，可执行：\n"
            "sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0"
        )
        if _CV2_IMPORT_ERROR is not None:
            st.text(f"OpenCV 导入错误: {_CV2_IMPORT_ERROR}")
        st.stop()

    init_params_state_once()

    # Apply per-image params before widgets are instantiated to avoid Streamlit key mutation errors.
    pending_params = st.session_state.get("_pending_params_to_apply")
    if isinstance(pending_params, dict):
        apply_params_to_session(normalize_params_dict(pending_params))
        st.session_state["_pending_params_to_apply"] = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.markdown(f"**设备**: {device}")

    model_rel = st.sidebar.text_input(
        "模型文件",
        key="p_model_rel",
        help="用于分割预测的模型权重文件路径（相对项目根目录）。",
    )
    model_path = (PROJECT_ROOT / model_rel).resolve()
    if not model_path.exists():
        st.sidebar.error("模型文件不存在，请修改路径")

    uploaded = st.file_uploader(
        "选择图片文件",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="上传待分析图片。系统会生成分割、纹理、方向、热图和最严重框等中间结果。",
    )
    batch_uploaded = st.file_uploader(
        "批量选择图片文件",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="batch_uploader",
        help="可一次选择多张图片批量运行。结果会保存到同一批次目录下。",
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
            st.session_state["image_size_text"] = f"{w}×{h} 像素"
        else:
            st.session_state["image_size_text"] = "读取失败"

    current_output_folder = st.session_state.get("output_folder")
    if current_output_folder and st.session_state.get("_params_loaded_for_output_folder") != current_output_folder:
        out_dir = OUTPUT_DIR / current_output_folder
        loaded_params = load_params_from_output_dir(out_dir)
        if loaded_params is None:
            loaded_params = normalize_params_dict(PARAM_DEFAULTS)
        st.session_state["_pending_params_to_apply"] = loaded_params
        st.session_state["_params_loaded_for_output_folder"] = current_output_folder
        st.rerun()

    st.sidebar.header("热图参数")
    target_class = st.sidebar.selectbox(
        "目标类别",
        options=[1, 2],
        key="p_target_class",
        help="选择做局部评分的区域类别。1通常为有效病灶区域，2通常为更核心区域。",
    )
    radius_mode = st.sidebar.selectbox(
        "Radius模式",
        options=["动态", "固定"],
        key="p_radius_mode",
        help="动态：按图像大小自动设半径；固定：使用你指定的像素半径。",
    )
    fixed_radius = None
    if radius_mode == "固定":
        fixed_radius = st.sidebar.number_input(
            "Heatmap Radius",
            min_value=8,
            max_value=120,
            step=1,
            key="p_fixed_radius",
            help="局部统计窗口半径（像素）。越大越平滑，越小越敏感。",
        )

    texture_threshold = st.sidebar.number_input(
        "纹理像素阈值",
        min_value=0.05,
        max_value=0.95,
        step=0.01,
        format="%.2f",
        key="p_texture_threshold",
        help="将纹理图像素判定为纹理点的阈值。阈值越高，识别到的纹理点越少。",
    )
    density_weight = st.sidebar.number_input(
        "密度权重",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        key="p_density_weight",
        help="严重度评分中“纹理密度”项的权重。",
    )
    consistency_weight = st.sidebar.number_input(
        "一致性权重",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        key="p_consistency_weight",
        help="严重度评分中“方向一致性”项的权重。",
    )
    heat_alpha = st.sidebar.number_input(
        "热图叠加透明度",
        min_value=0.1,
        max_value=0.9,
        step=0.05,
        format="%.2f",
        key="p_heat_alpha",
        help="Severity Overlay 中热图颜色叠加到原图的强度。",
    )

    st.sidebar.header("Presence参数")
    n_bins = st.sidebar.number_input(
        "Presence分级数",
        min_value=2,
        max_value=6,
        step=1,
        key="p_n_bins",
        help="Presence Map 分成多少档颜色等级。",
    )
    n_bins = int(n_bins)
    presence_mode = st.sidebar.selectbox(
        "Presence划分模式",
        options=["threshold", "quantile"],
        key="p_presence_mode",
        help="threshold：按固定阈值切分；quantile：按分位数切分。",
    )
    presence_cuts = []
    if presence_mode == "threshold":
        st.sidebar.caption("阈值模式：每个划分线是0~1范围，按归一化分数直接切分")
        for i in range(n_bins - 1):
            v = st.sidebar.number_input(
                f"阈值线{i + 1}",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key=f"p_presence_cut_{i}",
                help="第{i}到第{i+1}档的分界阈值（0~1）。",
            )
            presence_cuts.append(v)
    else:
        st.sidebar.caption("分位模式：每个划分线是0~100分位，按mask内分数分布切分")
        for i in range(n_bins - 1):
            q = st.sidebar.number_input(
                f"分位线{i + 1} (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.1f",
                key=f"p_presence_cut_{i}",
                help="第{i}到第{i+1}档的分界分位点（0~100%）。",
            )
            presence_cuts.append(q)
    presence_cuts = _normalize_presence_cuts([float(v) for v in presence_cuts], n_bins, presence_mode)

    st.sidebar.header("最严重框参数")
    box_mode = st.sidebar.selectbox(
        "Box Size模式",
        options=["动态", "固定"],
        key="p_box_mode",
        help="动态：按图像大小自动设框边长；固定：使用你指定的框大小。",
    )
    box_size = None
    if box_mode == "固定":
        box_size = st.sidebar.number_input(
            "Box Size",
            min_value=24,
            max_value=240,
            step=2,
            key="p_box_size",
            help="最严重框的边长（像素）。",
        )
    num_boxes = st.sidebar.number_input(
        "框数量",
        min_value=1,
        max_value=5,
        step=1,
        key="p_num_boxes",
        help="输出几个不重叠的高严重度区域框（1~5）。",
    )
    min_overlap = st.sidebar.number_input(
        "最小mask覆盖率",
        min_value=0.0,
        max_value=0.95,
        step=0.01,
        format="%.2f",
        key="p_min_overlap",
        help="候选框中有效区域像素占比下限。值越大，框越集中在目标区域内。",
    )
    num_boxes = int(num_boxes)
    if fixed_radius is not None:
        fixed_radius = int(fixed_radius)
    if box_size is not None:
        box_size = int(box_size)

    with st.sidebar.expander("参数解释总览", expanded=False):
        st.markdown(
            "- 目标类别：决定在哪个分割区域上计算严重度。\n"
            "- Radius：局部统计邻域大小，影响热图平滑程度。\n"
            "- 纹理像素阈值：决定哪些像素计入纹理密度。\n"
            "- 密度/一致性权重：共同决定严重度分数。\n"
            "- Presence模式：threshold按固定分数切，quantile按分位切。\n"
            "- Box参数：控制最严重框的大小、数量和有效区域约束。"
        )

    st.sidebar.markdown('<div class="floating-actions">', unsafe_allow_html=True)
    st.sidebar.subheader("运行按钮")
    run_all = st.sidebar.button("1) 生成全流程", help="首次运行建议点击，生成全部中间结果与可视化。")
    rerun_part = st.sidebar.button("2) 仅重算热图/框", help="不重跑分割和方向，仅按当前参数更新热图与最严重框。")
    run_batch = st.sidebar.button("3) 批量处理", help="对批量上传的多张图按当前参数一次性处理。")
    st.sidebar.caption("滚动页面时该按钮区会固定显示。")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["case_id"] is not None:
        st.info(f"当前样本ID: {st.session_state['case_id']}")
        st.caption(f"输入图片尺寸: {st.session_state['image_size_text']}")
        st.caption(f"输出目录: web_demo_output/{st.session_state['output_folder']}")

    if run_all:
        if st.session_state["image_path"] is None:
            st.error("请先上传图片")
            st.stop()
        if not model_path.exists():
            st.error("模型路径无效")
            st.stop()

        st.session_state["batch_records"] = []
        st.session_state["batch_id"] = None

        params_snapshot = build_current_params_snapshot(
            model_rel=model_rel,
            target_class=target_class,
            radius_mode=radius_mode,
            fixed_radius=fixed_radius,
            texture_threshold=texture_threshold,
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
        )
        with st.spinner("正在生成全流程中间结果..."):
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
                texture_threshold=texture_threshold,
                density_weight=density_weight,
                consistency_weight=consistency_weight,
                heat_alpha=heat_alpha,
                presence_mode=presence_mode,
                presence_cuts=presence_cuts,
                box_size=box_size,
                num_boxes=num_boxes,
                min_overlap=min_overlap,
                device=device,
            )
        st.success(f"完成。radius={heat_info['radius']}，box_size={heat_info['box_size']}")
        run_params_path = save_params_to_output_dir(
            out_dir=Path(heat_info["out_dir"]),
            input_image_path=image_path,
            params_snapshot=params_snapshot,
            heat_info=heat_info,
        )
        st.caption(f"参数已保存: {run_params_path.name}")
        if heat_info.get("presence_mode") == "quantile":
            st.caption(
                f"Presence分位线(%): {heat_info['presence_cuts']} -> 实际阈值: "
                f"{[round(v, 4) for v in heat_info['presence_thresholds']]}"
            )
        else:
            st.caption(f"Presence阈值线: {[round(v, 4) for v in heat_info['presence_thresholds']]}")

    if rerun_part:
        if st.session_state["image_path"] is None:
            st.error("请先上传图片并至少跑一次全流程")
            st.stop()
        if not model_path.exists():
            st.error("模型路径无效")
            st.stop()

        st.session_state["batch_records"] = []
        st.session_state["batch_id"] = None

        params_snapshot = build_current_params_snapshot(
            model_rel=model_rel,
            target_class=target_class,
            radius_mode=radius_mode,
            fixed_radius=fixed_radius,
            texture_threshold=texture_threshold,
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
        )
        with st.spinner("正在按新参数重算热图和最严重框..."):
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
                texture_threshold=texture_threshold,
                density_weight=density_weight,
                consistency_weight=consistency_weight,
                heat_alpha=heat_alpha,
                presence_mode=presence_mode,
                presence_cuts=presence_cuts,
                box_size=box_size,
                num_boxes=num_boxes,
                min_overlap=min_overlap,
                device=device,
            )
        st.success(f"重算完成。radius={heat_info['radius']}，box_size={heat_info['box_size']}")
        run_params_path = save_params_to_output_dir(
            out_dir=Path(heat_info["out_dir"]),
            input_image_path=image_path,
            params_snapshot=params_snapshot,
            heat_info=heat_info,
        )
        st.caption(f"参数已保存: {run_params_path.name}")
        if heat_info.get("presence_mode") == "quantile":
            st.caption(
                f"Presence分位线(%): {heat_info['presence_cuts']} -> 实际阈值: "
                f"{[round(v, 4) for v in heat_info['presence_thresholds']]}"
            )
        else:
            st.caption(f"Presence阈值线: {[round(v, 4) for v in heat_info['presence_thresholds']]}")

    if run_batch:
        if not batch_uploaded:
            st.error("请先在“批量选择图片文件”里选择至少一张图片")
            st.stop()
        if not model_path.exists():
            st.error("模型路径无效")
            st.stop()

        params_snapshot = build_current_params_snapshot(
            model_rel=model_rel,
            target_class=target_class,
            radius_mode=radius_mode,
            fixed_radius=fixed_radius,
            texture_threshold=texture_threshold,
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
        )

        batch_id = new_batch_id()
        batch_root = OUTPUT_DIR / batch_id
        batch_items = []
        failed = []

        with st.spinner(f"正在批量处理 {len(batch_uploaded)} 张图片..."):
            for idx, one in enumerate(batch_uploaded, start=1):
                try:
                    image_path, case_id, original_name, output_folder = save_uploaded_file(one)
                    output_folder = f"{batch_id}/{output_folder}"

                    original_for_size = imread_unicode(image_path, cv2.IMREAD_COLOR)
                    if original_for_size is not None:
                        h, w = original_for_size.shape[:2]
                        image_size_text = f"{w}×{h} 像素"
                    else:
                        image_size_text = "读取失败"

                    run_full_pipeline(image_path, case_id, output_folder, original_name, model_path, device)
                    heat_info = run_heatmap_and_worst(
                        image_path=image_path,
                        case_id=case_id,
                        output_folder=output_folder,
                        original_name=original_name,
                        model_path=model_path,
                        target_class=target_class,
                        fixed_radius=fixed_radius,
                        texture_threshold=texture_threshold,
                        density_weight=density_weight,
                        consistency_weight=consistency_weight,
                        heat_alpha=heat_alpha,
                        presence_mode=presence_mode,
                        presence_cuts=presence_cuts,
                        box_size=box_size,
                        num_boxes=num_boxes,
                        min_overlap=min_overlap,
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
            st.success(f"批量完成：成功 {len(batch_items)} 张，失败 {len(failed)} 张")
            st.caption(f"批次目录: web_demo_output/{batch_id}")
        else:
            st.error("批量处理失败，未生成可用结果。")
        if failed:
            st.warning("部分图片处理失败：")
            st.code("\n".join(failed), language="text")

    batch_records = st.session_state.get("batch_records", [])
    if batch_records:
        st.subheader("批量结果浏览")
        st.caption(f"当前批次目录: web_demo_output/{st.session_state.get('batch_id')}")
        options = [f"{x['index']:02d}. {x['original_name']}" for x in batch_records]
        selected = st.selectbox("点击选择图片查看结果", options=options, key="batch_viewer_select")
        sel_idx = options.index(selected)
        sel_item = batch_records[sel_idx]
        render_case_results(
            case_id=sel_item["case_id"],
            output_folder=sel_item["output_folder"],
            image_size_text=sel_item.get("image_size_text"),
        )

    if st.session_state["case_id"] is not None and not st.session_state.get("batch_records"):
        case_id = st.session_state["case_id"]
        output_folder = st.session_state.get("output_folder") or case_id
        render_case_results(case_id, output_folder, st.session_state.get("image_size_text"))


if __name__ == "__main__":
    app()
