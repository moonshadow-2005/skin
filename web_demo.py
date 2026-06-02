from __future__ import annotations

import hashlib
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
    find_worst_box,
    mean_orientation_degrees,
    scaled_box_size_from_shape,
)


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "web_demo_inputs"
OUTPUT_DIR = PROJECT_ROOT / "web_demo_output"


def imread_unicode(image_path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def save_uploaded_file(uploaded) -> tuple[Path, str]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    content = uploaded.getvalue()
    digest = hashlib.md5(content).hexdigest()[:10]
    ext = Path(uploaded.name).suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        ext = ".jpg"
    case_id = f"web_{digest}"
    image_path = INPUT_DIR / f"{case_id}{ext}"
    image_path.write_bytes(content)
    return image_path, case_id


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


def make_segmentation_visuals(original_bgr: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = original_bgr.shape[:2]
    if pred.shape[:2] != (h, w):
        pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    filled = np.zeros_like(rgb, dtype=np.uint8)
    # class 0/1/2 colors in RGB
    filled[pred == 0] = [70, 130, 255]
    filled[pred == 1] = [80, 220, 120]
    filled[pred == 2] = [255, 110, 90]

    boundary = np.zeros_like(rgb, dtype=np.uint8)
    for class_id, color in [(2, (255, 0, 0)), (1, (0, 255, 0))]:
        binary = (pred == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundary, contours, -1, color, 2)

    alpha_map = np.zeros((h, w), dtype=np.float32)
    alpha_map[pred == 0] = 0.12
    alpha_map[pred == 1] = 0.28
    alpha_map[pred == 2] = 0.40
    overlay = rgb.astype(np.float32).copy()
    for c in range(3):
        overlay[:, :, c] = (1.0 - alpha_map) * overlay[:, :, c] + alpha_map * filled[:, :, c]
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


def run_heatmap_and_worst(
    image_path: Path,
    case_id: str,
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
    out_dir = OUTPUT_DIR / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

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

    cv2.imwrite(str(out_dir / "severity_map.png"), heatmap_bgr)
    cv2.imwrite(str(out_dir / "severity_overlay.png"), severity_overlay)
    cv2.imwrite(str(out_dir / "presence_map.png"), presence_map)
    cv2.imwrite(str(out_dir / "presence_overlay.png"), presence_overlay)

    if box_size is None:
        used_box_size = scaled_box_size_from_shape(result["region_mask"].shape[0], result["region_mask"].shape[1])
    else:
        used_box_size = int(box_size)

    boxes = []
    means = []
    forbidden = []
    for _ in range(num_boxes):
        box = find_worst_box(
            result["score_norm"],
            result["region_mask"],
            box_size=used_box_size,
            min_overlap_ratio=min_overlap,
            forbidden_boxes=forbidden,
        )
        boxes.append(box)
        means.append(mean_orientation_degrees(result["orientations"], result["region_mask"], box))
        forbidden.append(box)

    # Remove stale worst-box artifacts so UI always reflects the current run.
    for stale in out_dir.glob(f"{image_path.stem}_worst*_box.png"):
        stale.unlink(missing_ok=True)
    for stale in out_dir.glob(f"{image_path.stem}_worst*_direction.png"):
        stale.unlink(missing_ok=True)
    for stale in out_dir.glob(f"{image_path.stem}_worst*_info.txt"):
        stale.unlink(missing_ok=True)

    draw_outputs(
        image_path=image_path,
        out_dir=out_dir,
        box_size=used_box_size,
        boxes=boxes,
        mean_degs=means,
        pred=result["pred"],
    )

    return {
        "radius": result["radius"],
        "box_size": used_box_size,
        "out_dir": out_dir,
        "presence_mode": presence_mode,
        "presence_cuts": used_cuts,
        "presence_thresholds": presence_thresholds,
    }


def run_full_pipeline(image_path: Path, case_id: str, model_path: Path, device: str) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUT_DIR / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = predict_mask(str(image_path), str(model_path), device)
    original = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"无法读取原图: {image_path}")
    filled, boundary, seg_overlay = make_segmentation_visuals(original, pred)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.markdown(f"**设备**: {device}")

    model_rel = st.sidebar.text_input(
        "模型文件",
        value="best_trans_unet_model_20250614_122913.pth",
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

    st.sidebar.header("热图参数")
    target_class = st.sidebar.selectbox(
        "目标类别",
        options=[1, 2],
        index=0,
        help="选择做局部评分的区域类别。1通常为有效病灶区域，2通常为更核心区域。",
    )
    radius_mode = st.sidebar.selectbox(
        "Radius模式",
        options=["动态", "固定"],
        index=1,
        help="动态：按图像大小自动设半径；固定：使用你指定的像素半径。",
    )
    fixed_radius = None
    if radius_mode == "固定":
        fixed_radius = st.sidebar.slider(
            "Heatmap Radius",
            min_value=8,
            max_value=120,
            value=40,
            step=1,
            help="局部统计窗口半径（像素）。越大越平滑，越小越敏感。",
        )

    texture_threshold = st.sidebar.slider(
        "纹理像素阈值",
        min_value=0.05,
        max_value=0.95,
        value=0.40,
        step=0.01,
        help="将纹理图像素判定为纹理点的阈值。阈值越高，识别到的纹理点越少。",
    )
    density_weight = st.sidebar.slider(
        "密度权重",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help="严重度评分中“纹理密度”项的权重。",
    )
    consistency_weight = st.sidebar.slider(
        "一致性权重",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="严重度评分中“方向一致性”项的权重。",
    )
    heat_alpha = st.sidebar.slider(
        "热图叠加透明度",
        min_value=0.1,
        max_value=0.9,
        value=0.55,
        step=0.05,
        help="Severity Overlay 中热图颜色叠加到原图的强度。",
    )

    st.sidebar.header("Presence参数")
    n_bins = st.sidebar.slider(
        "Presence分级数",
        min_value=2,
        max_value=6,
        value=4,
        step=1,
        help="Presence Map 分成多少档颜色等级。",
    )
    presence_mode = st.sidebar.selectbox(
        "Presence划分模式",
        options=["threshold", "quantile"],
        index=0,
        help="threshold：按固定阈值切分；quantile：按分位数切分。",
    )
    presence_cuts = []
    if presence_mode == "threshold":
        st.sidebar.caption("阈值模式：每个划分线是0~1范围，按归一化分数直接切分")
        for i in range(n_bins - 1):
            default_v = float((i + 1) / n_bins)
            v = st.sidebar.slider(
                f"阈值线{i + 1}",
                min_value=0.0,
                max_value=1.0,
                value=default_v,
                step=0.01,
                key=f"presence_threshold_{i}",
                help="第{i}到第{i+1}档的分界阈值（0~1）。",
            )
            presence_cuts.append(v)
    else:
        st.sidebar.caption("分位模式：每个划分线是0~100分位，按mask内分数分布切分")
        for i in range(n_bins - 1):
            default_q = float((i + 1) * 100.0 / n_bins)
            q = st.sidebar.slider(
                f"分位线{i + 1} (%)",
                min_value=0.0,
                max_value=100.0,
                value=default_q,
                step=1.0,
                key=f"presence_quantile_{i}",
                help="第{i}到第{i+1}档的分界分位点（0~100%）。",
            )
            presence_cuts.append(q)
    presence_cuts = sorted(presence_cuts)

    st.sidebar.header("最严重框参数")
    box_mode = st.sidebar.selectbox(
        "Box Size模式",
        options=["动态", "固定"],
        index=1,
        help="动态：按图像大小自动设框边长；固定：使用你指定的框大小。",
    )
    box_size = None
    if box_mode == "固定":
        box_size = st.sidebar.slider(
            "Box Size",
            min_value=24,
            max_value=240,
            value=80,
            step=2,
            help="最严重框的边长（像素）。",
        )
    num_boxes = st.sidebar.slider(
        "框数量",
        min_value=1,
        max_value=3,
        value=1,
        step=1,
        help="输出几个不重叠的高严重度区域框。",
    )
    min_overlap = st.sidebar.slider(
        "最小mask覆盖率",
        min_value=0.0,
        max_value=0.95,
        value=0.30,
        step=0.01,
        help="候选框中有效区域像素占比下限。值越大，框越集中在目标区域内。",
    )

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
    st.sidebar.caption("滚动页面时该按钮区会固定显示。")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    if "image_path" not in st.session_state:
        st.session_state["image_path"] = None
    if "case_id" not in st.session_state:
        st.session_state["case_id"] = None

    if uploaded is not None:
        image_path, case_id = save_uploaded_file(uploaded)
        st.session_state["image_path"] = str(image_path)
        st.session_state["case_id"] = case_id
        st.info(f"当前样本ID: {case_id}")

    if run_all:
        if st.session_state["image_path"] is None:
            st.error("请先上传图片")
            st.stop()
        if not model_path.exists():
            st.error("模型路径无效")
            st.stop()

        with st.spinner("正在生成全流程中间结果..."):
            image_path = Path(st.session_state["image_path"])
            case_id = st.session_state["case_id"]
            full_info = run_full_pipeline(image_path, case_id, model_path, device)
            heat_info = run_heatmap_and_worst(
                image_path=image_path,
                case_id=case_id,
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

        with st.spinner("正在按新参数重算热图和最严重框..."):
            image_path = Path(st.session_state["image_path"])
            case_id = st.session_state["case_id"]
            heat_info = run_heatmap_and_worst(
                image_path=image_path,
                case_id=case_id,
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
        if heat_info.get("presence_mode") == "quantile":
            st.caption(
                f"Presence分位线(%): {heat_info['presence_cuts']} -> 实际阈值: "
                f"{[round(v, 4) for v in heat_info['presence_thresholds']]}"
            )
        else:
            st.caption(f"Presence阈值线: {[round(v, 4) for v in heat_info['presence_thresholds']]}")

    if st.session_state["case_id"] is not None:
        case_id = st.session_state["case_id"]
        out_dir = OUTPUT_DIR / case_id
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
        show_image_with_explain(
            out_dir / "segmentation_overlay.png",
            "分割叠加图",
            "显示三分类分割结果叠加在原图上，用于快速检查分割边界是否合理。",
        )
        show_image_with_explain(
            out_dir / "segmentation_filled.png",
            "分割填充图",
            "将各类别区域直接上色填充，便于查看面积分布与类别关系。",
        )
        show_image_with_explain(
            PROJECT_ROOT / "skin_output" / f"only_texture_line_{case_id}.png",
            "纯纹理线条",
            "仅保留纹理线条信号，作为方向分析与局部评分的核心输入。",
        )
        show_image_with_explain(
            PROJECT_ROOT / "skin_output" / f"texture_line_{case_id}.png",
            "纹理对比图",
            "展示原图、纹理线条和叠加效果，帮助判断纹理提取是否过强或过弱。",
        )
        show_image_with_explain(
            orientation_only_show,
            "方向图（纯）",
            "在纹理线条上绘制局部方向与主方向，便于观察纹理走向。",
        )
        show_image_with_explain(
            orientation_full_show,
            "方向图（含背景）",
            "在背景上下文中查看方向信息，更容易定位方向异常区域。",
        )
        show_image_with_explain(
            out_dir / "final_overlay.png",
            "最终叠加图",
            "将主要方向结果叠加回原图，作为整体效果展示图。",
        )
        show_image_with_explain(
            sector_vis_show,
            "8扇区分析",
            "把区域划分为8个扇区，比较各扇区纹理密度与方向一致性。",
        )

        st.subheader("热图与最严重框")
        show_image_with_explain(
            out_dir / "severity_map.png",
            "Severity Map",
            "像素级严重度图，分数由纹理密度与方向一致性加权得到。",
        )
        show_image_with_explain(
            out_dir / "presence_map.png",
            "Presence Map",
            "按你设置的阈值/分位把严重度分档着色，最高档固定为红色。",
        )
        show_image_with_explain(
            out_dir / "severity_overlay.png",
            "Severity Overlay",
            "将 Severity Map 叠加到原图，方便观察高分区与真实组织位置关系。",
        )
        show_image_with_explain(
            out_dir / "presence_overlay.png",
            "Presence Overlay",
            "将 Presence 分级结果叠加到原图，用于直观查看各档分布范围。",
        )

        worst_box_img = sorted(out_dir.glob("*_worst*_box.png"), key=lambda p: p.stat().st_mtime)
        worst_dir_img = sorted(out_dir.glob("*_worst*_direction.png"), key=lambda p: p.stat().st_mtime)
        if worst_box_img:
            show_image_with_explain(
                worst_box_img[-1],
                "最严重框",
                "在目标区域内筛选出的高严重度框，支持1~3个不重叠框。",
            )
        if worst_dir_img:
            show_image_with_explain(
                worst_dir_img[-1],
                "框内方向",
                "在最严重框中心绘制主方向箭头，反映该区域主要纹理方向。",
            )

        info_files = sorted(out_dir.glob("*_worst*_info.txt"))
        if info_files:
            st.markdown("**最严重框参数文本**")
            st.code(info_files[-1].read_text(encoding="utf-8"), language="text")


if __name__ == "__main__":
    app()
