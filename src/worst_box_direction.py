from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .local_score_heatmap import (
        build_disk_kernel,
        dynamic_radius_from_size,
        build_effective_texture_region_mask,
        compute_orientations,
        predict_mask,
    )
except ImportError:
    from src.local_score_heatmap import (
        build_disk_kernel,
        dynamic_radius_from_size,
        build_effective_texture_region_mask,
        compute_orientations,
        predict_mask,
    )
from skin import analyze_skin_texture


def scaled_box_size_from_shape(
    height: int,
    width: int,
    base_box_size: int = 80,
    ref_min_dim: int = 920,
    min_box_size: int = 32,
    max_box_size: int = 240,
) -> int:
    """Scale worst-box size by image short side. Reference: min_dim=920 -> box=80."""
    min_dim = min(height, width)
    if min_dim <= 0:
        return base_box_size

    scaled = int(round(base_box_size * (min_dim / float(ref_min_dim))))
    scaled = max(min_box_size, min(max_box_size, scaled))
    scaled = min(scaled, min_dim)
    # Keep an even size to preserve symmetric center math.
    if scaled % 2 != 0:
        scaled = max(min_box_size, scaled - 1)
    return scaled


def compute_score_map(image_path: Path, model_path: Path, target_class: int, radius: int | None, device: str):
    pred = predict_mask(str(image_path), str(model_path), device)
    raw_region_mask = (pred == target_class).astype(np.uint8)

    if target_class == 1:
        region_mask = build_effective_texture_region_mask(raw_region_mask)
    else:
        region_mask = raw_region_mask

    if np.sum(region_mask) == 0:
        raise RuntimeError(f"Target class region is empty: class={target_class}")

    analyze_skin_texture(str(image_path), model_path=str(model_path), device=device)

    texture_path = image_path.parent.parent.parent / "skin_output" / f"only_texture_line_{image_path.stem}.png"
    tex = cv2.imread(str(texture_path), cv2.IMREAD_GRAYSCALE)
    if tex is None:
        raise FileNotFoundError(f"Texture image not found: {texture_path}")

    if tex.shape != pred.shape:
        tex = cv2.resize(tex, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    h, w = tex.shape
    if radius is None:
        radius = dynamic_radius_from_size(h, w)

    kernel = build_disk_kernel(radius)
    tex_norm = tex.astype(np.float32) / 255.0
    texture_binary = ((tex_norm > 0.4) & (region_mask > 0)).astype(np.float32)
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

    score = 0.7 * density + 0.3 * consistency
    score[region_mask == 0] = 0.0

    return score, region_mask, orientations, pred, radius


def find_worst_box(
    score: np.ndarray,
    mask: np.ndarray,
    box_size: int = 80,
    min_overlap_ratio: float = 0.30,
    overlap_power: float = 0.35,
    forbidden_boxes: list[dict] | None = None,
):
    h, w = score.shape
    if h < box_size or w < box_size:
        raise ValueError("Image is smaller than box size.")

    k = np.ones((box_size, box_size), dtype=np.float32)

    # Sum score in each candidate box center.
    score_sum = cv2.filter2D(score.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)

    # Overlap statistics in each candidate box center.
    overlap_count = cv2.filter2D(mask.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    overlap_ratio = overlap_count / float(box_size * box_size)

    # Use masked mean severity (only pixels inside mask are counted).
    eps = 1e-6
    mean_severity = np.zeros_like(score_sum, dtype=np.float32)
    valid_overlap = overlap_count > eps
    mean_severity[valid_overlap] = score_sum[valid_overlap] / overlap_count[valid_overlap]

    # New rule: objective is only the masked mean severity.
    # Pixels outside mask are not counted in statistics.
    objective = mean_severity.copy()

    half = box_size // 2
    valid = np.zeros_like(score, dtype=bool)
    valid[half : h - (box_size - half) + 1, half : w - (box_size - half) + 1] = True

    if not np.any(valid):
        raise RuntimeError("No valid center for the selected box size.")

    # New rule: box center must be inside mask1.
    center_in_mask = mask > 0
    valid = valid & center_in_mask

    if not np.any(valid):
        raise RuntimeError("No valid center inside mask for the selected box size.")

    # Apply minimum-overlap hard constraint first; fallback to valid if no candidate survives.
    constrained = valid & (overlap_ratio >= min_overlap_ratio)
    if not np.any(constrained):
        constrained = valid

    ys, xs = np.where(constrained)
    if ys.size == 0:
        raise RuntimeError("No constrained candidates available.")

    obj_vals = objective[ys, xs]
    mean_vals = mean_severity[ys, xs]
    overlap_vals = overlap_ratio[ys, xs]

    # Lexicographic ranking: objective desc, mean_severity desc, overlap desc
    order = np.lexsort((-overlap_vals, -mean_vals, -obj_vals))

    def boxes_overlap(a: dict, b: dict) -> bool:
        return not (a["x2"] <= b["x1"] or a["x1"] >= b["x2"] or a["y2"] <= b["y1"] or a["y1"] >= b["y2"])

    if forbidden_boxes is None:
        forbidden_boxes = []

    yc, xc = None, None
    chosen = None
    for i in order:
        yy = int(ys[i])
        xx = int(xs[i])

        x1_t = xx - half
        y1_t = yy - half
        x2_t = x1_t + box_size
        y2_t = y1_t + box_size
        candidate_box = {"x1": x1_t, "y1": y1_t, "x2": x2_t, "y2": y2_t}

        if forbidden_boxes and any(boxes_overlap(candidate_box, fb) for fb in forbidden_boxes):
            continue

        yc, xc = yy, xx
        chosen = candidate_box
        break

    if yc is None or xc is None:
        raise RuntimeError("No non-overlapping candidate box found.")

    x1 = chosen["x1"]
    y1 = chosen["y1"]
    x2 = chosen["x2"]
    y2 = chosen["y2"]

    return {
        "xc": xc,
        "yc": yc,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "score_sum": float(score_sum[yc, xc]),
        "mean_severity": float(mean_severity[yc, xc]),
        "objective": float(objective[yc, xc]),
        "overlap_count": float(overlap_count[yc, xc]),
        "overlap_ratio": float(overlap_ratio[yc, xc]),
    }


def mean_orientation_degrees(orientations: np.ndarray, mask: np.ndarray, box):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    patch_o = orientations[y1:y2, x1:x2]
    patch_m = mask[y1:y2, x1:x2] > 0

    vals = patch_o[patch_m]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None

    doubled = 2.0 * vals
    mc = float(np.mean(np.cos(doubled)))
    ms = float(np.mean(np.sin(doubled)))
    mean_doubled = float(np.arctan2(ms, mc))
    mean_angle = (mean_doubled / 2.0) % np.pi
    deg = np.degrees(mean_angle)
    return float(deg)


def draw_outputs(
    image_path: Path,
    out_dir: Path,
    box_size: int,
    boxes: list[dict],
    mean_degs: list[float | None],
    pred: np.ndarray,
):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Draw both mask boundaries on top of the original image.
    # class 1 (affected area): green, class 2 (keloid body): cyan.
    class1 = (pred == 1).astype(np.uint8)
    class2 = (pred == 2).astype(np.uint8)

    def draw_boundary(canvas: np.ndarray, binary_mask: np.ndarray, color_bgr: tuple[int, int, int]):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color_bgr, 2)

    if len(boxes) == 0:
        raise RuntimeError("No boxes to draw.")

    vis_box = img.copy()
    draw_boundary(vis_box, class1, (0, 255, 0))
    draw_boundary(vis_box, class2, (255, 255, 0))
    box_colors = [(0, 0, 255), (255, 0, 255), (0, 165, 255)]
    for i, box in enumerate(boxes):
        color = box_colors[i % len(box_colors)]
        cv2.rectangle(vis_box, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
        label = f"Box{i+1}, overlap={box['overlap_ratio']:.3f}, mean={box['mean_severity']:.3f}"
        y_text = max(20 + i * 20, box["y1"] - 8)
        cv2.putText(
            vis_box,
            label,
            (max(5, box["x1"]), y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    vis_dir = vis_box.copy()
    for i, box in enumerate(boxes):
        mean_deg = mean_degs[i] if i < len(mean_degs) else None
        if mean_deg is None:
            continue
        color = box_colors[i % len(box_colors)]
        theta = np.radians(mean_deg)
        length = 35
        cx, cy = box["xc"], box["yc"]
        dx = int(round(np.cos(theta) * length))
        dy = int(round(np.sin(theta) * length))
        cv2.arrowedLine(vis_dir, (cx, cy), (cx + dx, cy + dy), color, 3, tipLength=0.2)
        y_text = min(vis_dir.shape[0] - 10 - i * 25, box["y2"] + 25)
        cv2.putText(
            vis_dir,
            f"Box{i+1} mean direction(masked): {mean_deg:.1f} deg",
            (max(5, box["x1"]), max(20, y_text)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{image_path.stem}_worst{box_size}_box.png"), vis_box)
    cv2.imwrite(str(out_dir / f"{image_path.stem}_worst{box_size}_direction.png"), vis_dir)

    with (out_dir / f"{image_path.stem}_worst{box_size}_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"image={image_path.stem}\n")
        f.write(f"box_size={box_size}\n")
        f.write(f"num_boxes={len(boxes)}\n")
        for i, box in enumerate(boxes, start=1):
            f.write(f"box{i}_x1={box['x1']}\n")
            f.write(f"box{i}_y1={box['y1']}\n")
            f.write(f"box{i}_x2={box['x2']}\n")
            f.write(f"box{i}_y2={box['y2']}\n")
            f.write(f"box{i}_xc={box['xc']}\n")
            f.write(f"box{i}_yc={box['yc']}\n")
            f.write(f"box{i}_score_sum={box['score_sum']:.6f}\n")
            f.write(f"box{i}_mean_severity={box['mean_severity']:.6f}\n")
            f.write(f"box{i}_objective={box['objective']:.6f}\n")
            f.write(f"box{i}_overlap_count={box['overlap_count']:.1f}\n")
            f.write(f"box{i}_overlap_ratio={box['overlap_ratio']:.6f}\n")
            mean_deg = mean_degs[i - 1] if i - 1 < len(mean_degs) else None
            if mean_deg is None:
                f.write(f"box{i}_mean_direction_deg=nan\n")
            else:
                f.write(f"box{i}_mean_direction_deg={mean_deg:.6f}\n")


def main():
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Find scaled worst box and masked mean direction")
    parser.add_argument("num", help="Image id, e.g. 66")
    parser.add_argument("--radius", type=int, default=None, help="Local score radius; default scales by image size")
    parser.add_argument("--target-class", type=int, default=1, choices=[1, 2], help="Target mask class")
    parser.add_argument("--box-size", type=int, default=None, help="Worst box size; default scales by image size")
    parser.add_argument("--num-boxes", type=int, default=1, choices=[1, 2, 3], help="Number of non-overlapping boxes")
    parser.add_argument("--min-overlap", type=float, default=0.30, help="Minimum mask overlap ratio for candidate boxes")
    parser.add_argument("--overlap-power", type=float, default=0.35, help="Soft overlap weighting exponent in objective")
    parser.add_argument("--model", default="best_trans_unet_model_20250614_122913.pth", help="Model checkpoint")
    parser.add_argument("--output-subdir", default="r40", help="Output folder under heatmap_output/<num>/")
    args = parser.parse_args()

    root = PROJECT_ROOT
    image_path = root / "dataset" / "final_labeled" / f"{args.num}.jpg"
    model_path = root / args.model
    out_dir = root / "heatmap_output" / str(args.num) / args.output_subdir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    score, mask, orientations, pred, used_radius = compute_score_map(
        image_path=image_path,
        model_path=model_path,
        target_class=args.target_class,
        radius=args.radius,
        device=device,
    )

    if args.box_size is None:
        used_box_size = scaled_box_size_from_shape(mask.shape[0], mask.shape[1])
    else:
        used_box_size = int(args.box_size)

    boxes = []
    means = []
    forbidden = []
    for _ in range(args.num_boxes):
        box = find_worst_box(
            score,
            mask,
            box_size=used_box_size,
            min_overlap_ratio=args.min_overlap,
            overlap_power=args.overlap_power,
            forbidden_boxes=forbidden,
        )
        boxes.append(box)
        means.append(mean_orientation_degrees(orientations, mask, box))
        forbidden.append(box)

    draw_outputs(image_path, out_dir, used_box_size, boxes, means, pred)

    print("Done.")
    print(f"output_dir={out_dir}")
    print(f"radius_used={used_radius}")
    print(f"box_size_used={used_box_size}")
    print(f"num_boxes={len(boxes)}")
    for i, box in enumerate(boxes, start=1):
        print(f"box{i}=({box['x1']},{box['y1']})-({box['x2']},{box['y2']})")
        print(f"box{i}_mean_severity={box['mean_severity']:.4f}")
        print(f"box{i}_objective={box['objective']:.4f}")
        print(f"box{i}_overlap_ratio={box['overlap_ratio']:.4f}")
        mean_deg = means[i - 1]
        if mean_deg is None:
            print(f"box{i}_mean_direction_deg=nan")
        else:
            print(f"box{i}_mean_direction_deg={mean_deg:.2f}")


if __name__ == "__main__":
    main()
