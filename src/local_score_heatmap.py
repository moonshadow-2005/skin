import os
import sys
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Unet import UNet
from skin import analyze_skin_texture


def imread_unicode(image_path: str, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def dynamic_radius_from_size(
    height: int,
    width: int,
    base_radius: int = 80,
    ref_min_dim: int = 920,
    min_radius: int = 12,
    max_radius: int = 96,
) -> int:
    """Scale radius by image short side. Reference: min_dim=920 -> radius=80."""
    min_dim = min(height, width)
    if min_dim <= 0:
        return base_radius

    scaled = int(round(base_radius * (min_dim / float(ref_min_dim))))
    scaled = max(min_radius, min(max_radius, scaled))
    return scaled


def build_disk_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = (x * x + y * y) <= radius * radius
    return disk.astype(np.float32)


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected foreground component in a binary mask."""
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    binary = (mask > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def predict_mask(image_path: str, model_path: str, device: str) -> np.ndarray:
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Use unicode-safe reading for Windows non-ASCII paths.
    image_bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    x = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    x = transforms.ToTensor()(x).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).cpu().numpy()[0].astype(np.uint8)

    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred


def build_effective_texture_region_mask(raw_region_mask: np.ndarray) -> np.ndarray:
    """Replicate skin.py mask pipeline: close -> dilate -> erode."""
    if raw_region_mask.ndim != 2:
        raise ValueError("raw_region_mask must be a 2D array")

    mask = (raw_region_mask > 0).astype(np.uint8)

    # 1) Morphological close with fixed 5x5 ellipse.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # 2) Dynamic dilation based on image height.
    h = mask.shape[0]
    expand_pixels = h // 30
    expand_ksize = 2 * expand_pixels + 1
    expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_ksize, expand_ksize))
    expanded = cv2.dilate(cleaned, expand_kernel, iterations=1)

    # 3) Dynamic erosion for refined final area.
    shrink_pixels = max(1, h // 80)
    shrink_ksize = 2 * shrink_pixels + 1
    shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_ksize, shrink_ksize))
    refined = cv2.erode(expanded, shrink_kernel, iterations=1)

    # Enforce connectivity requirement: keep only the largest connected component.
    return keep_largest_connected_component(refined)


def compute_orientations(gray_img: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(gray_img, dtype=np.float32)
    dy = np.zeros_like(gray_img, dtype=np.float32)

    scharr_x = cv2.Scharr(gray_img, cv2.CV_32F, 1, 0)
    scharr_y = cv2.Scharr(gray_img, cv2.CV_32F, 0, 1)
    dx[valid_mask > 0] = scharr_x[valid_mask > 0]
    dy[valid_mask > 0] = scharr_y[valid_mask > 0]

    j11 = dx * dx
    j22 = dy * dy
    j12 = dx * dy

    sigma = max(3, int(min(gray_img.shape[:2]) / 100))
    ksize = (6 * sigma + 1, 6 * sigma + 1)
    j11 = cv2.GaussianBlur(j11, ksize, sigma)
    j22 = cv2.GaussianBlur(j22, ksize, sigma)
    j12 = cv2.GaussianBlur(j12, ksize, sigma)

    orientations = np.full(gray_img.shape, np.nan, dtype=np.float32)
    ys, xs = np.where(valid_mask > 0)
    for y, x in zip(ys, xs):
        st = np.array([[j11[y, x], j12[y, x]], [j12[y, x], j22[y, x]]], dtype=np.float32)
        eigvals, eigvecs = np.linalg.eigh(st)
        main_dir = eigvecs[:, np.argmax(eigvals)]
        if main_dir[0] < 0:
            main_dir = -main_dir
        orientations[y, x] = np.arctan2(main_dir[1], main_dir[0]) + np.pi / 2
    return orientations


def normalize_on_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    valid = mask > 0
    if np.any(valid):
        v = values[valid]
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if vmax > vmin:
            out[valid] = (v - vmin) / (vmax - vmin)
        else:
            out[valid] = 0.0
    return out


def make_heatmap_image(score_norm: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    color = cm.get_cmap("jet")(score_norm)[:, :, :3]
    color = (color * 255).astype(np.uint8)
    color[region_mask == 0] = 0
    return color


def overlay_heatmap(original_bgr: np.ndarray, heatmap_bgr: np.ndarray, region_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = original_bgr.copy().astype(np.float32)
    heat = heatmap_bgr.astype(np.float32)
    m = region_mask > 0
    for c in range(3):
        overlay[:, :, c][m] = (1.0 - alpha) * overlay[:, :, c][m] + alpha * heat[:, :, c][m]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def add_vertical_colorbar_bgr(
    image_bgr: np.ndarray,
    cmap_name: str = "jet",
    bar_width: int = 28,
    pad: int = 10,
    margin: int = 12,
) -> np.ndarray:
    """Append a vertical 0-1 colorbar to the right side of image (BGR)."""
    h, w = image_bgr.shape[:2]
    canvas_w = w + pad + bar_width + 56
    canvas = np.full((h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[:, :w] = image_bgr

    y = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    cmap_rgb = cm.get_cmap(cmap_name)(y)[:, :, :3]
    bar_rgb = (cmap_rgb * 255).astype(np.uint8)
    bar_bgr = cv2.cvtColor(bar_rgb, cv2.COLOR_RGB2BGR)
    bar = np.repeat(bar_bgr, bar_width, axis=1)

    x0 = w + pad
    x1 = x0 + bar_width
    canvas[:, x0:x1] = bar
    cv2.rectangle(canvas, (x0, 0), (x1 - 1, h - 1), (0, 0, 0), 1)

    # Ticks and labels: 1.0, 0.75, 0.50, 0.25, 0.0
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


def draw_mask_boundary(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    out = image_bgr.copy()
    binary = (mask > 0).astype(np.uint8)
    # Use RETR_LIST so both outer and inner contours (holes) are drawn.
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color_bgr, thickness)
    return out


def build_presence_map(
    score_norm: np.ndarray,
    region_mask: np.ndarray,
    q1_color_bgr: tuple[int, int, int] = (255, 0, 0),
    q2_color_bgr: tuple[int, int, int] = (255, 255, 0),
    q3_color_bgr: tuple[int, int, int] = (0, 255, 255),
    q4_color_bgr: tuple[int, int, int] = (0, 0, 255),
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Quartile presence map inside effective mask: low->high uses blue->cyan->yellow->red."""
    h, w = score_norm.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    inside = region_mask > 0
    if not np.any(inside):
        return out, inside, (0.0, 0.0, 0.0)

    vals = score_norm[inside]
    q25, q50, q75 = np.quantile(vals, [0.25, 0.50, 0.75])
    q25 = float(q25)
    q50 = float(q50)
    q75 = float(q75)

    q1 = inside & (score_norm <= q25)
    q2 = inside & (score_norm > q25) & (score_norm <= q50)
    q3 = inside & (score_norm > q50) & (score_norm <= q75)
    q4 = inside & (score_norm > q75)

    out[q1] = q1_color_bgr
    out[q2] = q2_color_bgr
    out[q3] = q3_color_bgr
    out[q4] = q4_color_bgr
    return out, inside, (q25, q50, q75)


def run_for_id(
    num: str,
    model_path: str = "best_trans_unet_model_20250614_122913.pth",
    target_class: int = 1,
    fixed_radius: int | None = None,
    output_subdir: str | None = None,
) -> None:
    root = PROJECT_ROOT
    image_path = root / "dataset" / "final_labeled" / f"{num}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir = root / "heatmap_output" / str(num)
    if output_subdir:
        out_dir = out_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Image: {image_path}")

    pred = predict_mask(str(image_path), str(root / model_path), device)
    raw_region_mask = (pred == target_class).astype(np.uint8)

    # For class-1, use the same post-processed effective area as skin.py.
    if target_class == 1:
        region_mask = build_effective_texture_region_mask(raw_region_mask)
    else:
        region_mask = raw_region_mask

    if np.sum(region_mask) == 0:
        raise RuntimeError(f"Target class region is empty for this image: class={target_class}")

    # Ensure texture line image exists and is up-to-date for this image.
    analyze_skin_texture(str(image_path), model_path=str(root / model_path), device=device)
    texture_path = root / "skin_output" / f"only_texture_line_{num}.png"
    tex = imread_unicode(str(texture_path), cv2.IMREAD_GRAYSCALE)
    if tex is None:
        raise FileNotFoundError(f"Texture image not found: {texture_path}")

    if tex.shape != pred.shape:
        tex = cv2.resize(tex, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    h, w = tex.shape
    radius = dynamic_radius_from_size(h, w) if fixed_radius is None else int(fixed_radius)
    kernel = build_disk_kernel(radius)

    if fixed_radius is None:
        print(f"Dynamic radius: {radius} (image size: {w}x{h})")
    else:
        print(f"Fixed radius: {radius} (image size: {w}x{h})")

    tex_norm = tex.astype(np.float32) / 255.0
    texture_binary = ((tex_norm > 0.4) & (region_mask > 0)).astype(np.float32)
    region_mask_f = region_mask.astype(np.float32)

    # Local texture density.
    valid_count = cv2.filter2D(region_mask_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    texture_count = cv2.filter2D(texture_binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    density = np.zeros_like(tex_norm, dtype=np.float32)
    valid_local = valid_count > 1e-6
    density[valid_local] = texture_count[valid_local] / valid_count[valid_local]

    # Local direction consistency using doubled-angle resultant length.
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
    score_norm = normalize_on_mask(score, region_mask)

    # Save outputs.
    heatmap_rgb = make_heatmap_image(score_norm, region_mask)
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

    original = imread_unicode(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if original.shape[:2] != heatmap_bgr.shape[:2]:
        original = cv2.resize(original, (heatmap_bgr.shape[1], heatmap_bgr.shape[0]))

    overlay = overlay_heatmap(original, heatmap_bgr, region_mask, alpha=0.55)
    overlay_with_bar = add_vertical_colorbar_bgr(overlay, cmap_name="jet")

    presence_map, presence_inside_mask, presence_quantiles = build_presence_map(
        score_norm,
        region_mask=region_mask,
    )
    # Overlay quartile colors only inside effective mask.
    presence_overlay = original.copy().astype(np.float32)
    for c in range(3):
        presence_overlay[:, :, c][presence_inside_mask] = (
            0.50 * presence_overlay[:, :, c][presence_inside_mask]
            + 0.50 * presence_map[:, :, c][presence_inside_mask]
        )
    presence_overlay = np.clip(presence_overlay, 0, 255).astype(np.uint8)

    # Draw effective-mask boundary for readability.
    presence_map = draw_mask_boundary(presence_map, region_mask, color_bgr=(0, 255, 255), thickness=2)
    presence_overlay = draw_mask_boundary(presence_overlay, region_mask, color_bgr=(0, 255, 255), thickness=2)

    cv2.imwrite(str(out_dir / f"{num}_class{target_class}_mask_raw.png"), raw_region_mask * 255)
    cv2.imwrite(str(out_dir / f"{num}_class{target_class}_mask_effective.png"), region_mask * 255)
    cv2.imwrite(str(out_dir / f"{num}_local_density.png"), (density * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / f"{num}_local_consistency.png"), (consistency * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / f"{num}_severity_map.png"), heatmap_bgr)
    cv2.imwrite(str(out_dir / f"{num}_severity_overlay.png"), overlay_with_bar)
    cv2.imwrite(str(out_dir / f"{num}_presence_map.png"), presence_map)
    cv2.imwrite(str(out_dir / f"{num}_presence_overlay.png"), presence_overlay)

    yx = np.where(region_mask > 0)
    vals = score_norm[yx]
    top_idx = np.argsort(vals)[-10:][::-1]
    top_rows = []
    for i in top_idx:
        y = int(yx[0][i])
        x = int(yx[1][i])
        v = float(vals[i])
        top_rows.append((x, y, v))

    txt_path = out_dir / f"{num}_top10_points.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"image={num}\n")
        f.write(f"target_class={target_class}\n")
        f.write(f"radius={radius}\n")
        f.write("presence_mode=quartile_inside_effective_mask\n")
        f.write(f"presence_q25={presence_quantiles[0]:.6f}\n")
        f.write(f"presence_q50={presence_quantiles[1]:.6f}\n")
        f.write(f"presence_q75={presence_quantiles[2]:.6f}\n")
        f.write("x\ty\tscore_norm\n")
        for x, y, v in top_rows:
            f.write(f"{x}\t{y}\t{v:.6f}\n")

    print("Done.")
    print(f"Output dir: {out_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Per-pixel local scoring heatmap in selected class region")
    parser.add_argument("num", help="Image id, e.g. 66")
    parser.add_argument("--model", default="best_trans_unet_model_20250614_122913.pth", help="Model checkpoint file name")
    parser.add_argument("--target-class", type=int, default=1, choices=[1, 2], help="Target region class label (1 or 2)")
    parser.add_argument("--radius", type=int, default=None, help="Fixed local radius in pixels; if omitted use dynamic radius")
    parser.add_argument("--output-subdir", default=None, help="Output subdirectory under heatmap_output/<num>/")
    args = parser.parse_args()

    run_for_id(
        args.num,
        model_path=args.model,
        target_class=args.target_class,
        fixed_radius=args.radius,
        output_subdir=args.output_subdir,
    )


if __name__ == "__main__":
    main()
