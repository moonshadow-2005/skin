import os
from pathlib import Path
import sys
import argparse

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skin import analyze_skin_texture
from predict import analyze_texture_orientation
from report import generate_report


def imread_unicode(image_path: str, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(image_path: str, image) -> bool:
    ext = Path(image_path).suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(image_path)
    return True


def overlay_images_unicode(original_path: str, direction_path: str, output_path: str) -> None:
    original = imread_unicode(original_path, cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"无法读取原始图片: {original_path}")

    direction = imread_unicode(direction_path, cv2.IMREAD_UNCHANGED)
    if direction is None:
        raise FileNotFoundError(f"无法读取方向图: {direction_path}")

    if direction.shape[:2] != original.shape[:2]:
        direction = cv2.resize(direction, (original.shape[1], original.shape[0]))

    if direction.ndim == 2:
        direction = cv2.cvtColor(direction, cv2.COLOR_GRAY2BGRA)
    elif direction.shape[2] == 3:
        alpha = np.full((direction.shape[0], direction.shape[1], 1), 255, dtype=np.uint8)
        direction = np.concatenate([direction, alpha], axis=2)

    direction_rgb = direction[:, :, :3].astype(np.float32)
    alpha = (direction[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    original_f = original.astype(np.float32)
    overlay = (1.0 - alpha) * original_f + alpha * direction_rgb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not imwrite_unicode(output_path, overlay):
        raise RuntimeError(f"无法写入最终叠加图: {output_path}")


def resolve_input_image(root: Path, image_or_id: str | None) -> Path:
    """Resolve input from explicit image path or case id.

    Priority:
    1) Existing path (absolute or relative to cwd/root)
    2) dataset/final_labeled/<id>.(jpg|png|jpeg)
    3) First jpg under dataset/final_labeled when not provided
    """
    data_dir = root / "dataset" / "final_labeled"

    if image_or_id is None:
        candidates = sorted(data_dir.glob("*.jpg"))
        if not candidates:
            raise FileNotFoundError(f"未找到输入图片，请传入路径或病例ID。目录: {data_dir}")
        return candidates[0]

    raw = Path(image_or_id)
    if raw.exists():
        return raw.resolve()

    raw_from_root = (root / raw).resolve()
    if raw_from_root.exists():
        return raw_from_root

    case_id = image_or_id
    for ext in [".jpg", ".png", ".jpeg"]:
        p = data_dir / f"{case_id}{ext}"
        if p.exists():
            return p

    raise FileNotFoundError(
        f"输入不存在: {image_or_id}。请传入有效图片路径，或确保 dataset/final_labeled/{case_id}.jpg 存在。"
    )


def resolve_orientation_output(root: Path, case_id: str) -> Path:
    """predict.py may use last token after underscore as output suffix; support both."""
    predict_dir = root / "predict_output"
    suffix = case_id.split("_")[-1]

    exact = predict_dir / f"orientation_only_texture_line_{case_id}.png"
    if exact.exists():
        return exact

    fallback = predict_dir / f"orientation_only_texture_line_{suffix}.png"
    if fallback.exists():
        return fallback

    wildcard = sorted(predict_dir.glob(f"orientation_only_texture_line_*{suffix}.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if wildcard:
        return wildcard[0]

    raise FileNotFoundError(f"未找到方向图: {exact} (或兼容后缀 {suffix})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one full pipeline for one image path or case id")
    parser.add_argument(
        "image_or_id",
        nargs="?",
        default=None,
        help="Image path, or case id from dataset/final_labeled (e.g. 66)",
    )
    parser.add_argument(
        "--model",
        default="best_trans_unet_model_20250614_122913.pth",
        help="Model checkpoint relative to project root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT
    image_path = resolve_input_image(root, args.image_or_id)

    num = image_path.stem

    print(f"开始处理: {image_path}")
    print(f"样本编号: {num}")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1) 纹理提取
    analyze_skin_texture(str(image_path), model_path=args.model, device=device)

    # 2) 方向分析
    texture_input = root / "skin_output" / f"only_texture_line_{num}.png"
    if not texture_input.exists():
        raise FileNotFoundError(f"未找到纹理线条图: {texture_input}")
    analyze_texture_orientation(str(texture_input))

    # 3) 最终叠加图
    direction_only = resolve_orientation_output(root, num)
    final_output = root / "final_output" / f"final_result_{num}.jpg"
    overlay_images_unicode(str(image_path), str(direction_only), str(final_output))

    # 4) 报告生成
    generate_report(num)

    print("\n=== 全流程完成 ===")
    print(f"纹理图: skin_output/only_texture_line_{num}.png")
    print(f"方向图: predict_output/orientation_only_texture_line_{num}.png")
    print(f"扇区图: predict_output/spatial_sector_directions_{num}.png")
    print(f"叠加图: final_output/final_result_{num}.jpg")
    print(f"报告: report/{num}_skin_texture_analysis_report.md")


if __name__ == "__main__":
    main()
