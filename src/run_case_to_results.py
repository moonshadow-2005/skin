from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .build_final_results import process_one_case
    from .local_score_heatmap import run_for_id
    from .run_one_full_pipeline import overlay_images_unicode
except ImportError:
    from src.build_final_results import process_one_case
    from src.local_score_heatmap import run_for_id
    from src.run_one_full_pipeline import overlay_images_unicode

from report import generate_report
from skin import analyze_skin_texture
from test import visualize_prediction
from predict import analyze_texture_orientation


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def copy_tree_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def run_case(case_id: str, model_ckpt: str, radius: int, box_size: int) -> None:
    root = PROJECT_ROOT
    img_path = root / "dataset" / "final_labeled" / f"{case_id}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {img_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_case = root / "results" / case_id
    seg_dir = results_case / "01_segmentation"
    tex_dir = results_case / "02_texture"
    ori_dir = results_case / "03_orientation"
    final_dir = results_case / "04_final_overlay"
    rep_dir = results_case / "05_report"
    heat_dir = results_case / "06_heatmap_r40"
    worst_dir = results_case / "07_worst_boxes"

    for d in [seg_dir, tex_dir, ori_dir, final_dir, rep_dir, heat_dir, worst_dir]:
        ensure_dir(d)

    print(f"[1/8] 分割可视化 -> {seg_dir}")
    visualize_prediction(
        image_path=str(img_path),
        model_path=str(root / model_ckpt),
        output_dir=str(seg_dir),
        device=device,
    )

    print(f"[2/8] 纹理线提取")
    analyze_skin_texture(str(img_path), model_path=str(root / model_ckpt), device=device)

    copy_if_exists(root / "skin_output" / f"texture_line_{case_id}.png", tex_dir / f"texture_line_{case_id}.png")
    copy_if_exists(root / "skin_output" / f"only_texture_line_{case_id}.png", tex_dir / f"only_texture_line_{case_id}.png")

    print(f"[3/8] 方向分析")
    texture_input = root / "skin_output" / f"only_texture_line_{case_id}.png"
    if not texture_input.exists():
        raise FileNotFoundError(f"纹理输入不存在: {texture_input}")
    analyze_texture_orientation(str(texture_input))

    for name in [
        f"orientation_only_texture_line_{case_id}.png",
        f"orientation_texture_line_{case_id}.png",
        f"spatial_sector_directions_{case_id}.png",
        f"sector_info_{case_id}.json",
        f"sector_info_{case_id}.pkl",
    ]:
        copy_if_exists(root / "predict_output" / name, ori_dir / name)

    print(f"[4/8] 最终叠加图")
    final_output = root / "final_output" / f"final_result_{case_id}.jpg"
    overlay_images_unicode(
        str(img_path),
        str(root / "predict_output" / f"orientation_only_texture_line_{case_id}.png"),
        str(final_output),
    )
    copy_if_exists(final_output, final_dir / f"final_result_{case_id}.jpg")

    print(f"[5/8] 报告生成")
    generate_report(case_id)
    copy_if_exists(root / "report" / f"{case_id}_skin_texture_analysis_report.md", rep_dir / f"{case_id}_skin_texture_analysis_report.md")
    copy_tree_if_exists(root / "report" / f"{case_id}_pic", rep_dir / f"{case_id}_pic")

    print(f"[6/8] 局部严重度图")
    run_for_id(
        case_id,
        model_path=model_ckpt,
        target_class=1,
        fixed_radius=radius,
        output_subdir="r40",
    )
    src_heat = root / "heatmap_output" / case_id / "r40"
    for p in src_heat.iterdir():
        if p.is_file():
            copy_if_exists(p, heat_dir / p.name)

    print(f"[7/8] 最严重框")
    cmd = [
        sys.executable,
        str(root / "src" / "worst_box_direction.py"),
        case_id,
        "--radius",
        str(radius),
        "--box-size",
        str(box_size),
        "--output-subdir",
        "r40",
        "--model",
        model_ckpt,
    ]
    subprocess.run(cmd, check=True, cwd=str(root))

    for p in src_heat.glob(f"{case_id}_worst*_*"):
        if p.is_file():
            copy_if_exists(p, worst_dir / p.name)

    print(f"[8/8] 生成精简目录 final_results/{case_id}")
    process_one_case(results_case, root / "final_results")

    print("\n完成")
    print(f"results目录: {results_case}")
    print(f"final_results目录: {root / 'final_results' / case_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one case pipeline to results/<id> and final_results/<id>")
    parser.add_argument("case_id", help="Case id, e.g. 30")
    parser.add_argument("--model", default="best_trans_unet_model_20250614_122913.pth", help="Model checkpoint")
    parser.add_argument("--radius", type=int, default=40, help="Heatmap and worst-box local radius")
    parser.add_argument("--box-size", type=int, default=80, help="Worst-box size")
    args = parser.parse_args()

    run_case(case_id=args.case_id, model_ckpt=args.model, radius=args.radius, box_size=args.box_size)


if __name__ == "__main__":
    main()
