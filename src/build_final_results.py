from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(image_path: Path, flags: int = cv2.IMREAD_UNCHANGED):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_png_unicode(out_path: Path, image: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"无法编码为PNG: {out_path}")
    encoded.tofile(str(out_path))


def convert_to_png(src: Path, dst_png: Path) -> bool:
    if not src.exists():
        return False
    img = imread_unicode(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    imwrite_png_unicode(dst_png, img)
    return True


def pick_existing_file(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def pick_latest_worst_direction(case_dir: Path) -> Path | None:
    worst_dir = case_dir / "07_worst_boxes"
    if not worst_dir.exists():
        return None

    candidates = sorted(worst_dir.glob("*_worst*_direction.png"))
    if not candidates:
        return None

    # Prefer latest file so it works with both worst80/worst32 updates.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def copy_sector_details(case_name: str, case_dir: Path, dst_dir: Path) -> bool:
    src_report = case_dir / "05_report"
    if not src_report.exists():
        return False

    src_pic = src_report / f"{case_name}_pic"
    if not src_pic.exists():
        pic_dirs = sorted([p for p in src_report.iterdir() if p.is_dir() and p.name.endswith("_pic")])
        if not pic_dirs:
            return False
        src_pic = pic_dirs[0]

    shutil.copytree(src_pic, dst_dir / "05_sector_details", dirs_exist_ok=True)
    return True


def process_one_case(case_dir: Path, output_root: Path) -> tuple[str, list[str], list[str]]:
    case_name = case_dir.name
    out_case = output_root / case_name
    out_case.mkdir(parents=True, exist_ok=True)

    done: list[str] = []
    missing: list[str] = []

    mapping = [
        (
            [case_dir / "01_segmentation" / f"{case_name}_overlay_lineonly.png"],
            out_case / "01_segment.png",
            "01_segment",
        ),
        (
            [case_dir / "02_texture" / f"texture_line_{case_name}.png"],
            out_case / "02_texture.png",
            "02_texture",
        ),
        (
            [case_dir / "03_orientation" / f"orientation_texture_line_{case_name}.png"],
            out_case / "03_orientation.png",
            "03_orientation",
        ),
        (
            [case_dir / "04_final_overlay" / f"final_result_{case_name}.jpg"],
            out_case / "04_orientation_overlay.png",
            "04_orientation_overlay",
        ),
        (
            [case_dir / "06_heatmap_r40" / f"{case_name}_severity_overlay.png"],
            out_case / "06_severity.png",
            "06_severity",
        ),
        (
            [case_dir / "06_heatmap_r40" / f"{case_name}_presence_overlay.png"],
            out_case / "08_presence_overlay.png",
            "08_presence_overlay",
        ),
    ]

    for src_candidates, dst_png, tag in mapping:
        src = pick_existing_file(src_candidates)
        if src is None or not convert_to_png(src, dst_png):
            missing.append(tag)
        else:
            done.append(tag)

    worst_src = pick_latest_worst_direction(case_dir)
    if worst_src is None or not convert_to_png(worst_src, out_case / "07_worst.png"):
        missing.append("07_worst")
    else:
        done.append("07_worst")

    if copy_sector_details(case_name, case_dir, out_case):
        done.append("05_sector_details")
    else:
        missing.append("05_sector_details")

    return case_name, done, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert results/* to final_results/* with simplified PNG outputs")
    parser.add_argument(
        "--results-root",
        default="results",
        help="Source root directory, default: results",
    )
    parser.add_argument(
        "--output-root",
        default="final_results",
        help="Target root directory, default: final_results",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional case names, e.g. --cases 66 67",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    results_root = (root / args.results_root).resolve()
    output_root = (root / args.output_root).resolve()

    if not results_root.exists():
        raise FileNotFoundError(f"results根目录不存在: {results_root}")

    if args.cases:
        case_dirs = [results_root / c for c in args.cases]
    else:
        case_dirs = sorted([p for p in results_root.iterdir() if p.is_dir()])

    if not case_dirs:
        print("未找到可处理的病例目录")
        return

    total = 0
    for case_dir in case_dirs:
        if not case_dir.exists() or not case_dir.is_dir():
            print(f"[SKIP] 不存在目录: {case_dir}")
            continue

        case_name, done, missing = process_one_case(case_dir, output_root)
        total += 1
        print(f"[{case_name}] 完成: {', '.join(done) if done else '无'}")
        if missing:
            print(f"[{case_name}] 缺失: {', '.join(missing)}")

    print(f"\n批处理结束，共处理 {total} 个病例。")
    print(f"输出目录: {output_root}")


if __name__ == "__main__":
    main()
