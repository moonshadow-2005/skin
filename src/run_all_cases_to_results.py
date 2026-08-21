from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .run_case_to_results import run_case
except ImportError:
    from src.run_case_to_results import run_case
from src.project_paths import DEFAULT_MODEL_NAME, FINAL_RESULTS_DIR, LABELED_DATASET_DIR, RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run dataset images to runtime/results/<id> and runtime/final_results/<id>"
    )
    parser.add_argument(
        "--data-dir",
        default=str(LABELED_DATASET_DIR),
        help="Input image directory",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model checkpoint",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=40,
        help="Heatmap and worst-box local radius",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=80,
        help="Worst-box size",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases that already have runtime/final_results/<id>/08_presence_overlay.png",
    )
    parser.add_argument(
        "--only-cases",
        nargs="*",
        default=None,
        help="Optional explicit case ids, e.g. --only-cases 30 66 100",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT
    raw_data_dir = Path(args.data_dir)
    data_dir = raw_data_dir.resolve() if raw_data_dir.is_absolute() else (root / raw_data_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    if args.only_cases:
        case_ids = sorted({str(x) for x in args.only_cases})
    else:
        case_ids = sorted({p.stem for p in data_dir.glob("*.jpg")})

    if not case_ids:
        print(f"未在目录中找到jpg: {data_dir}")
        return

    total = len(case_ids)
    success = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    print(f"总病例数: {total}")
    print(f"数据目录: {data_dir}")
    print(f"模型: {args.model}")

    for idx, case_id in enumerate(case_ids, start=1):
        final_flag = FINAL_RESULTS_DIR / case_id / "08_presence_overlay.png"
        if args.skip_existing and final_flag.exists():
            skipped += 1
            print(f"[{idx}/{total}] 跳过 {case_id} (已存在 {final_flag.name})")
            continue

        print(f"\n[{idx}/{total}] 开始处理 {case_id}")
        try:
            run_case(
                case_id=case_id,
                model_ckpt=args.model,
                radius=args.radius,
                box_size=args.box_size,
            )
            success += 1
            print(f"[{idx}/{total}] 完成 {case_id}")
        except Exception as exc:
            failed.append((case_id, str(exc)))
            print(f"[{idx}/{total}] 失败 {case_id}: {exc}")
            traceback.print_exc()

    print("\n=== 批处理结束 ===")
    print(f"成功: {success}")
    print(f"跳过: {skipped}")
    print(f"失败: {len(failed)}")

    if failed:
        fail_log = RESULTS_DIR / "batch_run_case_failures.txt"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        with fail_log.open("w", encoding="utf-8") as f:
            for case_id, err in failed:
                f.write(f"{case_id}\t{err}\n")
        print(f"失败日志: {fail_log}")


if __name__ == "__main__":
    main()
