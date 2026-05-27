import traceback
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test import visualize_prediction


def main() -> None:
    root = PROJECT_ROOT
    input_dir = root / "dataset" / "final_labeled"
    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No .jpg files found in: {input_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = str(root / "best_trans_unet_model_20250614_122913.pth")

    print(f"Input dir: {input_dir}")
    print(f"Results root: {results_root}")
    print(f"Total images: {len(image_paths)}")
    print(f"Device: {device}")

    success = 0
    failed = 0
    failed_items = []

    for idx, image_path in enumerate(image_paths, start=1):
        image_name = image_path.stem
        output_dir = results_root / image_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}")
        try:
            visualize_prediction(
                image_path=str(image_path),
                model_path=model_path,
                output_dir=str(output_dir),
                device=device,
            )
            success += 1
        except Exception as exc:
            failed += 1
            failed_items.append((str(image_path), str(exc)))
            print(f"ERROR on {image_path.name}: {exc}")
            traceback.print_exc()

    print("\n=== Batch done ===")
    print(f"Success: {success}")
    print(f"Failed: {failed}")

    if failed_items:
        fail_log = results_root / "batch_failures.txt"
        with fail_log.open("w", encoding="utf-8") as f:
            for path, err in failed_items:
                f.write(f"{path}\t{err}\n")
        print(f"Failure log saved to: {fail_log}")


if __name__ == "__main__":
    main()
