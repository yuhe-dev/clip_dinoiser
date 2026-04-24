import argparse
import json
import os
import sys
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a directory of supervised probe result.json files and print deltas vs a reference run."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Directory whose immediate children contain result.json files.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional child directory name to use as the delta reference. Defaults to lexical first result.",
    )
    return parser.parse_args()


def _load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["metrics"]["summary"]
        rows.append(
            {
                "name": path.parent.name,
                "path": str(path),
                "mIoU": float(summary["mIoU"]),
                "mAcc": float(summary["mAcc"]),
                "aAcc": float(summary["aAcc"]),
                "train_seconds": float(payload["timing"]["train_seconds"]),
                "total_seconds": float(payload["timing"]["total_seconds"]),
                "subset_size": int(payload["subset_size"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    rows = _load_rows(root)
    if not rows:
        raise SystemExit(f"No result.json files found under {root}")

    by_name = {row["name"]: row for row in rows}
    reference_name = str(args.reference) if args.reference else rows[0]["name"]
    if reference_name not in by_name:
        raise SystemExit(f"Reference '{reference_name}' not found under {root}")
    reference = by_name[reference_name]

    print(f"reference={reference_name}")
    for row in rows:
        print(
            f"{row['name']}: "
            f"mIoU={row['mIoU']:.2f} "
            f"(Δ{row['mIoU'] - reference['mIoU']:+.2f}), "
            f"mAcc={row['mAcc']:.2f} "
            f"(Δ{row['mAcc'] - reference['mAcc']:+.2f}), "
            f"aAcc={row['aAcc']:.2f} "
            f"(Δ{row['aAcc'] - reference['aAcc']:+.2f}), "
            f"train_s={row['train_seconds']:.1f}, total_s={row['total_seconds']:.1f}"
        )


if __name__ == "__main__":
    main()
