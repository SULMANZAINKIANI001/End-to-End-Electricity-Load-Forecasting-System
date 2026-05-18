from __future__ import annotations

import argparse
from pathlib import Path

from src.config import settings
from src.data import normalize_entsoe_export_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize an ENTSO-E Total Load export to timestamp,load_mw CSV.")
    parser.add_argument("--input", default=str(settings.entsoe_export_raw_path), help="Raw ENTSO-E CSV/XLSX export path.")
    parser.add_argument("--output", default=str(settings.de_lu_load_csv), help="Normalized output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = normalize_entsoe_export_file(Path(args.input), Path(args.output))
    print(f"Wrote {len(frame)} hourly rows to {args.output}")
    print(f"Range: {frame.index.min().isoformat()} to {frame.index.max().isoformat()}")


if __name__ == "__main__":
    main()
