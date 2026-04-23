#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FILENAME_PATTERN = re.compile(r"^ntt_(\d+)core_logn(\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compute average execution time excluding first run for each CSV."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=base_dir,
        help="Directory containing input CSV files (default: script directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=base_dir / "exectime_avg.csv",
        help="Output CSV path (default: exectime_avg.csv in script directory).",
    )
    return parser.parse_args()


def read_values(csv_path: Path) -> list[float]:
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                values.append(float(text))
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{line_no} is not numeric: {text}") from exc
    return values


def collect_rows(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        match = FILENAME_PATTERN.match(csv_path.name)
        if not match:
            continue

        core = int(match.group(1))
        logn = int(match.group(2))
        values = read_values(csv_path)
        if len(values) < 2:
            raise ValueError(f"{csv_path} must contain at least 2 values.")

        filtered = values[1:]
        avg = sum(filtered) / len(filtered)
        rows.append(
            {
                "file": csv_path.name,
                "core": str(core),
                "logn": str(logn),
                "total_runs": str(len(values)),
                "used_runs": str(len(filtered)),
                "avg_us": f"{avg:.6f}",
            }
        )

    rows.sort(key=lambda x: (int(x["core"]), int(x["logn"])))
    return rows


def write_output(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "core",
                "logn",
                "total_runs",
                "used_runs",
                "avg_us",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.input_dir)
    if not rows:
        raise FileNotFoundError(f"No matching CSV files found in {args.input_dir}")
    write_output(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
