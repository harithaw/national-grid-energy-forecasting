from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_INPUT_DIR = Path("data")
DEFAULT_OUTPUT_FILE = Path("data/merged-generation-profile-data.csv")
MISSING_VALUE = "Data N/A"

KNOWN_COLUMN_ORDER = [
    "Date (GMT+5:30)",
    "Solar (Energy in GWh/day)",
    "Rooftop Solar (Est.) (Energy in GWh/day)",
    "Solar (Estimated) (Energy in GWh/day)",
    "Mini Hydro (Telemetered) (Energy in GWh/day)",
    "Mini Hydro (Estimated) (Energy in GWh/day)",
    "Biomass and Waste Heat (Energy in GWh/day)",
    "Wind (Energy in GWh/day)",
    "Major Hydro (Energy in GWh/day)",
    "Oil (IPP) (Energy in GWh/day)",
    "Oil (CEB) (Energy in GWh/day)",
    "Coal (Energy in GWh/day)",
]


def normalize_header(header: str) -> str:
    return " ".join(header.strip().lower().split())


def canonicalize_header(header: str) -> str:
    normalized = normalize_header(header)

    if normalized.startswith("date"):
        return "Date (GMT+5:30)"
    if "rooftop" in normalized and "solar" in normalized:
        return "Rooftop Solar (Est.) (Energy in GWh/day)"
    if "solar" in normalized and "estimated" in normalized:
        return "Solar (Estimated) (Energy in GWh/day)"
    if "solar" in normalized:
        return "Solar (Energy in GWh/day)"
    if "mini hydro" in normalized and "telemetered" in normalized:
        return "Mini Hydro (Telemetered) (Energy in GWh/day)"
    if "mini hydro" in normalized and "estimated" in normalized:
        return "Mini Hydro (Estimated) (Energy in GWh/day)"
    if "biomass" in normalized:
        return "Biomass and Waste Heat (Energy in GWh/day)"
    if normalized.startswith("wind"):
        return "Wind (Energy in GWh/day)"
    if "major hydro" in normalized:
        return "Major Hydro (Energy in GWh/day)"
    if "oil" in normalized and "ipp" in normalized:
        return "Oil (IPP) (Energy in GWh/day)"
    if "oil" in normalized and "ceb" in normalized:
        return "Oil (CEB) (Energy in GWh/day)"
    if normalized.startswith("coal"):
        return "Coal (Energy in GWh/day)"

    return header.strip()


def find_source_files(input_dir: Path) -> List[Path]:
    csv_files = sorted(input_dir.glob("generation-profile-data-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No source CSV files found in: {input_dir}")
    return csv_files


def build_merged_schema(source_files: Iterable[Path]) -> List[str]:
    discovered_columns = []
    seen = set()

    for file_path in source_files:
        with file_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if not header:
                continue

            for column in header:
                canonical = canonicalize_header(column)
                if canonical not in seen:
                    seen.add(canonical)
                    discovered_columns.append(canonical)

    ordered = [column for column in KNOWN_COLUMN_ORDER if column in seen]
    extras = sorted(column for column in discovered_columns if column not in KNOWN_COLUMN_ORDER)
    return ordered + extras


def clean_value(value: str | None) -> str:
    if value is None:
        return MISSING_VALUE

    stripped = value.strip()
    return stripped if stripped else MISSING_VALUE


def merge_csv_files(source_files: Iterable[Path], output_file: Path) -> int:
    source_files = list(source_files)
    schema = build_merged_schema(source_files)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with output_file.open("w", newline="", encoding="utf-8") as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=schema)
        writer.writeheader()

        for file_path in source_files:
            with file_path.open("r", newline="", encoding="utf-8") as in_csv:
                reader = csv.DictReader(in_csv)
                if reader.fieldnames is None:
                    continue

                column_map: Dict[str, str] = {
                    source_name: canonicalize_header(source_name)
                    for source_name in reader.fieldnames
                }

                for row in reader:
                    merged_row = {column: MISSING_VALUE for column in schema}

                    for source_name, raw_value in row.items():
                        canonical = column_map.get(source_name)
                        if canonical is None:
                            continue

                        merged_row[canonical] = clean_value(raw_value)

                    writer.writerow(merged_row)
                    total_rows += 1

    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge generation profile CSV files with schema alignment and Data N/A fill values.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing yearly CSV files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Merged CSV output path (default: {DEFAULT_OUTPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_files = find_source_files(args.input_dir)
    rows = merge_csv_files(source_files, args.output)

    print(f"Merged {len(source_files)} files into '{args.output}' with {rows} rows.")


if __name__ == "__main__":
    main()
