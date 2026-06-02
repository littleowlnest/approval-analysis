from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run loan/application approval profile analysis on an Excel workbook."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the .xlsx file. Defaults to the newest Excel file in input/.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Folder to scan for an .xlsx file when --input is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Folder where analysis outputs are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_analysis(
        input_path=args.input,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print(result.message)
    if result.report_path:
        print(f"Report: {result.report_path}")
    return 0
