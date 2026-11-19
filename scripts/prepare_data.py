"""Run the D1 row-removal audit from the repository root."""

import argparse
from pathlib import Path

from src.data.clean import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path("data/raw/Online Retail.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("results/cleaning_audit.csv"))
    args = parser.parse_args()
    audit = run_audit(args.workbook, args.output)
    print(audit.to_string(index=False))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
