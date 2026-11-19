"""Build the evidence-linked cleaning audit, paid lines and basket-item events."""

import argparse
import gzip
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.baskets import build_basket_events
from src.data.clean import RAW_COLUMNS, WORKBOOK_SHA256, audit_transactions, read_workbook
from src.data.merchandise import merchandise_rules
from src.data.purchase_cleaning import clean_purchase_lines


def cleaning_manifest(raw, baseline, steps, purchases, events):
    return {
        "source": {
            "dataset_doi": "10.24432/C5BW33",
            "workbook_sha256": WORKBOOK_SHA256,
            "sheet": "Online Retail",
            "raw_fields": RAW_COLUMNS,
            "excel_first_data_row": 2,
            "rows": len(raw),
            "first_timestamp": raw["InvoiceDate"].min().isoformat(),
            "last_timestamp": raw["InvoiceDate"].max().isoformat(),
        },
        "merchandise_evidence_period": {"start": "2010-12-01", "end_exclusive": "2011-10-01"},
        "counts": {
            "after_five_rules": int(baseline.iloc[-1]["rows_after"]),
            "removed_total": int(steps["rows_removed"].sum()),
            "paid_merchandise_lines": len(purchases),
            "basket_item_events": len(events),
            "baskets": len(events[["CustomerID", "InvoiceNo"]].drop_duplicates()),
            "customers": purchases["CustomerID"].nunique(),
            "stock_codes": purchases["StockCode"].nunique(),
            "paid_units": int(purchases["Quantity"].sum()),
            "recorded_amount_gbp": float(format(purchases["line_amount_gbp"].sum(), ".15g")),
        },
        "event_key": ["CustomerID", "InvoiceNo", "StockCode"],
        "basket_time": "Latest retained line timestamp on the customer invoice.",
        "scope": "Whole-workbook cleaning. Slice by basket_time before constructing model histories or labels.",
    }


def save_cleaning_results(project, baseline_path, baseline, steps, decisions, purchases, events, rules, manifest):
    """Stage each artifact beside its destination, then publish the checksum manifest last."""
    results = project / "results"
    processed = project / "data/processed"
    frames = [
        (baseline_path, baseline),
        (results / "cleaning_steps.csv", steps),
        (results / "merchandise_rules.csv", rules),
        (processed / "row_decisions.csv.gz", decisions),
        (processed / "purchase_lines.csv.gz", purchases),
        (processed / "basket_events.csv.gz", events),
    ]
    manifest_path = results / "data_manifest.json"
    if baseline_path.suffix.lower() != ".csv":
        raise ValueError("Choose a .csv file for the five-rule audit.")
    if len({path.resolve() for path, _ in frames}) != len(frames):
        raise ValueError("Choose distinct output paths for each table.")

    with ExitStack() as stack:
        workdirs = {}
        for directory in {path.parent for path, _ in frames}:
            directory.mkdir(parents=True, exist_ok=True)
            workdirs[directory] = Path(stack.enter_context(TemporaryDirectory(dir=directory)))
        outputs = {}
        staged = []
        for path, frame in frames:
            temp = workdirs[path.parent] / path.name
            if path.suffix == ".gz":
                with temp.open("wb") as stream:
                    with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0, compresslevel=6) as zipped:
                        frame.to_csv(zipped, index=False, float_format="%.15g")
            else:
                frame.to_csv(temp, index=False)
            with temp.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            outputs[os.path.relpath(path, project)] = {"rows": len(frame), "sha256": digest}
            staged.append((temp, path))
        manifest_temp = workdirs[results] / manifest_path.name
        manifest_temp.write_text(json.dumps({**manifest, "outputs": outputs}, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        for temp, path in staged:
            temp.replace(path)
        manifest_temp.replace(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("."), help="Root for the workbook and generated files.")
    parser.add_argument("--workbook", type=Path, help="Defaults to data/raw/Online Retail.xlsx in the project.")
    parser.add_argument("--output", type=Path, help="Override the path of the original five-rule audit CSV.")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    raw = read_workbook(args.workbook or project / "data/raw/Online Retail.xlsx")
    baseline = audit_transactions(raw)
    steps, decisions, purchases = clean_purchase_lines(raw)
    events = build_basket_events(purchases)
    manifest = cleaning_manifest(raw, baseline, steps, purchases, events)
    save_cleaning_results(project, args.output or project / "results/cleaning_audit.csv",
                          baseline, steps, decisions, purchases, events, merchandise_rules(), manifest)
    print(steps[["step", "rule", "raw_flagged", "rows_removed", "rows_after"]].to_string(index=False))
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Saved the output manifest to {project / 'results/data_manifest.json'}")


if __name__ == "__main__":
    main()
