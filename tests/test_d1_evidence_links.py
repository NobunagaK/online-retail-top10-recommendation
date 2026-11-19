"""Check that the submitted row-decision audit can be traced to the evidence workbook."""

import csv
from pathlib import Path

from openpyxl import load_workbook


def test_each_published_cleaning_step_identifies_its_evidence_and_explains_the_decision():
    results = Path(__file__).resolve().parents[1] / "results"
    workbook = load_workbook(results / "transaction_exceptions.xlsx", read_only=True)
    try:
        sheets = set(workbook.sheetnames)
    finally:
        workbook.close()

    with (results / "cleaning_steps.csv").open(encoding="utf-8", newline="") as stream:
        steps = list(csv.DictReader(stream))
    assert steps
    for step in steps:
        rule = step["rule"]
        assert step["reason_en"].strip() and step["reason_zh"].strip(), rule
        for reference in step["evidence_sheets"].split(","):
            name = reference.strip()
            assert name in sheets or (results / name).is_file(), (rule, name)
