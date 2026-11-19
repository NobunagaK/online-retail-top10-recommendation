"""Summarize net retained purchases at the training cutoff."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data.training_eda import (
    TRAIN_END, TRAIN_START, distribution_tables, monthly_activity, select_training_data,
)
from src.data.training_eda_plots import build_training_figures
from src.data.training_preference import training_preference


EVENT_PATH = "data/processed/training_basket_events.csv.gz"
LINE_PATH = "data/processed/training_purchase_lines.csv.gz"
MANIFEST_PATH = "results/data_manifest.json"
QUANTITY_POLICY = "net_retained_all_matched_credits_v1"


def load_training_inputs(project):
    """Load the D1 snapshot that uses only credits observed before the training cutoff."""
    manifest = json.loads((project / MANIFEST_PATH).read_text(encoding="utf-8"))
    if manifest.get("recommendation_quantity_policy") != QUANTITY_POLICY:
        raise ValueError("Run D1 cleaning to build a net retained purchase snapshot for this EDA.")
    input_hashes = {}
    if not {EVENT_PATH, LINE_PATH}.issubset(manifest["outputs"]):
        raise ValueError("Regenerate the D1 cutoff-specific training snapshot and its manifest.")
    snapshot = manifest["training_snapshot"]
    if (pd.Timestamp(snapshot["start_inclusive"]) != TRAIN_START
            or pd.Timestamp(snapshot["end_exclusive"]) != TRAIN_END
            or pd.Timestamp(snapshot["credit_time_end_exclusive"]) != TRAIN_END):
        raise ValueError("The D1 snapshot uses a different basket or credit time window.")
    for relative in (EVENT_PATH, LINE_PATH):
        with (project / relative).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != manifest["outputs"][relative]["sha256"]:
            raise ValueError(f"File differs from its D1 cleaning manifest: {relative}")
        input_hashes[relative] = digest
    identifiers = {"CustomerID": "string", "InvoiceNo": "string", "StockCode": "string"}
    events = pd.read_csv(project / EVENT_PATH, dtype=identifiers, keep_default_na=False,
                         parse_dates=["basket_time"])
    lines = pd.read_csv(project / LINE_PATH, dtype=identifiers, keep_default_na=False,
                        usecols=["ExcelRow", *identifiers, "InvoiceDate", "Quantity", "UnitPrice",
                                 "original_quantity", "matched_credit_units", "cancelled_units",
                                 "returned_units", "other_matched_units"],
                        parse_dates=["InvoiceDate"])
    if not lines["Quantity"].eq(lines["original_quantity"] - lines["matched_credit_units"]).all():
        raise ValueError("The surviving quantity must subtract all matched credits at the training cutoff.")
    return events, lines, input_hashes


def summarize_window(train_events, train_lines, preferences, input_hashes):
    return {
        "window": {"field": "basket_time", "start_inclusive": TRAIN_START.isoformat(),
                   "end_exclusive": TRAIN_END.isoformat()},
        "recommendation_quantity_policy": QUANTITY_POLICY,
        "input_sha256": input_hashes,
        "source_lines": len(train_lines),
        "baskets": len(train_events[["CustomerID", "InvoiceNo"]].drop_duplicates()),
        "units": int(train_events["quantity"].sum()),
        "recorded_amount_gbp": float(format(train_events["amount_gbp"].sum(), ".15g")),
        **preferences,
        "definitions": {
            "source_lines": "Original positive sale lines with a positive remaining quantity at the exclusive training cutoff.",
            "units": "Original sale units less all matched credit units issued strictly before the cutoff, including operational cancellation and return proxies.",
            "recorded_amount_gbp": "Net retained units multiplied by the original sale line price, summed over surviving training purchases.",
            "repeat_event_share": "Net retained basket-item events with an earlier net retained basket for the same customer and SKU / all net retained training basket-item events. Equal timestamps provide no earlier history.",
            "repeated_pair_share": "Observed customer-SKU pairs with a strictly later purchase / all observed training customer-SKU pairs.",
            "interaction_sparsity": "1 - observed distinct customer-SKU pairs / (active customers x observed SKUs).",
            "head_items": "Ceiling of 10% of observed SKUs, ranked by distinct customers descending, then exact StockCode ascending.",
            "head_event_share": "Events for head SKUs / all training events.",
            "head_customer_reach_share": "Distinct customers with any head SKU / distinct active training customers.",
        },
    }


def save_training_eda(output_dir, tables, summary, figures):
    """Stage the tables and figures together and replace the summary last."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as dirname:
        staged = Path(dirname)
        paths = []
        for name, table in tables.items():
            path = staged / f"{name}.csv"
            table.to_csv(path, index=False, float_format="%.15g")
            paths.append(path)
        for name, figure in figures.items():
            path = staged / f"{name}.png"
            figure.savefig(path, format="png", dpi=170, bbox_inches="tight", facecolor="white")
            paths.append(path)
        summary_path = staged / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                encoding="utf-8")
        for path in [*paths, summary_path]:
            path.replace(output_dir / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("."),
                        help="Project folder containing the outputs of scripts.prepare_data.")
    project = parser.parse_args().project_dir.resolve()
    events, lines, input_hashes = load_training_inputs(project)
    train_events, train_lines = select_training_data(events, lines)
    monthly = monthly_activity(train_events)
    frequency, distributions = distribution_tables(train_events, train_lines)
    popularity, preferences = training_preference(train_events)
    tables = {"monthly_activity": monthly, "distributions": frequency,
              "distribution_summary": distributions, "item_popularity": popularity}
    summary = summarize_window(train_events, train_lines, preferences, input_hashes)
    figures = build_training_figures(tables)
    output_dir = project / "results/training_eda"
    save_training_eda(output_dir, tables, summary, figures)
    print(f"{len(train_events):,} training events in {summary['baskets']:,} baskets")
    print(f"Saved training-period tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
