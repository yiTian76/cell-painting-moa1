"""Assess how well plate design predicts MoA using the saved compound folds.

Inputs are the curated 10 uM cohort and the primary model's folds.csv.
Only source plate-map IDs and/or well row/column are used as predictors.
Uses the Python standard library.
"""
import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def audit_data(profiles, split_file):
    with split_file.open(encoding="utf-8-sig", newline="") as handle:
        split = {r["compound_id"]: r for r in csv.DictReader(handle)}
    rows = []
    with gzip.open(profiles, "rt", encoding="utf8", newline="") as handle:
        for r in csv.DictReader(handle):
            key = r["compound_id"]
            if key not in split:
                continue
            if split[key]["moa"] != r["moa"]:
                raise ValueError(f"Split and profiles disagree for {key}")
            wells = [re.fullmatch(r"([A-Z]+)([0-9]+)", v) for v in r["Metadata_wells"].split("|")]
            if not all(wells):
                raise ValueError(f"Unexpected well format for {key}")
            rows.append({
                "compound_id": key, "moa": r["moa"], "fold": int(split[key]["fold"]),
                "map": r["Metadata_plate_maps"].split("|"),
                "row": sorted({w.group(1) for w in wells}),
                "column": sorted({str(int(w.group(2))) for w in wells}),
            })
    if len(rows) != len(split) or len({r["compound_id"] for r in rows}) != len(rows):
        raise ValueError("Input must contain exactly one row for every compound in the saved folds.")
    return rows


def fit_layout(train):
    counts = Counter(r["moa"] for r in train)
    frequencies = {f: defaultdict(Counter) for f in ("map", "row", "column")}
    for r in train:
        for field, table in frequencies.items():
            for value in r[field]:
                table[value][r["moa"]] += 1 / len(r[field])
    return counts, frequencies


def predict(row, labels, counts, tables):
    # Average category evidence if a profile aggregates several source materials.
    # Laplace smoothing adds one observation per category. No hyperparameter is fitted.
    map_scores = {label: 0 for label in labels}
    combined = {label: math.log(counts[label] + 1) for label in labels}
    for field, table in tables.items():
        for label in labels:
            mean_count = sum(table.get(v, {}).get(label, 0) for v in row[field]) / len(row[field])
            # Reserve one extra category for unseen values.
            combined[label] += math.log((mean_count + 1) / (counts[label] + len(table) + 1))
            if field == "map":
                map_scores[label] = sum(
                    (table.get(v, {}).get(label, 0) + 1) / (sum(table.get(v, {}).values()) + len(labels))
                    for v in row[field]
                ) / len(row[field])
    scores = {"majority": counts, "map_only": map_scores, "map_row_column": combined}
    return {method: sorted(labels, key=lambda c: -values.get(c, 0))[:3] for method, values in scores.items()}


def all_metrics(predictions, labels):
    true_count = Counter(r["moa"] for r in predictions)
    pred_count = Counter(r["top1"] for r in predictions)
    hits = Counter(r["moa"] for r in predictions if r["moa"] == r["top1"])
    return {
        "accuracy": sum(hits.values()) / len(predictions),
        "balanced_accuracy": sum(hits[c] / true_count[c] for c in labels) / len(labels),
        "macro_f1": sum(2 * hits[c] / (true_count[c] + pred_count[c]) for c in labels) / len(labels),
        "top3_accuracy": sum(r["moa"] in (r["top1"], r["top2"], r["top3"]) for r in predictions) / len(predictions),
    }


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(profiles, fold_file, dest):
    rows = audit_data(profiles, fold_file)
    labels = sorted({r["moa"] for r in rows})
    dest.mkdir(parents=True, exist_ok=True)
    pooled = defaultdict(list)
    for fold in sorted({r["fold"] for r in rows}):
        train = [r for r in rows if r["fold"] != fold]
        counts, tables = fit_layout(train)
        for row in (r for r in rows if r["fold"] == fold):
            for method, ranked in predict(row, labels, counts, tables).items():
                pooled[method].append({
                    "compound_id": row["compound_id"], "fold": fold, "moa": row["moa"], "method": method,
                    "top1": ranked[0], "top2": ranked[1], "top3": ranked[2],
                })
    metrics = [{"method": method, **all_metrics(result, labels)} for method, result in pooled.items()]
    save_csv(dest / "layout_metrics.csv", metrics)
    save_csv(dest / "layout_oof.csv", [r for result in pooled.values() for r in result])
    plate_counts = {label: len({p for r in rows if r["moa"] == label for p in r["map"]}) for label in labels}
    coverage = {"compounds": len(rows), "classes": len(labels), "plate_map_count": len({v for r in rows for v in r["map"]}),
                "minimum_maps_per_class": min(plate_counts.values()), "maximum_maps_per_class": max(plate_counts.values()),
                "maps_per_moa": plate_counts}
    (dest / "layout_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf8")
    print(*metrics, sep="\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--folds", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    run(a.profiles, a.folds, a.output)
