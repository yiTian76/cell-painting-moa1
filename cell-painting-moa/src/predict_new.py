"""Fit the experiment's channel-balanced classifier and rank MoAs for a new profile.

Use the same Cell Painting conditions, feature definitions and source-level
normalization as the reference table. Cosine scores are similarities on the
training label set. Query labels, if supplied, are copied only after scoring.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_experiment import assign_block, fit_transform_fold, read_profiles, score_blocks, stable_ranking


def connectivity_key(value):
    # The reference cohort treats salts/stereochemical variants that share
    # the first InChIKey block as one group. Match full keys the same way.
    key = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z]{14}(?:-[A-Z]{10}-[A-Z])?", key):
        raise ValueError(f"Expected InChIKey14 or a full InChIKey in compound_id: {value!r}")
    return key[:14]


def read_query(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    suffixes = [s.lower() for s in path.suffixes]
    delimiter = "\t" if ".tsv" in suffixes or ".tab" in suffixes else ","
    return pd.read_csv(path, sep=delimiter, dtype={"compound_id": "string"}, low_memory=False)


def predict(train, query, min_compounds, overlap, features=None):
    options = argparse.Namespace(input=train, features=features, min_compounds=min_compounds)
    names, x, feat, _, _ = read_profiles(options)
    train_ids = pd.Index([connectivity_key(i) for i in names.index])
    if not train_ids.is_unique:
        raise ValueError("Prepare the training reference with one compound per InChIKey14 first.")
    data = read_query(query)
    if data.empty or "compound_id" not in data:
        raise ValueError("Provide at least one query row with a compound_id.")
    ids = data.compound_id.map(connectivity_key)
    if not ids.is_unique:
        raise ValueError("Multiple query rows share an InChIKey14. Supply a single consensus profile per compound.")
    missing = [col for col in feat if col not in data]
    if missing:
        raise ValueError(f"{len(missing)} reference feature columns are missing. Examples: {', '.join(missing[:5])}")
    shared = train_ids.intersection(ids)
    if len(shared) and overlap == "error":
        raise ValueError(f"{len(shared)} query compound IDs occur in the eligible training reference, "
                         f"including {shared[0]}. Pass --overlap holdout to remove all query IDs before fitting.")
    remaining = ~train_ids.isin(ids)
    moa = names.moa.to_numpy(dtype=str)
    remaining_counts = pd.Series(moa[remaining]).value_counts()
    supported = remaining_counts.index[remaining_counts >= min_compounds]
    fit = remaining & np.isin(moa, supported)
    if len(supported) < 2:
        raise ValueError("Fewer than two eligible classes remain after excluding query IDs.")
    assert not len(train_ids[fit].intersection(ids))
    labels, y = np.unique(moa[fit], return_inverse=True)
    q = data[list(feat)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    tr_z, q_z, keep = fit_transform_fold(x[fit], q)
    no_observations = ~np.isfinite(q[:, keep]).any(axis=1)
    if no_observations.any():
        raise ValueError(f"Query has no observed retained features: {ids[no_observations].iloc[0]}")
    blocks = np.array([assign_block(s) for s in feat[keep]])
    scores = score_blocks(tr_z, q_z, y, blocks, len(labels))[0]
    ranks = stable_ranking(scores)
    result = pd.DataFrame({"compound_id": ids})
    if "compound_name" in data:
        result["compound_name"] = data.compound_name
    for j in range(min(3, len(labels))):
        result[f"top{j+1}"] = labels[ranks[:,j]]
        result[f"score{j+1}"] = scores[np.arange(len(data)),ranks[:,j]]
    if "moa" in data:
        result["provided_moa"] = data.moa
        result["provided_moa_in_training_classes"] = data.moa.isin(labels)
    log = {"training_compounds": int(fit.sum()), "training_classes": len(labels),
           "query_compounds": len(result), "removed_query_ids_from_training": len(shared),
           "eligible_reference_compounds_before_holdout": len(train_ids),
           "retained_features": int(keep.sum()), "overlap_policy": overlap}
    return result, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path, help="Prepared wide reference CSV, CSV.GZ or Parquet")
    parser.add_argument("--query", required=True, type=Path, help="New CSV/TSV, optionally gzipped, or Parquet")
    parser.add_argument("--out", required=True, type=Path, help="Output predictions CSV or CSV.GZ")
    parser.add_argument("--min-compounds", type=int, default=5)
    parser.add_argument("--features", type=Path, help="Optional reference feature allowlist, one name per line")
    parser.add_argument("--overlap", choices=["error", "holdout"], default="error")
    args = parser.parse_args()
    if args.min_compounds < 2:
        parser.error("At least two training compounds per class are required.")
    if args.out.resolve() in (args.query.resolve(), args.train.resolve()):
        parser.error("Choose an output path different from the input files.")
    try:
        with threadpool_limits(limits=1):
            result, log = predict(args.train, args.query, args.min_compounds, args.overlap, args.features)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, float_format="%.10g")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
