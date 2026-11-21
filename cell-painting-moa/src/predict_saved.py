"""Rank mechanisms for new Cell Painting profiles using a fitted NPZ model."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def connectivity_key(value):
    key = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z]{14}(?:-[A-Z]{10}-[A-Z])?", key):
        raise ValueError(f"Expected InChIKey14 or full InChIKey in compound_id: {value!r}")
    return key[:14]


def read_query(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    suffixes = {s.lower() for s in path.suffixes}
    sep = "\t" if suffixes & {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep, dtype={"compound_id": "string"}, low_memory=False)


def predict(model_path, query_path):
    data = read_query(query_path)
    if data.empty or "compound_id" not in data:
        raise ValueError("Each query needs at least one compound and a compound_id column.")
    ids = data["compound_id"].map(connectivity_key)
    if not ids.is_unique:
        raise ValueError("Provide one profile per unique compound structure (InChIKey14).")
    with np.load(model_path, allow_pickle=False) as model:
        if model["schema_version"].item() != 1:
            raise ValueError("Unsupported model schema version.")
        overlap = pd.Index(model["training_ids"]).intersection(ids)
        if len(overlap):
            raise ValueError(f"Query includes {len(overlap)} fitted training structures, including {overlap[0]}.")
        features = model["features"]
        missing = [s for s in features if s not in data]
        if missing:
            raise ValueError(f"{len(missing)} required feature columns missing, including {missing[0]}.")
        x = data.loc[:, list(features)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        observed = np.isfinite(x)
        empty_rows = ~observed.any(axis=1)
        if empty_rows.any():
            raise ValueError(f"No finite measurements on retained features for {ids[empty_rows].iloc[0]}.")
        filled = np.where(observed, x, model["median"])
        z = np.clip((filled - model["mean"]) / model["scale"], -5, 5)
        classes = model["classes"]
        method = model["method"].item()
        if method == "channel_balanced":
            blocks, prototypes = model["blocks"], model["prototypes"]
            distinct_blocks = np.unique(blocks)
            scores = np.zeros((len(z), len(classes)), dtype=np.float64)
            for block in distinct_blocks:
                cols = np.flatnonzero(blocks == block)
                block_z = z[:, cols]
                norm = np.linalg.norm(block_z, axis=1, keepdims=True)
                unit = np.divide(block_z, norm, out=np.zeros_like(block_z), where=norm > 0)
                scores += unit @ prototypes[:, cols].T
            scores /= len(distinct_blocks)
        elif method == "ridge_balanced":
            scores = z @ model["coef"].T + model["intercept"]
        else:
            raise ValueError(f"Unknown model method: {method!r}")
    order = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    result = pd.DataFrame({"compound_id": ids})
    if "compound_name" in data:
        result["compound_name"] = data["compound_name"]
    for rank, label_ix in enumerate(order.T, start=1):
        result[f"top{rank}"] = classes[label_ix]
        result[f"score{rank}"] = scores[np.arange(len(scores)), label_ix]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Fitted channel or ridge .npz file")
    parser.add_argument("--query", required=True, type=Path, help="New profiles, CSV/TSV(.gz) or Parquet")
    parser.add_argument("--out", required=True, type=Path, help="Prediction CSV or CSV.GZ")
    args = parser.parse_args()
    if args.out.resolve() in (args.model.resolve(), args.query.resolve()):
        parser.error("Choose an output path different from the inputs.")
    try:
        result = predict(args.model, args.query)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, float_format="%.10g")


if __name__ == "__main__":
    main()
