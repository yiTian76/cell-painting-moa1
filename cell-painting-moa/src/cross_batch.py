"""Prepare and evaluate held-out LINCS batch-2 Cell Painting profiles.

Place alongside prepare_data.py and run_experiment.py. Preparation downloads the
pinned public batch and compound catalogs when missing. Evaluation uses only the
processed files. All outcomes are generated without adapting a model to batch 2.
"""
import argparse
import csv
import gzip
import json
import re
import shutil
import tempfile
import urllib.request
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from prepare_data import MEDIA, RAW, REVISION, SOURCES, identity_reason, sha256, write_gzip
from run_experiment import assign_block, fit_transform_fold, score_blocks, stable_ranking, unit_rows

BATCH = "2017_12_05_Batch2"
EXTERNAL_SOURCES = {
    "batch2.csv.gz": (f"{MEDIA}/consensus/{BATCH}/{BATCH}_consensus_modz_dmso.csv.gz",
                      "c42146a5b2cdba919358f212291c71d25f59e804308e87b2a53aafe3dd22e617"),
    "moa.tsv": (f"{RAW}/metadata/moa/repurposing_info.tsv",
                "aa8d8f4e2a175ed3c6c0c97e18699f62eda7caf9fed445590f364d278e01dd2d"),
    "resolved.tsv": SOURCES["resolved.tsv"],
}


def fetch(cache):
    cache.mkdir(exist_ok=True, parents=True)
    for name, (url, digest) in EXTERNAL_SOURCES.items():
        path = cache / name
        if not path.is_file():
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                tmp = Path(stream.name)
                try:
                    with urllib.request.urlopen(url, timeout=150) as source:
                        shutil.copyfileobj(source, stream)
                except Exception:
                    stream.close()
                    tmp.unlink(missing_ok=True)
                    raise
            if sha256(tmp) != digest:
                tmp.unlink()
                raise ValueError(f"Checksum mismatch for download: {url}")
            tmp.replace(path)
        elif sha256(path) != digest:
            raise ValueError(f"Checksum mismatch for cached file: {path}")


def read_catalog(path):
    with path.open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def catalog_index(cache):
    by_id, by_parent, by_key = (defaultdict(list) for _ in range(3))
    for r in read_catalog(cache / "moa.tsv"):
        r["alternative_moa"] = ""
        sample = r["broad_id"]
        by_id[sample].append(r)
        by_parent["-".join(sample.split("-")[:2])].append(r)
        by_key[r["InChIKey14"]].append(r)
    for r in read_catalog(cache / "resolved.tsv"):
        by_id[r["broad_sample"]].append(r)
        by_parent[r["broad_id"]].append(r)
        by_key[r["InChIKey14"]].append(r)
    status = {k: identity_reason(rows) for k, rows in by_key.items()}
    return by_id, by_parent, by_key, status


def annotate(sample, profile_moa, catalog):
    by_id, by_parent, by_key, status = catalog
    rows = by_id.get(sample)
    match = "exact_material"
    if not rows:
        rows = by_parent.get("-".join(sample.split("-")[:2]), [])
        match = "unambiguous_broad_parent"
    keys = {r["InChIKey14"] for r in rows}
    if len(keys) != 1:
        return None, "absent_or_ambiguous_identity"
    key = next(iter(keys))
    if not re.fullmatch(r"[A-Z]{14}", key):
        return None, "missing_connectivity"
    if status[key] != "retained":
        return None, status[key]
    if any(r.get("qc_incompatible") == "1" for r in by_id.get(sample, [])):
        return None, "qc_incompatible_material"
    moa = by_key[key][0]["moa"].strip()
    if profile_moa.strip().lower() not in {"", "unknown", moa.casefold()}:
        return None, "profile_annotation_disagreement"
    names = sorted({r["pert_iname"].strip() for r in rows} - {""})
    return (key, "|".join(names), moa, match), "retained"


def prepare(args):
    fetch(args.cache)
    args.out.mkdir(parents=True, exist_ok=True)
    features = [col for col in pd.read_csv(args.primary, nrows=0).columns
                if col.startswith(("Cells_", "Nuclei_", "Cytoplasm_"))]
    catalog = catalog_index(args.cache)
    counter, groups, flow = Counter(), defaultdict(list), []
    with gzip.open(args.cache / "batch2.csv.gz", "rt", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        ix = {f: i for i, f in enumerate(header)}
        feature_idx = [ix[f] for f in features]
        for row in reader:
            counter["all_rows"] += 1
            if row[ix["Metadata_cell_id"]] != "A549" or row[ix["Metadata_time_point"]] != "48H":
                continue
            counter["A549_48H"] += 1
            if row[ix["Metadata_dose_recode"]] == "6":
                counter["nominal_dose6"] += 1
            if float(row[ix["Metadata_mmoles_per_liter"]]) != 10.0:
                continue
            counter["exact_10uM"] += 1
            sample = row[ix["Metadata_broad_sample"]]
            raw_label = row[ix["Metadata_moa"]]
            annotation, reason = annotate(sample, raw_label, catalog)
            flow.append([sample, raw_label, reason, *(annotation or ("", "", "", ""))])
            if annotation is None:
                continue
            key, name, moa, match = annotation
            values = np.array([float(row[i]) if row[i].strip() else np.nan for i in feature_idx])
            values[~np.isfinite(values)] = np.nan
            groups[key].append((sample, name, moa, values))
    records = []
    for key, group in sorted(groups.items()):
        samples, names, labels, vectors = zip(*group)
        if len(set(labels)) != 1:
            raise ValueError(f"Conflicting batch-2 label for {key}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            profile = np.nanmedian(vectors, axis=0)
        records.append([key, "|".join(sorted(set(names))), labels[0], "|".join(sorted(set(samples))),
                        len(group), *profile])
    metadata = ["compound_id", "compound_name", "moa", "Metadata_broad_samples", "Metadata_num_profiles"]
    prepared = pd.DataFrame(records, columns=metadata + features)
    out_profile = args.out / "profiles_batch2_10uM.csv.gz"
    with write_gzip(out_profile) as writer:
        prepared.to_csv(writer, index=False, lineterminator="\n")
    audit = pd.DataFrame(flow, columns=["material", "profile_moa", "decision", "key", "name", "moa", "match"])
    audit.to_csv(args.out / "batch2_annotation_flow.csv", index=False)
    manifest = dict(revision=REVISION, source_license="CC0 1.0", sources=EXTERNAL_SOURCES,
                    filters=dict(counter), retained_features=len(features), output_compounds=len(prepared),
                    source_row_flow=audit.decision.value_counts().to_dict(),
                    sha256=sha256(out_profile))
    (args.out / "batch2_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(manifest, indent=2))


def evaluate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(args.folds).set_index("compound_id", verify_integrity=True)
    primary = pd.read_csv(args.primary).set_index("compound_id", verify_integrity=True).loc[folds.index]
    if not primary.moa.equals(folds.moa):
        raise ValueError("Primary labels do not match folds.csv")
    classes = np.sort(primary.moa.unique())
    ext = pd.read_csv(args.external).set_index("compound_id", verify_integrity=True)
    ext = ext.loc[ext.moa.isin(classes)].copy()
    # Avoid a name-alias crossing the split, even if reported connectivity differs.
    names_to_keys = defaultdict(set)
    for key, name in primary.compound_name.fillna("").items():
        for alias in name.casefold().split("|"):
            if alias:
                names_to_keys[alias].add(key)
    for key, name in ext.compound_name.fillna("").items():
        if any(names_to_keys.get(n, set()) - {key} for n in name.casefold().split("|") if n):
            raise ValueError(f"Resolve alias with multiple connectivity keys: {key}")
    shared = ext.index.intersection(primary.index)
    if not ext.loc[shared, "moa"].equals(primary.loc[shared, "moa"]):
        raise ValueError("Conflicting mechanism across batches")
    features = [col for col in primary if col.startswith(("Cells_", "Nuclei_", "Cytoplasm_"))]
    blocks = np.array([assign_block(name) for name in features])
    ext["excluded_fold"] = folds.fold.reindex(ext.index).fillna(0).astype(int)
    rows = []
    for fold, test in ext.groupby("excluded_fold"):
        train = primary.loc[folds.fold != fold]
        if len(test.index.intersection(train.index)):
            raise ValueError("A test compound occurs in its own training set.")
        overlap = int(fold) != 0
        evaluation = test[features].to_numpy()
        if overlap:
            evaluation = np.concatenate([evaluation, primary.loc[test.index, features].to_numpy()])
        tr_z, q_z, keep = fit_transform_fold(train[features].to_numpy(), evaluation)
        y = np.searchsorted(classes, train.moa)
        scores = score_blocks(tr_z, q_z, y, blocks[keep], len(classes))[0]
        ranking = classes[stable_ranking(scores)[:, :3]]
        sims = unit_rows(q_z[:len(test)]) @ unit_rows(tr_z).T
        nn_scores = np.stack([sims[:, y == c].max(axis=1) for c in range(len(classes))], axis=1)
        nn = classes[stable_ranking(nn_scores)[:, :3]]
        for k, (key, record) in enumerate(test.iterrows()):
            rows.append(dict(compound_id=key, compound_name=record.compound_name, moa=record.moa,
                             comparison="shared_held_out_fold" if overlap else "absent_from_primary_model_cohort",
                             excluded_fold=int(fold), n_training=len(train),
                             prediction=ranking[k, 0], second=ranking[k, 1], third=ranking[k, 2],
                             nn1_prediction=nn[k, 0], nn1_second=nn[k, 1], nn1_third=nn[k, 2],
                             paired_primary_prediction=ranking[k + len(test), 0] if overlap else None))
    results = pd.DataFrame(rows).sort_values("compound_id")
    results.to_csv(args.out / "batch2_predictions.csv", index=False)
    summary = []
    for tag, group in [("all", results), *results.groupby("comparison")]:
        truth = group.moa
        paired = group.dropna(subset=["paired_primary_prediction"])
        hits = group[["prediction", "second", "third"]].eq(truth, axis=0).any(axis=1)
        nn_hits = group[["nn1_prediction", "nn1_second", "nn1_third"]].eq(truth, axis=0).any(axis=1)
        summary.append(dict(subset=tag, n=len(group), n_true_classes=truth.nunique(), n_candidates=len(classes),
                            n_correct=(group.prediction == truth).sum(), accuracy=(group.prediction == truth).mean(),
                            n_top3=hits.sum(), top3_accuracy=hits.mean(), paired_n=len(paired),
                            nn1_correct=(group.nn1_prediction == truth).sum(),
                            nn1_accuracy=(group.nn1_prediction == truth).mean(),
                            nn1_top3=nn_hits.sum(), nn1_top3_accuracy=nn_hits.mean(),
                            paired_primary_correct=(paired.paired_primary_prediction == paired.moa).sum(),
                            paired_batch2_correct=(paired.prediction == paired.moa).sum(),
                            paired_prediction_agreement=(paired.paired_primary_prediction == paired.prediction).sum()))
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out / "batch2_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("--cache", type=Path, default=Path("raw_data"))
    prep.add_argument("--primary", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    ev = modes.add_parser("evaluate")
    for key in ("primary", "external", "folds", "out"):
        ev.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.mode == "prepare":
            prepare(args)
        else:
            evaluate(args)
