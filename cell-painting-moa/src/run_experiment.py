"""Compound-held-out, channel-balanced Cell Painting mechanism classification.

Input is a wide CSV/CSV.GZ or Parquet with compound_id, moa, optional
compound_name, and CellProfiler feature names. A plain-text --features list can
explicitly identify columns. All reported model scores are out-of-fold.
"""

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits


MAIN = "channel_balanced"
METHODS = [MAIN, "flat_mean_cosine", "cosine_1nn", "majority"]
CHANNELS = ("DNA", "ER", "RNA", "AGP", "Mito")
CHANNEL_ALIASES = {
    "DNA": {"dna", "hoechst", "hoechst33342", "dapi"},
    "ER": {"er", "concanavalina", "cona"},
    "RNA": {"rna", "syto14", "syto"},
    "AGP": {"agp", "wga", "phalloidin"},
    "Mito": {"mito", "mitotracker", "mitochondria"},
}
CP_NAME = re.compile(r"(?:^|_)(Cells|Cytoplasm|Nuclei|Image)_", re.I)
NON_MORPHOLOGY = re.compile(
    r"(?:^|_)(Location|Parent|ObjectNumber|ExecutionTime|"
    r"Number_Object_Number|AreaShape_Center_[XYZ])(?:_|$)", re.I
)


def assign_block(name):
    """Channel tokens refer to stains. A Nuclei object alone does not imply DNA."""
    tokens = {re.sub(r"^orig", "", t.lower()) for t in re.split(r"[^A-Za-z0-9]+", name)}
    hits = [key for key, aliases in CHANNEL_ALIASES.items() if tokens & aliases]
    if len(hits) == 1:
        return hits[0]
    return "cross_channel" if hits else "shape_other"


def fit_transform_fold(train, query):
    """Return train/query transformed by train-only statistics and feature mask."""
    train = np.where(np.isfinite(train), train, np.nan)
    query = np.where(np.isfinite(query), query, np.nan)
    keep = np.isfinite(train).mean(axis=0) >= 0.8
    if not keep.any():
        raise ValueError("No features meet the training-fold availability criterion.")
    observed = train[:, keep]
    median = np.nanmedian(observed, axis=0)
    filled = np.where(np.isfinite(observed), observed, median)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    variable = scale > 1e-8
    if not variable.any():
        raise ValueError("All available training features are constant.")
    q = query[:, keep]
    q = np.where(np.isfinite(q), q, median)
    keep[np.flatnonzero(keep)[~variable]] = False
    def transform(a):
        z = (a[:, variable] - mean[variable]) / scale[variable]
        return np.asarray(np.clip(z, -5, 5), dtype=np.float64)
    return transform(filled), transform(q), keep


def unit_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def class_means(x, labels, n_classes):
    return np.stack([x[labels == k].mean(axis=0) for k in range(n_classes)])


def score_blocks(train_z, query_z, y_train, blocks, n_classes, return_kernel=False):
    """Return class scores, block scores, optional mean kernel, and evidence vectors.

    Main score is mean cosine similarity to all training compounds of each
    class, calculated separately inside each block and averaged over blocks.
    """
    blocks = np.asarray(blocks)
    by_block, evidence = {}, {}
    kernel = np.zeros((len(query_z), len(train_z))) if return_kernel else None
    for block in sorted(set(blocks)):
        cols = np.flatnonzero(blocks == block)
        train_unit = unit_rows(train_z[:, cols])
        query_unit = unit_rows(query_z[:, cols])
        prototypes = class_means(train_unit, y_train, n_classes)
        by_block[block] = query_unit @ prototypes.T
        if return_kernel:
            kernel += query_unit @ train_unit.T
        evidence[block] = (cols, query_unit, prototypes)
    if return_kernel:
        kernel /= len(by_block)
    return sum(by_block.values()) / len(by_block), by_block, kernel, evidence


def confusion(y, pred, k):
    return np.bincount(y * k + pred, minlength=k*k).reshape(k, k)


def confusion_metrics(cm):
    tp = cm.diagonal()
    actual, predicted = cm.sum(axis=1), cm.sum(axis=0)
    f1 = np.divide(2 * tp, actual + predicted, out=np.zeros(len(tp)), where=(actual+predicted)>0)
    recall = np.divide(tp, actual, out=np.zeros(len(tp)), where=actual>0)
    return [tp.sum()/cm.sum(), recall.mean(), f1.mean()]


def stable_ranking(scores):
    # Columns are in alphabetic class order, providing a reproducible tie break.
    return np.argsort(-scores, axis=1, kind="stable")


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_profiles(args):
    if args.input.suffix.lower() == ".parquet":
        raw = pd.read_parquet(args.input)
    else:
        raw = pd.read_csv(args.input, low_memory=False, dtype={"compound_id": "string"})
    if args.features:
        features = [s.strip() for s in args.features.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    else:
        features = [s for s in raw.columns if CP_NAME.search(s) and not s.lower().startswith("metadata")]
    features = [s for s in features if not NON_MORPHOLOGY.search(s)]
    if len(set(features)) != len(features) or not features:
        raise ValueError("Need unique morphology feature names (supply --features for other naming schemes).")
    frame = raw.copy()
    for col in ["moa", "compound_id"]:
        frame[col] = frame[col].astype("string").str.strip().replace("", pd.NA)
    frame = frame.dropna(subset=["moa", "compound_id"])
    label_counts = frame.groupby("compound_id").moa.nunique()
    ambiguous = label_counts.index[label_counts > 1]
    frame = frame.loc[~frame.compound_id.isin(ambiguous)].copy()
    if "compound_name" not in frame:
        frame["compound_name"] = frame.compound_id
    frame["compound_name"] = frame.compound_name.fillna(frame.compound_id)
    names = frame.groupby("compound_id")[["compound_name", "moa"]].first()
    x = frame[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    x.index = frame.compound_id
    x = x.groupby(level=0, sort=True).median()
    no_signal = x.isna().all(axis=1)
    x = x.loc[~no_signal]
    names = names.loc[x.index]
    counts = names.moa.value_counts().sort_index().rename("n_compounds").to_frame()
    counts["eligible"] = counts.n_compounds >= args.min_compounds
    names = names.loc[names.moa.isin(counts.index[counts.eligible])]
    if names.moa.nunique() < 2:
        raise ValueError("Fewer than two mechanisms qualify at the requested minimum.")
    qc = dict(input_rows=len(raw), ambiguous_compound_ids=len(ambiguous),
              all_missing_compounds=int(no_signal.sum()), compounds_before_class_filter=len(x),
              classes_before_filter=len(counts), eligible_compounds=len(names),
              eligible_classes=names.moa.nunique(), morphology_features=len(features))
    return names, x.loc[names.index].to_numpy(dtype=np.float64), np.array(features), counts, qc


def cross_validate(x, y, folds, feature_names, blocks, labels):
    n, k = len(y), len(labels)
    scores = {m: np.empty((n, k)) for m in METHODS}
    drops = {b: np.full((n, k), np.nan) for b in np.unique(blocks)}
    kernels, fold_features, test_margins = [], [], []
    feature_sum = np.zeros((k, len(feature_names)))
    z_sum = np.zeros_like(feature_sum)
    z_count = np.zeros_like(feature_sum)
    for fold in np.unique(folds):
        tr, te = np.where(folds != fold)[0], np.where(folds == fold)[0]
        tr_z, te_z, keep = fit_transform_fold(x[tr], x[te])
        fold_blocks = blocks[keep]
        main, by_block, kernel, evidence = score_blocks(
            tr_z, te_z, y[tr], fold_blocks, k, return_kernel=True)
        scores[MAIN][te] = main
        kernels.append((tr, te, kernel))
        flat = unit_rows(te_z) @ unit_rows(tr_z).T
        for c in range(k):
            flat_class = flat[:, y[tr] == c]
            scores["flat_mean_cosine"][te, c] = flat_class.mean(axis=1)
            scores["cosine_1nn"][te, c] = flat_class.max(axis=1)
            scores["majority"][te, c] = flat_class.shape[1] / len(tr)
        rank = stable_ranking(main)
        truth = y[te]
        rival = np.where(rank[:, 0] == truth, rank[:, 1], rank[:, 0])
        n_blocks = len(by_block)
        local_rows = np.arange(len(te))
        for block in drops:
            fold_features.append(dict(fold=int(fold), block=block,
                                      retained_features=int(np.sum(fold_blocks == block))))
            # When a block is absent after train-only filtering, its removal is a no-op.
            if block not in by_block:
                drops[block][te] = main
                continue
            if n_blocks > 1:
                drops[block][te] = (n_blocks * main - by_block[block]) / (n_blocks - 1)
            cols, q, proto = evidence[block]
            orig_cols = np.flatnonzero(keep)[cols]
            contribution = q * (proto[truth] - proto[rival]) / n_blocks
            for c in np.unique(truth):
                member = truth == c
                feature_sum[c, orig_cols] += contribution[member].sum(axis=0)
                z_sum[c, orig_cols] += te_z[member][:, cols].sum(axis=0)
                z_count[c, orig_cols] += member.sum()
            margins = (by_block[block][local_rows, truth] -
                       by_block[block][local_rows, rival]) / n_blocks
            for i, v in zip(te, margins):
                test_margins.append((i, block, v))
    return scores, drops, kernels, fold_features, test_margins, feature_sum, z_sum, z_count


def evaluate(scores, y, labels):
    k = len(labels)
    rankings = {name: stable_ranking(s) for name, s in scores.items()}
    class_rows, summary, cm_main = [], [], None
    by_method = {}
    for name, rank in rankings.items():
        pred = rank[:, 0]
        cm = confusion(y, pred, k)
        hit3 = np.any(rank[:, :min(3, k)] == y[:, None], axis=1)
        by_method[name] = (pred, hit3)
        summary.append([name] + confusion_metrics(cm) + [hit3.mean()])
        for c, actual in enumerate(labels):
            tp = cm[c, c]
            support, n_pred = cm[c].sum(), cm[:, c].sum()
            class_rows.append([name, actual, support, n_pred, tp,
                               tp/n_pred if n_pred else 0.0, tp/support,
                               2*tp/(support+n_pred), hit3[y == c].mean()])
        if name == MAIN:
            cm_main = cm
    return rankings, by_method, summary, class_rows, cm_main


def bootstrap(y, by_method, k, seed, n_draws):
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(y == c) for c in range(k)]
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy"]
    draws = {m: np.zeros((n_draws, 4)) for m in by_method}
    for i in range(n_draws):
        idx = np.concatenate([rng.choice(v, len(v), replace=True) for v in members])
        for name, (pred, hit3) in by_method.items():
            draws[name][i] = confusion_metrics(confusion(y[idx], pred[idx], k)) + [hit3[idx].mean()]
    rows = []
    for name in draws:
        for j, metric in enumerate(metric_names):
            lo, hi = np.quantile(draws[name][:, j], [.025, .975])
            rows.append([name, metric, lo, hi])
    return rows


def permutation_test(y, folds, kernels, k, seed, n_draws):
    """Condition on each fold's class counts. Refit all class prototypes per draw."""
    rng = np.random.default_rng(seed)
    member = [np.flatnonzero(folds == f) for f in np.unique(folds)]
    rows = []
    for i in range(n_draws):
        perm = y.copy()
        for subset in member:
            perm[subset] = rng.permutation(y[subset])
        pred = np.zeros_like(y)
        for tr, te, kernel in kernels:
            train_labels = perm[tr]
            order = np.argsort(train_labels, kind="stable")
            counts = np.bincount(train_labels, minlength=k)
            starts = np.r_[0, np.cumsum(counts[:-1])]
            class_sum = np.add.reduceat(kernel[:, order], starts, axis=1)
            pred[te] = np.argmax(class_sum/counts, axis=1)
        rows.append([i+1] + confusion_metrics(confusion(perm, pred, k)))
    return rows


def csv(out, name, rows, cols):
    pd.DataFrame(rows, columns=cols).to_csv(out / name, index=False, float_format="%.10g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--features", type=Path, help="Optional newline-delimited list of morphology feature columns")
    parser.add_argument("--min-compounds", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if min(args.bootstrap, args.permutations) < 1 or min(args.min_compounds, args.folds) < 2:
        parser.error("Need >=1 permutation/bootstrap draw, >=2 compounds and >=2 folds.")
    start = time.perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    names, x, feats, counts, qc = read_profiles(args)
    counts.rename_axis("moa").to_csv(out/"class_eligibility.csv")
    labels, y = np.unique(names.moa.to_numpy(dtype=str), return_inverse=True)
    k = len(labels)
    n_folds = min(args.folds, np.bincount(y).min())
    folds = np.zeros(len(y), dtype=int)
    for f, (_, te) in enumerate(StratifiedKFold(n_folds, shuffle=True,
                                                random_state=args.seed).split(x, y), start=1):
        folds[te] = f
    assert np.all(folds >= 1) and names.index.is_unique
    names = names.assign(fold=folds)
    names.to_csv(out/"folds.csv")
    blocks = np.array([assign_block(col) for col in feats])
    csv(out, "feature_blocks.csv", zip(feats, blocks), ["feature", "block"])
    with threadpool_limits(limits=1):
        result = cross_validate(x, y, folds, feats, blocks, labels)
    scores, drops, kernels, fold_features, margins, fsum, zsum, zcount = result
    rank, by_method, summary, class_rows, cm = evaluate(scores, y, labels)
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy"]
    csv(out, "metrics.csv", summary, ["method"] + metrics)
    csv(out, "per_class.csv", class_rows,
        ["method", "moa", "support", "n_predicted", "n_correct", "precision", "recall", "f1", "top3_accuracy"])
    pd.DataFrame(cm, index=labels, columns=labels).rename_axis("true_moa").to_csv(out/"confusion_main.csv")
    preds = names.copy()
    for name in METHODS:
        preds["pred_"+name] = labels[rank[name][:,0]]
    for r in range(min(3,k)):
        preds["top"+str(r+1)] = labels[rank[MAIN][:,r]]
        preds["score"+str(r+1)] = scores[MAIN][np.arange(len(y)),rank[MAIN][:,r]]
    preds.to_csv(out/"oof_predictions.csv", float_format="%.10g")
    pd.DataFrame(scores[MAIN], index=names.index, columns=labels).to_csv(out/"scores_main.csv.gz", float_format="%.10g")
    pd.DataFrame(fold_features).to_csv(out/"fold_features.csv", index=False)
    summary_main = summary[METHODS.index(MAIN)]
    ablation = []
    for block, s in drops.items():
        if not np.isfinite(s).all():
            continue
        r = stable_ranking(s)
        m = confusion_metrics(confusion(y, r[:,0], k))
        ablation.append([block, int(sum(blocks == block))] + m +
                        [np.any(r[:,:min(k,3)] == y[:,None], axis=1).mean(), summary_main[3]-m[2]])
    csv(out, "ablation.csv", ablation, ["omitted_block", "features_input"] + metrics + ["macro_f1_full_minus_omitted"])
    evidence = []
    support = np.bincount(y)
    for c in range(k):
        score = fsum[c]/support[c]
        avg_z = np.divide(zsum[c], zcount[c], out=np.full(len(feats), np.nan), where=zcount[c]>0)
        for j in np.argsort(-score, kind="stable")[:10]:
            if score[j] > 0:
                evidence.append([labels[c], support[c], feats[j], blocks[j], score[j], avg_z[j], int(zcount[c,j])])
    csv(out, "feature_evidence.csv", evidence,
        ["moa", "support", "feature", "block", "mean_true_minus_rival_contribution", "mean_held_out_z", "n_with_feature_retained"])
    bycompound = pd.DataFrame(margins, columns=["row","block","true_minus_rival_contribution"])
    bycompound["compound_id"] = names.index.to_numpy()[bycompound.row]
    bycompound["moa"] = labels[y[bycompound.row]]
    bycompound.drop(columns="row").to_csv(out/"channel_evidence.csv.gz", index=False, float_format="%.10g")
    bycompound.groupby(["moa","block"]).true_minus_rival_contribution.mean().to_csv(out/"channel_evidence_by_class.csv")
    fold_ms = []
    for f in np.unique(folds):
        use = folds == f
        for name, (pred, hit3) in by_method.items():
            fold_ms.append([f,name,int(sum(use))]+confusion_metrics(confusion(y[use],pred[use],k))+[hit3[use].mean()])
    csv(out, "metrics_by_fold.csv", fold_ms, ["fold","method","n_compounds"] + metrics)
    cis = bootstrap(y, by_method, k, args.seed+1, args.bootstrap)
    csv(out, "bootstrap_ci.csv", cis, ["method","metric","lower_95","upper_95"])
    null = permutation_test(y, folds, kernels, k, args.seed+2, args.permutations)
    csv(out, "permutation_scores.csv", null, ["permutation"]+metrics[:3])
    null_f1 = np.array(null)[:,3]
    p = (1 + np.count_nonzero(null_f1 >= summary_main[3]-1e-12)) / (1 + len(null_f1))
    log = dict(input_file=args.input.name, input_sha256=fingerprint(args.input), seed=args.seed,
               folds=int(n_folds), min_compounds=args.min_compounds, **qc,
               classes=labels.tolist(), features_by_block=pd.Series(blocks).value_counts().to_dict(),
               main_method=MAIN, bootstrap_draws=args.bootstrap, permutation_draws=args.permutations,
               permutation_macro_f1_mean=float(null_f1.mean()),
               permutation_macro_f1_q025=float(np.quantile(null_f1,.025)),
               permutation_macro_f1_q975=float(np.quantile(null_f1,.975)),
               permutation_macro_f1_p=float(p),
               runtime_seconds=round(time.perf_counter()-start,3),
               versions={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                         "sklearn":sklearn.__version__,"scipy":scipy.__version__})
    encoded = json.dumps(log, ensure_ascii=False, indent=2)
    (out/"run.json").write_text(encoded,encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
