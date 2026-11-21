"""Additional evaluations. Place beside run_experiment.py for standalone use."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from run_experiment import assign_block, fit_transform_fold, score_blocks


READOUTS = [
    "Nuclei_AreaShape_Area",
    "Cells_AreaShape_Area",
    "Nuclei_Intensity_MeanIntensity_DNA",
    "Cytoplasm_Intensity_MeanIntensity_Mito",
    "Cytoplasm_Intensity_MeanIntensity_ER",
    "Cytoplasm_Intensity_MeanIntensity_AGP",
]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lower-dose", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read(path):
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return df.set_index("compound_id", verify_integrity=True)


def collect(test, scores, classes, model, fold):
    ranking = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    preds = pd.DataFrame(classes[ranking], index=test.index,
                         columns=["prediction", "second", "third"])
    return test[["moa"]].join(preds).assign(model=model, fold=fold)


def run():
    args = get_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(args.folds).set_index("compound_id", verify_integrity=True)
    full = read(args.data).loc[folds.index]
    lower = read(args.lower_dose)
    assert full.moa.equals(folds.moa)
    classes = np.sort(full.moa.unique())
    features = [c for c in full if c.startswith(("Nuclei_", "Cells_", "Cytoplasm_"))]
    assert set(features) <= set(lower)
    blocks = np.array([assign_block(c) for c in features])
    predictions = []
    for fold in sorted(folds.fold.unique()):
        train = full.loc[folds.fold != fold]
        test = full.loc[folds.fold == fold]
        matched = test.index.intersection(lower.index)
        test_low = lower.loc[matched]
        assert test_low.moa.equals(test.loc[matched, "moa"])
        evaluation = pd.concat([test[features], test_low[features]]).to_numpy(float)
        ztrain, zeval, keep = fit_transform_fold(train[features].to_numpy(float), evaluation)
        y_codes = np.searchsorted(classes, train.moa)
        ridge = RidgeClassifier(alpha=100., class_weight="balanced", solver="cholesky")
        ridge.fit(ztrain, y_codes)
        scores = ridge.decision_function(zeval[:len(test)])
        predictions.append(collect(test, scores, classes, "ridge_balanced_10uM", fold))
        if len(test_low):
            low_scores = score_blocks(ztrain, zeval[len(test):], y_codes, blocks[keep], len(classes))[0]
            high_scores = score_blocks(ztrain, zeval[:len(test)], y_codes, blocks[keep], len(classes))[0]
            paired_high = high_scores[test.index.get_indexer(matched)]
            predictions.append(collect(test_low, low_scores, classes, "main_3.3333uM", fold))
            predictions.append(collect(test.loc[matched], paired_high, classes, "main_10uM_paired", fold))
    oof = pd.concat(predictions).reset_index()
    oof.to_csv(args.out / "supplement_predictions.csv", index=False)
    metrics = []
    for model, group in oof.groupby("model"):
        y = group.moa
        pred = group.prediction
        hit3 = group[["prediction", "second", "third"]].eq(y, axis=0).any(axis=1)
        metrics.append(dict(model=model, compounds=len(group), classes=y.nunique(),
                            accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred,
                            labels=classes, average="macro", zero_division=0),
                            balanced_accuracy=balanced_accuracy_score(y, pred),
                            top3_accuracy=hit3.mean()))
    summary = pd.DataFrame(metrics)
    summary.to_csv(args.out / "supplement_metrics.csv", index=False)
    main_table = pd.read_csv(args.folds.parent / "metrics.csv")
    ridge_row = summary.loc[summary.model == "ridge_balanced_10uM"].rename(columns={"model":"method"})
    pd.concat([main_table, ridge_row[main_table.columns]], ignore_index=True).to_csv(
        args.out / "all_model_metrics.csv", index=False)
    # Descriptive readouts in the published DMSO-normalized units. These never enter model selection.
    summaries = []
    for label, group in full.groupby("moa"):
        for feature in READOUTS:
            if feature not in features:
                continue
            values = group[feature].dropna()
            summaries.append(dict(moa=label,feature=feature,n=len(values),
                median=values.median(),q25=values.quantile(.25),q75=values.quantile(.75),
                rest_median=full.loc[full.moa != label,feature].median()))
    pd.DataFrame(summaries).to_csv(args.out / "class_readouts.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
