"""Fit the published models to the full training cohort and export their parameters."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from run_experiment import assign_block, fingerprint, fit_transform_fold, read_profiles, score_blocks


def export_models(train_path, folds_path, out):
    options = argparse.Namespace(input=train_path, features=None, min_compounds=5)
    names, x, feature_names, _, _ = read_profiles(options)
    folds = pd.read_csv(folds_path, dtype={"compound_id": "string"})
    if folds.compound_id.duplicated().any() or set(folds.compound_id) != set(names.index):
        raise ValueError("The eligible compound IDs do not match the cross-validation cohort.")
    manifest_labels = folds.set_index("compound_id").loc[names.index, "moa"]
    if not np.array_equal(names.moa.to_numpy(), manifest_labels.to_numpy()):
        raise ValueError("A training label differs from the cross-validation manifest.")

    classes, codes = np.unique(names.moa.to_numpy(dtype=str), return_inverse=True)
    train_z, _, keep = fit_transform_fold(x, x[:0])
    features = feature_names[keep].astype(str)
    retained = np.where(np.isfinite(x[:, keep]), x[:, keep], np.nan)
    median = np.nanmedian(retained, axis=0)
    filled = np.where(np.isfinite(retained), retained, median)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    np.testing.assert_allclose(np.clip((filled - mean) / scale, -5.0, 5.0), train_z,
                               rtol=1e-12, atol=1e-12)
    shared = dict(schema_version=np.int64(1), features=features, median=median, mean=mean,
                  scale=scale, classes=classes, training_ids=names.index.to_numpy(dtype=str),
                  train_sha256=fingerprint(train_path), folds_sha256=fingerprint(folds_path))

    blocks = np.array([assign_block(feature) for feature in features])
    _, _, _, evidence = score_blocks(train_z, train_z[:0], codes, blocks, len(classes))
    prototypes = np.empty((len(classes), train_z.shape[1]), dtype=np.float64)
    for positions, _, block_prototypes in evidence.values():
        prototypes[:, positions] = block_prototypes

    ridge = RidgeClassifier(alpha=100.0, class_weight="balanced", solver="cholesky")
    ridge.fit(train_z, codes)
    coef, intercept = ridge.coef_, ridge.intercept_
    if len(classes) == 2:
        # sklearn stores one signed decision score for a binary model.
        coef = np.vstack((-coef, coef))
        intercept = np.r_[-intercept, intercept]
    models = {
        "channel_balanced": dict(blocks=blocks, prototypes=prototypes),
        "ridge_balanced": dict(coef=coef, intercept=intercept),
    }
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as directory:
        stage = Path(directory)
        for method, parameters in models.items():
            np.savez_compressed(stage / f"{method}.npz", method=method, **shared, **parameters)
        for method in models:
            name = f"{method}.npz"
            (stage / name).replace(out / name)
    print(f"Saved {len(models)} models fitted on {len(names)} compounds, {len(classes)} classes, "
          f"{len(features)} retained features and {len(evidence)} blocks to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        export_models(args.train, args.folds, args.out)
