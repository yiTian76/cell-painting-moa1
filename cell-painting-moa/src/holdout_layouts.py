"""Assess the fitted methods after withholding every profile from a plate map.

Maps spanned by a single merged compound form one connected component.
Each component serves as the test set once. Every compound receives one
prediction. No training compound shares a plate map with a test compound.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def linked_maps(members):
    adjacency = {}
    for labels in members:
        for label in labels:
            adjacency.setdefault(label, set()).update(labels)
    groups = {}
    for label in sorted(adjacency):
        if label in groups:
            continue
        stack = [label]
        while stack:
            item = stack.pop()
            if item in groups:
                continue
            groups[item] = label
            stack.extend(adjacency[item] - groups.keys())
    return groups


def run(args):
    spec = importlib.util.spec_from_file_location("main_experiment", args.model_script)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    fold = pd.read_csv(args.folds, dtype=str).set_index("compound_id")
    data = pd.read_csv(args.profiles, low_memory=False).set_index("compound_id")
    if not fold.index.is_unique or not data.index.is_unique:
        raise ValueError("Fold and profile tables must each contain a unique row for every compound.")
    data = data.loc[fold.index]
    if not np.array_equal(fold.moa.values, data.moa.values):
        raise ValueError("Primary fold IDs and true labels must agree.")
    features = [s for s in data if model.CP_NAME.search(s) and not s.lower().startswith("metadata")
                and not model.NON_MORPHOLOGY.search(s)]
    maps = [s.split("|") for s in data.Metadata_plate_maps]
    groups = linked_maps(maps)
    partition = np.array([groups[m[0]] for m in maps])
    y_names, y = np.unique(data.moa.values, return_inverse=True)
    blocks = np.array([model.assign_block(name) for name in features])
    x = data[features].to_numpy(dtype=float)
    outputs = []
    results = {}
    minimum_train_per_class = len(y)
    for group in np.unique(partition):
        tr = np.where(partition != group)[0]
        te = np.where(partition == group)[0]
        train_maps = {m for i in tr for m in maps[i]}
        test_maps = {m for i in te for m in maps[i]}
        if not train_maps.isdisjoint(test_maps):
            raise ValueError("A test map appears in the training set.")
        counts = np.bincount(y[tr], minlength=len(y_names))
        if counts.min() == 0:
            raise ValueError("A candidate class is missing in this holdout training set.")
        minimum_train_per_class = min(minimum_train_per_class, int(counts.min()))
        tr_z, te_z, keep = model.fit_transform_fold(x[tr], x[te])
        main, _, _, _ = model.score_blocks(tr_z, te_z, y[tr], blocks[keep], len(y_names))
        similarities = model.unit_rows(te_z) @ model.unit_rows(tr_z).T
        nearest = np.column_stack([similarities[:, y[tr] == c].max(axis=1) for c in range(len(y_names))])
        scores = {"channel_balanced": main, "cosine_1nn": nearest, "majority": np.tile(counts, (len(te), 1))}
        for method, values in scores.items():
            ranks = model.stable_ranking(values)[:, :3]
            if method not in results:
                results[method] = np.empty((len(y), 3), dtype=np.int64)
            results[method][te] = ranks
            for i, ranked in zip(te, ranks):
                outputs.append({"compound_id": data.index[i], "moa": y_names[y[i]], "held_out_maps": "|".join(sorted(test_maps)),
                                "method": method, "top1": y_names[ranked[0]], "top2": y_names[ranked[1]],
                                "top3": y_names[ranked[2]]})
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_csv(args.output / "leave_layout_out_predictions.csv", index=False)
    summaries = []
    for method, rank in results.items():
        measures = model.confusion_metrics(model.confusion(y, rank[:, 0], len(y_names)))
        summaries.append([method, *measures, (rank == y[:, None]).any(axis=1).mean()])
    header = ["method", "accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy"]
    df = pd.DataFrame(summaries, columns=header)
    df.to_csv(args.output / "leave_layout_out_metrics.csv", index=False)
    manifest = {
        "n_compounds": len(y), "n_classes": len(y_names), "maps": len(groups),
        "components_held_out": len(set(groups.values())), "minimum_training_members_any_class_any_split": minimum_train_per_class,
        "input_features": len(features), "component_by_plate_map": groups,
        "size_by_component": {k: int((partition == k).sum()) for k in sorted(set(partition))},
    }
    (args.output / "leave_layout_out_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--folds", required=True, type=Path)
    p.add_argument("--model-script", required=True, type=Path, help="run_experiment.py from this package")
    p.add_argument("--output", required=True, type=Path)
    with threadpool_limits(limits=1):
        run(p.parse_args())
