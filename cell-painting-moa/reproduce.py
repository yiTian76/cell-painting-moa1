"""Run all experiments from the included Cell Painting tables, without network or an API."""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def call(script, *args):
    subprocess.run([sys.executable, str(ROOT / "src" / script), *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "rerun")
    parser.add_argument("--font", help="Optional Chinese font file for chart labels")
    args = parser.parse_args()
    out = args.out.resolve()
    source = ROOT / "results"
    if out in (ROOT, source, DATA, ROOT / "src", ROOT / "demo", ROOT / "figures"):
        parser.error("Choose a separate output directory for the rerun.")
    ten = DATA / "profiles_10uM.csv.gz"
    folds = out / "folds.csv"
    call("run_experiment.py", "--input", ten, "--out", out)
    call("supplement.py", "--data", ten, "--lower-dose", DATA / "profiles_3p33uM.csv.gz",
         "--folds", folds, "--out", out)
    call("cross_batch.py", "evaluate", "--primary", ten, "--external", DATA / "profiles_batch2_10uM.csv.gz",
         "--folds", folds, "--out", out)
    call("layout_control.py", "--profiles", ten, "--folds", folds, "--output", out)
    call("holdout_layouts.py", "--profiles", ten, "--folds", folds, "--model-script",
         ROOT / "src" / "run_experiment.py", "--output", out)
    call("predict_new.py", "--train", ten, "--query", ROOT / "demo" / "new_profiles.csv.gz",
         "--out", out / "demo_predictions.csv")
    call("make_figures.py", "--data", ten, "--results", out, "--out", out / "figures",
         *(["--font",args.font] if args.font else []))
    expected = pd.read_csv(source / "folds.csv")
    pd.testing.assert_frame_equal(pd.read_csv(folds), expected)
    for fname in ("metrics.csv", "all_model_metrics.csv", "ablation.csv", "batch2_metrics.csv", "supplement_metrics.csv",
                  "layout_metrics.csv", "leave_layout_out_metrics.csv", "bootstrap_ci.csv",
                  "permutation_scores.csv"):
        actual = pd.read_csv(out / fname)
        ref = pd.read_csv(source / fname)
        pd.testing.assert_frame_equal(actual, ref, check_exact=False, atol=1e-8, rtol=1e-8)
    demo = pd.read_csv(out / "demo_predictions.csv")
    pd.testing.assert_frame_equal(demo, pd.read_csv(ROOT / "demo" / "predictions.csv"),
                                  check_exact=False, atol=1e-8, rtol=1e-8)
    print(f"Verified drug folds, nine result tables and two demo predictions. Outputs: {out}")


if __name__ == "__main__":
    main()
