"""inference.py - generate submission.csv from trained fold models.

Example:
    python inference.py --data_dir /path/to/data --weights_dir ./outputs --out submission.csv
weights_dir must hold pipeline_config.json and model_<run>_fold<k>.pt written by train.py
(download them from the model-weights link in the README).
"""
import argparse
import numpy as np
from common import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True); p.add_argument("--weights_dir", default="./outputs")
    p.add_argument("--out", default="submission.csv"); p.add_argument("--work_dir", default="/tmp/pareidolia_work")
    p.add_argument("--n_folds", type=int, default=5)
    a = p.parse_args()

    cfg = load_config(a.weights_dir)
    _, _, test_meta, raw_te = load_data(a.data_dir, a.work_dir, need_train=False)
    az = test_meta["sun_azimuth_angle"].values.astype(np.float64)
    X = build_inputs(raw_te, az, cfg["mode"], cfg["phi0_deg"])           # identical physics normalisation as training
    probs, used = ensemble_predict(a.weights_dir, cfg["runs"], X, cfg["use_vflip"], n_folds=a.n_folds)
    sub = make_submission(test_meta, probs[:, 1], cfg.get("threshold", 0.5), a.out)
    print(f"Ensembled {len(used)} models | mode={cfg['mode']} | threshold={cfg.get('threshold', 0.5):.3f}")
    print("Predicted class share:", sub["label"].value_counts(normalize=True).sort_index().round(3).to_dict())
    print("Saved", a.out)


if __name__ == "__main__":
    main()
