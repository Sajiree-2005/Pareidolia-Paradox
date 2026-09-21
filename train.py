"""train.py - train the Pareidolia Paradox classifier (physics probe -> mode selection -> stratified K-fold).

Example:
    python train.py --data_dir /path/to/data --out_dir ./outputs
data_dir must contain train_images.zip (or train_images/), test_images.zip (or test_images/), train_metadata.csv, test_metadata.csv.
"""
import argparse, os
import numpy as np
from sklearn.metrics import confusion_matrix
from common import *


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True); p.add_argument("--out_dir", default="./outputs"); p.add_argument("--work_dir", default="/tmp/pareidolia_work")
    p.add_argument("--mode", default="auto", choices=("auto",) + MODES, help="input representation; auto = pick with a quick pilot")
    p.add_argument("--backbone", default="tf_efficientnetv2_s.in21k_ft_in1k"); p.add_argument("--run_name", default="effv2s")
    p.add_argument("--img_size", type=int, default=320); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_folds", type=int, default=5); p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--pilot_epochs", type=int, default=5); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_resume", action="store_true"); p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    a = parse(); set_seed(a.seed); os.makedirs(a.out_dir, exist_ok=True)
    train_meta, raw_tr, test_meta, raw_te = load_data(a.data_dir, a.work_dir)
    y, az = train_meta["label"].values.astype(int), train_meta["sun_azimuth_angle"].values.astype(np.float64)
    print("train:", raw_tr.shape, "class counts:", np.bincount(y).tolist(), "| azimuth range:", az.min(), az.max())

    # 1) physics probe -------------------------------------------------------------------------------------
    cfg_path = os.path.join(a.out_dir, "pipeline_config.json")
    if not a.no_resume and os.path.exists(cfg_path):
        cfg = load_config(a.out_dir); print(f"Resuming with saved physics config: mode={cfg['mode']}")
        d = None
    else:
        d = discover_physics(raw_tr, az, y); print("\n".join(summarize_probe(d)))
        phi0 = d["phi0_deg"]
        # 2) mode selection ----------------------------------------------------------------------------------
        if a.mode == "auto":
            res = pilot(raw_tr, az, y, phi0_deg=phi0, epochs=a.pilot_epochs, num_workers=a.num_workers)
            print("\nPILOT RESULTS (val balanced accuracy):", {k: round(v, 4) for k, v in res.items()})
            mode = max(res, key=lambda k: (round(res[k], 3), k == "spec"))     # ties -> prefer the competition rule
        else:
            mode, res = a.mode, {}
        strong = d["R_routed"] > 0.3 and d["R_routed"] > 3 * d["R_routed_null"] if mode == "routed" else \
                 d["fit"][{"spec": 1, "opp": -1}.get(mode, 1)]["R"] > 0.3
        cfg = {"mode": mode, "phi0_deg": {str(k): v for k, v in phi0.items()}, "use_vflip": bool(strong and mode != "dual"),
               "img_size": a.img_size, "backbone": a.backbone, "runs": [a.run_name], "threshold": 0.5,
               "pilot": res, "probe": summarize_probe(d)}
        save_config(a.out_dir, cfg)
    print(f"\n>>> mode={cfg['mode']}  vflip={cfg['use_vflip']}")

    # 3) K-fold training -----------------------------------------------------------------------------------
    X = build_inputs(raw_tr, az, cfg["mode"], cfg["phi0_deg"])
    kfold_train(X, y, a.run_name, a.out_dir, n_folds=a.n_folds, folds=a.folds, seed=a.seed, resume=not a.no_resume,
                backbone=a.backbone, img_size=a.img_size, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
                flip=cfg["use_vflip"], num_workers=a.num_workers)

    # 4) out-of-fold score + threshold ---------------------------------------------------------------------
    probs, mask = gather_oof(a.out_dir, cfg["runs"], a.n_folds, len(y))
    thr, best, acc05, *_ = tune_threshold(y[mask], probs[mask, 1])
    print(f"\nOOF balanced accuracy @0.5 = {acc05:.4f} | tuned threshold {thr:.3f} -> {best:.4f}")
    print(confusion_matrix(y[mask], (probs[mask, 1] >= thr).astype(int)))
    cfg["threshold"], cfg["oof_bal_acc"] = thr, max(best, acc05)
    save_config(a.out_dir, cfg)


if __name__ == "__main__":
    main()
