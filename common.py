"""common.py - data loading, model, training and inference utilities (shared by train.py, inference.py and the Colab notebook)."""
import os, json, math, time, random, copy, zipfile
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix  # noqa: F401 (re-exported)
from tqdm.auto import tqdm

from physics import *   # rotate_ccw, light_moments, discover_physics, build_inputs, ...

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ======================================================================================= data
def find_image_root(root, sample_name):
    for dp, _, files in os.walk(root):
        if sample_name in files:
            return dp
    raise FileNotFoundError(f"{sample_name} not found under {root}")


def _image_dir(data_dir, work_dir, name, sample):
    plain = os.path.join(data_dir, name)
    if os.path.isdir(plain) and os.listdir(plain):
        return find_image_root(plain, sample)
    zpath = os.path.join(data_dir, name + ".zip")
    if not os.path.exists(zpath):
        raise FileNotFoundError(f"Need {zpath} (or an extracted folder {plain})")
    out = os.path.join(work_dir, name)
    if not os.path.isdir(out) or not os.listdir(out):
        print(f"Extracting {name}.zip ..."); os.makedirs(out, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    return find_image_root(out, sample)


def load_gray_stack(df, img_dir, size=256):
    arr = np.empty((len(df), size, size), np.uint8)
    for i, iid in enumerate(tqdm(df["image_id"].values, desc=f"loading {os.path.basename(img_dir)}")):
        im = cv2.imread(os.path.join(img_dir, iid), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"Unreadable image: {iid}")
        if im.shape != (size, size):
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        arr[i] = im
    return arr


def load_data(data_dir, work_dir="/tmp/pareidolia_work", need_train=True):
    """Returns (train_meta, RAW_TR, test_meta, RAW_TE); train parts are None when need_train=False."""
    os.makedirs(work_dir, exist_ok=True)
    test_meta = pd.read_csv(os.path.join(data_dir, "test_metadata.csv"))
    assert {"image_id", "sun_azimuth_angle"} <= set(test_meta.columns) and test_meta["sun_azimuth_angle"].notna().all()
    raw_te = load_gray_stack(test_meta, _image_dir(data_dir, work_dir, "test_images", test_meta.image_id.iloc[0]))
    if not need_train:
        return None, None, test_meta, raw_te
    train_meta = pd.read_csv(os.path.join(data_dir, "train_metadata.csv"))
    assert {"image_id", "label", "sun_azimuth_angle"} <= set(train_meta.columns) and train_meta["sun_azimuth_angle"].notna().all()
    raw_tr = load_gray_stack(train_meta, _image_dir(data_dir, work_dir, "train_images", train_meta.image_id.iloc[0]))
    return train_meta, raw_tr, test_meta, raw_te


# ======================================================================================= dataset / augmentation
def make_transforms(img_size, flip):
    """`flip` = vertical flip.  It is label-preserving ONLY because build_inputs() puts the light at angle 0 (from the right):
    mirroring top<->bottom leaves the light direction unchanged.  Never flip left<->right or rotate by 90/180 deg."""
    S = img_size
    aug = [A.Resize(S, S)]
    if flip:
        aug.append(A.VerticalFlip(p=0.5))
    aug += [A.Affine(translate_percent=0.03, scale=(0.92, 1.08), rotate=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(p=0.2), A.CoarseDropout(p=0.15),
            A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    base = [A.Resize(S, S), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    tta = [A.Compose(base)]
    if flip:
        tta.append(A.Compose([A.Resize(S, S), A.VerticalFlip(p=1.0), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]))
    return A.Compose(aug), A.Compose(base), tta


class LunarDataset(Dataset):
    """imgs: uint8 (N,H,W,C) with C = 1 (replicated to 3) or 3."""
    def __init__(self, imgs, labels, transform):
        self.imgs, self.labels, self.transform = imgs, labels, transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        im = self.imgs[i]
        if im.shape[-1] == 1:
            im = np.repeat(im, 3, axis=2)
        x = self.transform(image=np.ascontiguousarray(im))["image"]
        return x if self.labels is None else (x, torch.tensor(int(self.labels[i]), dtype=torch.long))


# ======================================================================================= model
def build_backbone(name, pretrained=True, drop_path=0.0):
    for kw in ({"drop_path_rate": drop_path}, {}):
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg", **kw), name
        except TypeError:
            continue
        except Exception as e:
            print(f"Could not load '{name}' ({e}); falling back to resnet50.")
            break
    return timm.create_model("resnet50", pretrained=pretrained, num_classes=0, global_pool="avg"), "resnet50"


class LunarNet(nn.Module):
    def __init__(self, backbone_name, pretrained=True, drop_path=0.0, num_classes=2):
        super().__init__()
        self.backbone, self.backbone_name = build_backbone(backbone_name, pretrained, drop_path)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.backbone.num_features, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.2), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x))


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__(); self.alpha, self.gamma, self.ls = alpha, gamma, label_smoothing
    def forward(self, logits, y):
        pt = torch.exp(-F.cross_entropy(logits, y, label_smoothing=self.ls, reduction="none").detach())
        ce = F.cross_entropy(logits, y, weight=self.alpha, label_smoothing=self.ls, reduction="none")
        return (((1 - pt) ** self.gamma) * ce).mean()


def class_weights(y):
    c = np.bincount(y, minlength=2); return torch.tensor(c.sum() / (len(c) * c), dtype=torch.float32)


class EMA:
    """Exponential moving average of the weights (with warm-up)."""
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0
    @torch.no_grad()
    def update(self, model):
        self.n += 1; d = min(self.decay, (1 + self.n) / (10 + self.n)); msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def _evaluate(model, loader, crit, device, amp):
    model.eval(); tot, probs, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, enabled=amp):
            out = model(x)
        tot += crit(out.float(), y).item() * x.size(0)
        probs.append(F.softmax(out.float(), 1).cpu().numpy()); labels.append(y.cpu().numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    return tot / len(loader.dataset), balanced_accuracy_score(labels, probs.argmax(1)), probs


# ======================================================================================= training
def fit_model(X_tr, y_tr, X_va, y_va, backbone, img_size=320, epochs=10, lr=1e-4, batch_size=16, flip=False, use_ema=True,
              ema_decay=0.999, weight_decay=0.05, patience=4, gamma=2.0, label_smoothing=0.1, drop_path=0.2,
              head_lr_mult=10.0, grad_clip=1.0, warmup_epochs=1, num_workers=2, pretrained=True, save_path=None,
              meta=None, device=None, log=print):
    """Trains one model. Model selection uses balanced accuracy (the competition metric). Returns dict(best, probs, epochs_run)."""
    device = device or DEVICE; amp = device == "cuda"
    train_tf, val_tf, _ = make_transforms(img_size, flip)
    tr = DataLoader(LunarDataset(X_tr, y_tr, train_tf), batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=amp, drop_last=len(X_tr) >= 2 * batch_size, persistent_workers=num_workers > 0)
    va = DataLoader(LunarDataset(X_va, y_va, val_tf), batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=amp)
    model = LunarNet(backbone, pretrained, drop_path).to(device)
    crit = FocalLoss(class_weights(y_tr).to(device), gamma, label_smoothing)
    opt = torch.optim.AdamW([{"params": list(model.backbone.parameters()), "lr": lr},
                             {"params": list(model.head.parameters()), "lr": lr * head_lr_mult}], weight_decay=weight_decay)
    total, warm = epochs * len(tr), max(1, warmup_epochs * len(tr))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, ema_decay) if use_ema else None
    eval_model = ema.module if ema is not None else model

    best, best_probs, bad, ran = -1.0, None, 0, 0
    for ep in range(epochs):
        t0 = time.time(); model.train(); tot = 0.0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item() * x.size(0)
        vl, vb, vp = _evaluate(eval_model, va, crit, device, amp); ran = ep + 1
        log(f"  epoch {ep + 1:>2}/{epochs} | train {tot / len(tr.dataset):.4f} | val_loss {vl:.4f} | val_bal_acc {vb:.4f} | {time.time() - t0:.0f}s")
        if vb > best:
            best, best_probs, bad = vb, vp, 0
            if save_path:
                torch.save({"model_state": eval_model.state_dict(), "backbone": model.backbone_name, "img_size": img_size,
                            "val_bal_acc": vb, "meta": meta or {}}, save_path)
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop at epoch {ep + 1}"); break
    del model, ema
    if amp:
        torch.cuda.empty_cache()
    return {"best": best, "probs": best_probs, "epochs_run": ran}


def kfold_train(X, y, run_name, out_dir, n_folds=5, folds=None, seed=42, resume=True, log=print, **fit_kwargs):
    os.makedirs(out_dir, exist_ok=True)
    splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    scores = {}
    for f in (range(n_folds) if folds is None else folds):
        ckpt, oof = os.path.join(out_dir, f"model_{run_name}_fold{f}.pt"), os.path.join(out_dir, f"oof_{run_name}_fold{f}.npz")
        if resume and os.path.exists(ckpt) and os.path.exists(oof):
            log(f"fold {f}: already finished - skipping"); continue
        log(f"\n===== {run_name} | fold {f} ====="); set_seed(seed + f)
        tr, va = splits[f]
        res = fit_model(X[tr], y[tr], X[va], y[va], save_path=ckpt, log=log, **fit_kwargs)
        np.savez(oof, idx=va, probs=res["probs"]); scores[f] = res["best"]
        log(f"fold {f} best val balanced accuracy: {res['best']:.4f}")
    return scores


def gather_oof(out_dir, runs, n_folds, n):
    probs = np.zeros((n, 2), np.float32); cnt = np.zeros(n)
    for r in runs:
        for f in range(n_folds):
            p = os.path.join(out_dir, f"oof_{r}_fold{f}.npz")
            if os.path.exists(p):
                d = np.load(p); probs[d["idx"]] += d["probs"]; cnt[d["idx"]] += 1
    m = cnt > 0; probs[m] /= cnt[m, None]
    return probs, m


def tune_threshold(y, p1, lo=0.2, hi=0.8):
    """Threshold on P(class 1) maximising balanced accuracy; only adopted if it beats 0.5 by > 0.1 %."""
    ths = np.linspace(lo, hi, 121); accs = np.array([balanced_accuracy_score(y, (p1 >= t).astype(int)) for t in ths])
    acc05 = balanced_accuracy_score(y, (p1 >= 0.5).astype(int)); t = float(ths[accs.argmax()])
    return (t if accs.max() - acc05 > 0.001 else 0.5), float(accs.max()), float(acc05), ths, accs


def pilot(raw, az, y, backbone="efficientnet_b0.ra_in1k", modes=MODES, phi0_deg=None, img_size=224, epochs=5, n_max=5000,
          lr=3e-4, batch_size=32, seed=0, num_workers=2, pretrained=True, log=print):
    """Fast experiment: same subsample / same split / same small model for each candidate input representation."""
    idx = np.arange(len(y))
    if len(idx) > n_max:
        idx, _ = train_test_split(idx, train_size=n_max, stratify=y, random_state=seed)
    tr, va = train_test_split(np.arange(len(idx)), test_size=0.25, stratify=y[idx], random_state=seed)
    m_all = light_moments(raw[idx]); out = {}
    for mode in modes:
        log(f"\n--- pilot mode: {mode} ---")
        X = build_inputs(raw[idx], az[idx], mode, phi0_deg, m=m_all)
        res = fit_model(X[tr], y[idx][tr], X[va], y[idx][va], backbone=backbone, img_size=img_size, epochs=epochs, lr=lr,
                        batch_size=batch_size, flip=False, use_ema=False, drop_path=0.0, weight_decay=1e-2, patience=epochs,
                        num_workers=num_workers, pretrained=pretrained, log=log)
        out[mode] = res["best"]; log(f"pilot {mode}: best val balanced accuracy {res['best']:.4f}")
        del X
    return out


# ======================================================================================= inference
@torch.no_grad()
def predict_probs(model, X, img_size, flip, device=None, batch_size=64, num_workers=2):
    device = device or DEVICE; amp = device == "cuda"; model.eval()
    _, _, tta = make_transforms(img_size, flip); total = np.zeros((len(X), 2), np.float32)
    for tf in tta:
        loader = DataLoader(LunarDataset(X, None, tf), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        outs = []
        for x in loader:
            with torch.autocast(device_type=device, enabled=amp):
                outs.append(F.softmax(model(x.to(device)).float(), 1).cpu().numpy())
        total += np.concatenate(outs)
    return total / len(tta)


def ensemble_predict(weights_dir, runs, X, flip, n_folds=5, device=None, log=print):
    device = device or DEVICE; probs, used = [], []
    for r in runs:
        for f in range(n_folds):
            p = os.path.join(weights_dir, f"model_{r}_fold{f}.pt")
            if not os.path.exists(p):
                continue
            ck = torch.load(p, map_location=device, weights_only=False)
            m = LunarNet(ck["backbone"], pretrained=False).to(device); m.load_state_dict(ck["model_state"])
            log(f"{r} fold {f} (val_bal_acc={ck['val_bal_acc']:.4f}) -> inference")
            probs.append(predict_probs(m, X, ck["img_size"], flip, device)); used.append((r, f)); del m
    assert probs, f"No checkpoints matching model_<run>_fold<k>.pt found in {weights_dir}"
    return np.mean(probs, 0), used


def save_config(out_dir, cfg):
    with open(os.path.join(out_dir, "pipeline_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def load_config(out_dir):
    with open(os.path.join(out_dir, "pipeline_config.json")) as f:
        cfg = json.load(f)
    cfg["phi0_deg"] = {int(k): float(v) for k, v in cfg["phi0_deg"].items()}
    return cfg


def make_submission(test_meta, p1, threshold, path):
    sub = pd.DataFrame({"image_id": test_meta["image_id"].values, "label": (p1 >= threshold).astype(int)})
    assert len(sub) == 2000, f"Expected 2000 rows, got {len(sub)}"
    assert list(sub.columns) == ["image_id", "label"] and sub["label"].isin([0, 1]).all() and sub.notna().all().all()
    assert sub["image_id"].is_unique
    sub.to_csv(path, index=False)
    return sub
