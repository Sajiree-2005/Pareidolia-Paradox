"""physics.py - illumination handling for The Pareidolia Paradox (numpy + OpenCV only).

Background
----------
A crater and a mound look identical when the light direction is unknown (pareidolia).  The sun azimuth tells us where
the light comes from, so the task is to rotate every crop so the light always arrives from the same direction.

The competition text says "rotate CCW by -sun_azimuth_angle".  Rather than trusting that blindly, this module
*measures* how the shading direction in the images relates to the azimuth column:

  * light_moments()  : per-image "brightness dipole" vector (points toward the bright side of the object).
                       A mound is bright on the sun side, a crater on the far side, so the vector is +/- the light dir.
  * probe_light_model(): using the training labels, fits   light_angle = phi0 + a * azimuth   for a in {-2..2}
                       (a=+1 -> spec rule "-az" is right, a=-1 -> the opposite sign, both strong -> a MIXTURE of conventions).
  * route_hypotheses(): label-free per-image choice between a=+1 and a=-1 (works on test images too).
  * build_inputs()   : produces the network input for a chosen mode: spec | opp | dual | routed.

All angles are "visual": measured counter-clockwise from +x with y pointing UP.
"""
import math
import numpy as np
import cv2

MODES = ("spec", "opp", "dual", "routed")


# --------------------------------------------------------------------------------------- rotation
def rotate_ccw(img, deg):
    """Rotate a 2-D uint8 image counter-clockwise by `deg` degrees (reflect-pad -> rotate -> centre-crop; no black corners)."""
    h, w = img.shape[:2]
    diag = int(math.ceil(math.hypot(h, w)))
    ph, pw = (diag - h) // 2 + 4, (diag - w) // 2 + 4
    padded = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_REFLECT_101)
    H, W = padded.shape[:2]
    M = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), float(deg), 1.0)   # +deg = counter-clockwise
    rot = cv2.warpAffine(padded, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return rot[ph:ph + h, pw:pw + w]


def rotate_all(raw, degs):
    degs = np.broadcast_to(np.asarray(degs, dtype=np.float64), (len(raw),))
    return np.stack([rotate_ccw(raw[i], degs[i]) for i in range(len(raw))])


# --------------------------------------------------------------------------------------- measuring the light
def light_moments(raw, blur_sigma=24.0, win_sigma=70.0):
    """Per-image brightness-dipole vector (mx, my) in visual coords (x right, y up).
    High-passed (removes ramps), Gaussian-windowed around the centre object.  Returns float32 (N, 2)."""
    S = raw.shape[1]
    ys, xs = np.mgrid[0:S, 0:S]
    X = (xs - (S - 1) / 2.0).astype(np.float32)
    Y = -(ys - (S - 1) / 2.0).astype(np.float32)
    w = np.exp(-(X ** 2 + Y ** 2) / (2 * win_sigma ** 2)).astype(np.float32)
    out = np.empty((len(raw), 2), np.float32)
    for i, im in enumerate(raw):
        f = im.astype(np.float32)
        hp = (f - cv2.GaussianBlur(f, (0, 0), blur_sigma)) * w
        out[i] = ((hp * X).sum(), (hp * Y).sum())
    return out


def _weights(m):
    mag = np.hypot(m[:, 0], m[:, 1])
    return np.clip(mag / (np.median(mag) + 1e-9), 0, 3)


def probe_light_model(m, az_deg, y, n_null=5, seed=0):
    """Fit  light_angle = phi0 + a*azimuth  for a in {-2,-1,0,1,2} using labels to resolve the crater/mound 180-deg ambiguity.
    Returns {a: {"R": resultant length (0..1), "phi0": radians, "R_null": same statistic with shuffled azimuth}}."""
    theta = np.arctan2(m[:, 1], m[:, 0])
    psi = theta + np.pi * (1 - np.asarray(y))            # mound -> +light dir, crater -> flipped back by pi
    azr = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
    wt = _weights(m); rng = np.random.default_rng(seed)
    res = {}
    for a in (-2, -1, 0, 1, 2):
        c = np.sum(wt * np.exp(1j * (psi - a * azr))) / wt.sum()
        nulls = [abs(np.sum(wt * np.exp(1j * (psi - a * azr[rng.permutation(len(azr))]))) / wt.sum()) for _ in range(n_null)]
        res[a] = {"R": float(abs(c)), "phi0": float(np.angle(c)), "R_null": float(np.mean(nulls))}
    return res


def route_hypotheses(m, az_deg, phi0_deg):
    """Label-free routing: for each image pick a in {+1,-1} whose predicted light AXIS best matches the measured one.
    phi0_deg = {1: deg, -1: deg}.  Returns (a_per_image, score_plus, score_minus)."""
    theta = np.arctan2(m[:, 1], m[:, 0]); azr = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
    s = {a: np.cos(2 * (theta - np.deg2rad(phi0_deg[a]) - a * azr)) for a in (1, -1)}
    return np.where(s[1] >= s[-1], 1, -1), s[1], s[-1]


def discover_physics(raw, az_deg, y, seed=0):
    """Run the whole probe on the training set.  Returns a dict that is JSON-friendly except for 'm' (per-image moments)."""
    m = light_moments(raw)
    fit = probe_light_model(m, az_deg, y, seed=seed)
    phi0 = {1: math.degrees(fit[1]["phi0"]), -1: math.degrees(fit[-1]["phi0"])}
    theta = np.arctan2(m[:, 1], m[:, 0]); psi = theta + np.pi * (1 - np.asarray(y)); wt = _weights(m)

    def routed_R(az_used):
        a_i, _, _ = route_hypotheses(m, az_used, phi0)
        azr = np.deg2rad(az_used)
        L = np.where(a_i == 1, np.deg2rad(phi0[1]) + azr, np.deg2rad(phi0[-1]) - azr)
        return float(abs(np.sum(wt * np.exp(1j * (psi - L))) / wt.sum())), float(np.mean(a_i == 1))

    R_routed, p_plus = routed_R(np.asarray(az_deg, dtype=np.float64))
    rng = np.random.default_rng(seed + 1)
    R_routed_null = float(np.mean([routed_R(np.asarray(az_deg, dtype=np.float64)[rng.permutation(len(m))])[0] for _ in range(5)]))
    return {"m": m, "fit": fit, "phi0_deg": phi0, "R_routed": R_routed, "R_routed_null": R_routed_null, "p_plus": p_plus}


def summarize_probe(d):
    """Human-readable lines describing the probe result."""
    L = ["hypothesis: light_angle = phi0 + a*azimuth     (a=+1 <=> competition rule 'rotate CCW by -az' is correct)",
         f"{'a':>3} {'R':>8} {'null':>8} {'R/null':>8}"]
    for a in (-2, -1, 0, 1, 2):
        f = d["fit"][a]; L.append(f"{a:>3} {f['R']:>8.3f} {f['R_null']:>8.3f} {f['R'] / max(f['R_null'], 1e-9):>8.1f}")
    L.append(f"per-image routed between a=+1 / a=-1: R={d['R_routed']:.3f} (null {d['R_routed_null']:.3f}), share routed to a=+1: {d['p_plus']:.2f}")
    Rp, Rm = d["fit"][1]["R"], d["fit"][-1]["R"]
    if max(Rp, Rm) < 2 * max(d["fit"][1]["R_null"], d["fit"][-1]["R_null"]):
        L.append("verdict: weak/unclear relation between azimuth and measured shading (estimator is crude) -> rely on the pilot below.")
    elif min(Rp, Rm) > 0.5 * max(Rp, Rm) and d["R_routed"] > 1.2 * max(Rp, Rm):
        L.append("verdict: BOTH conventions present (mixture) -> 'routed' / 'dual' modes should win.")
    else:
        L.append(f"verdict: a single convention dominates ({'spec rule (-az)' if Rp >= Rm else 'opposite sign (+az)'}).")
    return L


# --------------------------------------------------------------------------------------- network inputs
def build_inputs(raw, az_deg, mode, phi0_deg, m=None):
    """Return uint8 array (N, H, W, C).  Every mode also adds a constant offset so the light ends up at angle 0 (from +x/right)
    under its hypothesis; a constant rotation is harmless for the CNN but makes the vertical flip label-preserving.
      spec   : rotate CCW by -(az + phi0[+1])           (competition rule)
      opp    : rotate CCW by -(-az + phi0[-1])          (opposite sign)
      dual   : channels [spec, opp, raw]                (network decides; safest fallback)
      routed : per image spec or opp, chosen from the image's own shading axis (needs `m`)"""
    az = np.asarray(az_deg, dtype=np.float64)
    d = lambda a: -(a * az + phi0_deg[a])
    if mode == "spec":
        return rotate_all(raw, d(1))[..., None]
    if mode == "opp":
        return rotate_all(raw, d(-1))[..., None]
    if mode == "dual":
        return np.stack([rotate_all(raw, d(1)), rotate_all(raw, d(-1)), raw], axis=-1)
    if mode == "routed":
        if m is None: m = light_moments(raw)
        a_i, _, _ = route_hypotheses(m, az, phi0_deg)
        return rotate_all(raw, np.where(a_i == 1, d(1), d(-1)))[..., None]
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
