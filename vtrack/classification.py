"""Track-level transport-mode classification.

One class per track, not one per frame. A confidence-weighted vote across the
whole history is far more reliable than any single detection, and it is what
stops a label flickering car -> van -> truck as a vehicle crosses the scene.

LGV and HGV are weight categories that appearance cannot settle from the air,
so they are resolved from physical length. Because a length estimate carries
real uncertainty, a measurement that sits too close to a threshold does NOT
override the appearance vote -- it lowers the reported confidence instead. An
unstable override that flips a vehicle between truck and HGV on a centimetre of
noise is worse than an honest "probably a truck".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .taxonomy import map_mode, MODES

# Ordered length ladder, in metres. Applied to the appearance mode.
LADDER = {
    "car":   [(6.2, "LGV"), (7.5, "truck"), (9.5, "HGV")],
    "LGV":   [(7.5, "truck"), (9.5, "HGV")],
    "truck": [(9.5, "HGV")],
    "HGV":   [],
    "bus":   [],
}
BUS_MIN_M = 7.0

# Length reasoning applies to vehicles only. A pedestrian or a two-wheeler must
# never be promoted into a goods-vehicle class by a bad box.
NON_VEHICLE = {"pedestrian", "motorcycle"}


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def length_samples(g: pd.DataFrame, cal, W: int, H: int, min_conf: float = 0.35,
                   min_box_px: int = 14, edge_px: int = 3):
    """Apparent length in metres per observation, with unusable boxes rejected.

    A box clipped by the frame edge is truncated and reads short; a tiny box has
    a length quantised by a handful of pixels. Both are excluded rather than
    averaged in, because the median is meant to describe the vehicle, not the
    measurement conditions.
    """
    obs = g[~g["interpolated"]]
    if obs.empty:
        return np.array([]), np.array([])
    x1 = obs["x1"].to_numpy(); y1 = obs["y1"].to_numpy()
    x2 = obs["x2"].to_numpy(); y2 = obs["y2"].to_numpy()
    conf = obs["conf"].to_numpy()
    w = x2 - x1
    h = y2 - y1
    cy = 0.5 * (y1 + y2)

    clipped = (x1 <= edge_px) | (y1 <= edge_px) | (x2 >= W - 1 - edge_px) | (y2 >= H - 1 - edge_px)
    ok = (~clipped) & (conf >= min_conf) & (np.maximum(w, h) >= min_box_px)
    if not ok.any():
        return np.array([]), np.array([])

    px = np.maximum(w, h)[ok]
    lm = np.asarray(cal.length_m(px, cy[ok]), float)
    wt = conf[ok]
    good = np.isfinite(lm) & (lm > 0)
    lm, wt = lm[good], wt[good]
    if lm.size < 3:
        return lm, wt

    # drop extreme outliers relative to the track's own median
    med = np.median(lm)
    keep = (lm > 0.35 * med) & (lm < 2.5 * med)
    return lm[keep], wt[keep]


def classify_track(g: pd.DataFrame, cal, W: int, H: int, cfg) -> dict:
    """Return the final mode and its supporting evidence for one track."""
    obs = g[~g["interpolated"]]
    votes = {}
    for name, c in zip(obs["cls_name"], obs["conf"]):
        m = map_mode(name)
        if m:
            votes[m] = votes.get(m, 0.0) + float(c)
    if not votes:
        return {"appearance_mode": "car", "final_mode": "car", "mode_confidence": 0.0,
                "classification_source": "appearance_fallback_no_calibration",
                "estimated_length_m": np.nan, "length_measurement_count": 0,
                "length_uncertainty_m": np.nan, "vote_margin": 0.0, "vote_entropy": np.nan}

    total = sum(votes.values())
    probs = {k: v / total for k, v in votes.items()}
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    app_mode, p1 = ranked[0]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    ent = float(-sum(p * np.log(max(p, 1e-12)) for p in probs.values()))

    res = {
        "appearance_mode": app_mode,
        "mode_confidence": float(p1),
        "vote_margin": float(p1 - p2),
        "vote_entropy": ent,
        "estimated_length_m": np.nan,
        "length_measurement_count": 0,
        "length_uncertainty_m": np.nan,
        "classification_source": "appearance",
        "final_mode": app_mode,
    }

    if cal.method == "uncalibrated":
        res["classification_source"] = "appearance_fallback_no_calibration"
        return res
    if app_mode in NON_VEHICLE:
        return res

    lm, wt = length_samples(g, cal, W, H)
    if lm.size < cfg.min_length_samples:
        return res

    med = _weighted_median(lm, wt)
    mad = float(np.median(np.abs(lm - med)))
    sigma = 1.4826 * mad
    unc = float(1.253 * sigma / max(np.sqrt(lm.size), 1.0)) if sigma > 0 else 0.0
    unc = max(unc, cfg.length_uncertainty_floor_m)
    res.update({"estimated_length_m": float(med),
                "length_measurement_count": int(lm.size),
                "length_uncertainty_m": unc})

    # bus is only demoted, never promoted, by length
    if app_mode == "bus":
        if med < BUS_MIN_M - unc:
            res["final_mode"] = "LGV"
            res["classification_source"] = "length_override"
            res["mode_confidence"] = float(p1 * 0.75)
        elif med < BUS_MIN_M + unc:
            res["mode_confidence"] = float(p1 * 0.8)
            res["classification_source"] = "appearance_and_length"
        return res

    final = app_mode
    ambiguous = False
    for thresh, promoted in LADDER.get(app_mode, []):
        if med > thresh + unc:
            final = promoted
        elif med > thresh - unc:
            ambiguous = True      # straddles the boundary: do not commit
            break
        else:
            break

    if ambiguous:
        # keep the appearance class and say we are less sure, rather than
        # flipping the label on a measurement that cannot resolve the question
        res["final_mode"] = app_mode
        res["mode_confidence"] = float(p1 * 0.7)
        res["classification_source"] = "appearance_and_length"
    elif final != app_mode:
        res["final_mode"] = final
        res["classification_source"] = "length_override"
        res["mode_confidence"] = float(min(1.0, p1 * 0.9 + 0.1))
    else:
        res["classification_source"] = "appearance_and_length"
        res["mode_confidence"] = float(min(1.0, p1 * 1.05))
    return res


def classify_all(df: pd.DataFrame, cal, W: int, H: int, cfg) -> pd.DataFrame:
    rows = {}
    for tid, g in df.groupby("track_id", sort=False):
        rows[tid] = classify_track(g, cal, W, H, cfg)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "track_id"
    assert set(out["final_mode"]) <= set(MODES), "classifier emitted an unknown mode"
    return out
