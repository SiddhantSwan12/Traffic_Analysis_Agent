"""L2 analytics stage: raw tracks -> calibrated, classified, kinematic objects.

Runs after tracking, because every decision here needs a track's whole history:
you cannot vote on a class from one frame, and you cannot tell a parked car
from one waiting at a signal without knowing where it went.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from . import calibration as calib
from . import kinematics as kin
from .classification import classify_all
from .postprocess import _interpolate_gaps, _smooth
from .taxonomy import MODES

MOVING, STOPPED, PARKED, UNKNOWN = "moving", "temporarily_stopped", "parked", "unknown"


class _ClsCfg:
    min_length_samples = 5
    length_uncertainty_floor_m = 0.35


def _pct(series, q):
    v = series.to_numpy(float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else np.nan


def _filter_tracks(df: pd.DataFrame, p) -> tuple[pd.DataFrame, int]:
    stats = df.groupby("track_id").agg(n=("frame", "size"), f0=("frame", "min"),
                                       f1=("frame", "max"))
    stats["life"] = stats["f1"] - stats["f0"] + 1
    keep = stats[(stats["life"] >= p.min_track_len) &
                 (stats["n"] >= p.min_track_hits)].index
    return df[df["track_id"].isin(keep)], int(len(stats) - len(keep))


def _motion_state(df: pd.DataFrame, agg: pd.DataFrame, m) -> np.ndarray:
    """Per-frame motion state, informed by the track-level parked verdict."""
    parked = df["track_id"].map(agg["parked"]).to_numpy()
    speed = df["speed_mps"].to_numpy()
    valid = df["kinematics_valid"].to_numpy()

    state = np.full(len(df), UNKNOWN, dtype=object)
    state[parked] = PARKED
    live = (~parked) & valid & np.isfinite(speed)
    state[live & (speed > m.moving_speed_mps)] = MOVING
    state[live & (speed <= m.moving_speed_mps)] = STOPPED
    return state


def analyze(tracks: pd.DataFrame, fps: float, W: int, H: int, telemetry: dict,
            cfg) -> tuple[pd.DataFrame, dict, object]:
    p, k, m = cfg.post, cfg.kinematics, cfg.motion
    df = tracks.copy()
    df["interpolated"] = False

    df, dropped = _filter_tracks(df, p)
    if df.empty:
        raise RuntimeError("no tracks survived filtering - loosen min_track_len")

    if "camera_motion_quality" not in df.columns:
        df["camera_motion_quality"] = np.nan   # not recorded by an older cache
    df = _interpolate_gaps(df, p.max_gap_interp)
    df = _smooth(df, p.smooth_window)
    df["cx"] = 0.5 * (df["x1"] + df["x2"])
    df["cy"] = 0.5 * (df["y1"] + df["y2"])
    df["timestamp_s"] = df["frame"] / fps

    # ---- calibration, fitted on observed car boxes
    cars = df[(df["cls_name"] == "car") & (~df["interpolated"])]
    ys = (0.5 * (cars["y1"] + cars["y2"])).to_numpy()
    px = np.maximum((cars["x2"] - cars["x1"]).to_numpy(),
                    (cars["y2"] - cars["y1"]).to_numpy())
    cal = calib.build(ys, px, telemetry, W, H, k.reference_car_length_m,
                      prefer=k.calibration)

    # ---- kinematics
    df = kin.compute(df, cal, fps, cfg)

    # ---- classification (track level)
    cls = classify_all(df, cal, W, H, _ClsCfg)
    for c in ("final_mode", "appearance_mode", "mode_confidence",
              "classification_source", "estimated_length_m",
              "length_measurement_count", "length_uncertainty_m"):
        df[c] = df["track_id"].map(cls[c])
    df["mode"] = df["final_mode"]

    # ---- track-level motion summary
    recs = {}
    for tid, g in df.groupby("track_id", sort=False):
        path, net = kin.resample_path(g, fps)
        win = kin.max_window_displacement(g, fps, m.parked_window_s)
        sp = g["speed_mps"].to_numpy()
        ok = np.isfinite(sp)
        recs[tid] = {
            "f0": int(g["frame"].min()), "f1": int(g["frame"].max()),
            "path_m": path, "net_m": net, "win_m": win,
            "moving_frac": float((sp[ok] > m.moving_speed_mps).mean()) if ok.any() else np.nan,
            "median_speed_mps": float(np.median(sp[ok])) if ok.any() else np.nan,
            "p95_speed_mps": float(np.percentile(sp[ok], 95)) if ok.any() else np.nan,
            "max_speed_mps": float(sp[ok].max()) if ok.any() else np.nan,
            # p95/p05 rather than max/min: an extreme-value statistic over a
            # long track is defined by its single worst sample, which is
            # exactly the sample most likely to be noise.
            "p95_accel_mps2": _pct(g["longitudinal_acceleration_mps2"], 95),
            "p05_brake_mps2": _pct(g["longitudinal_acceleration_mps2"], 5),
            "peak_accel_mps2": _pct(g["longitudinal_acceleration_mps2"], 99.5),
            "peak_brake_mps2": _pct(g["longitudinal_acceleration_mps2"], 0.5),
            "n_obs": int((~g["interpolated"]).sum()),
        }
    agg = pd.DataFrame.from_dict(recs, orient="index")
    agg.index.name = "track_id"
    agg["life_s"] = (agg["f1"] - agg["f0"] + 1) / fps

    # Parked requires ALL FOUR. The window test replaces a total-path budget,
    # which was duration-dependent and therefore unsatisfiable for long tracks.
    agg["parked"] = ((agg["life_s"] >= m.parked_min_life_s) &
                     (agg["moving_frac"] < m.parked_moving_frac) &
                     (agg["net_m"] < m.parked_net_disp_m) &
                     (agg["win_m"] < m.parked_window_disp_m))
    df["parked"] = df["track_id"].map(agg["parked"])
    df["motion_state"] = _motion_state(df, agg, m)
    df["observed"] = ~df["interpolated"]

    # ---- compact display ids, ordered by first appearance among moving tracks
    moving = agg[~agg["parked"]].sort_values("f0")
    df["display_id"] = df["track_id"].map(
        {t: i + 1 for i, t in enumerate(moving.index)}).astype("Int64")

    report = _report(df, agg, cal, fps, cfg, dropped)
    return df, report, cal


def _stats_by_mode(df: pd.DataFrame, agg: pd.DataFrame) -> dict:
    out = {}
    mode_of = df.groupby("track_id")["mode"].first()
    for mode in MODES:
        ids = mode_of[mode_of == mode].index
        sub = agg.loc[agg.index.intersection(ids)]
        sub = sub[~sub["parked"]]
        if sub.empty:
            continue
        def q(col, f):
            v = sub[col].to_numpy()
            v = v[np.isfinite(v)]
            return round(float(f(v)), 2) if v.size else None
        out[mode] = {
            "n": int(len(sub)),
            "median_speed_kmh": q("median_speed_mps", np.median) and
                round(q("median_speed_mps", np.median) * 3.6, 1),
            "p95_speed_kmh": q("p95_speed_mps", np.median) and
                round(q("p95_speed_mps", np.median) * 3.6, 1),
            "p95_accel_mps2": q("p95_accel_mps2", np.median),
            "p05_brake_mps2": q("p05_brake_mps2", np.median),
            "peak_accel_mps2": q("peak_accel_mps2", lambda v: np.percentile(v, 95)),
            "peak_brake_mps2": q("peak_brake_mps2", lambda v: np.percentile(v, 5)),
        }
    return out


def _report(df, agg, cal, fps, cfg, dropped) -> dict:
    mv = agg[~agg["parked"]]
    mode_of = df.groupby("track_id")["mode"].first()
    counts = mode_of.loc[mv.index].value_counts().to_dict()
    src = df.groupby("track_id")["classification_source"].first().value_counts().to_dict()
    states = df["motion_state"].value_counts().to_dict()
    kv = df["kinematics_valid"]
    return {
        "fps": round(float(fps), 4),
        "frames": int(df["frame"].max() + 1),
        "tracks_total": int(len(agg)),
        "tracks_moving": int((~agg["parked"]).sum()),
        "tracks_parked": int(agg["parked"].sum()),
        "tracks_dropped_short": int(dropped),
        "rows": int(len(df)),
        "rows_interpolated": int(df["interpolated"].sum()),
        "rows_kinematics_valid": int(kv.sum()),
        "rows_kinematics_null": int((~kv).sum()),
        "median_track_seconds": round(float(agg["life_s"].median()), 2),
        "p90_track_seconds": round(float(agg["life_s"].quantile(0.9)), 2),
        "counts_by_mode": {m: int(counts.get(m, 0)) for m in MODES},
        "classification_source": src,
        "motion_state_rows": {str(k): int(v) for k, v in states.items()},
        "speed_stats_by_mode": _stats_by_mode(df, agg),
        "calibration": calib.describe(cal),
    }


L2_COLUMNS = [
    "frame", "timestamp_s", "display_id", "track_id", "mode", "mode_confidence",
    "classification_source", "appearance_mode",
    "x1", "y1", "x2", "y2", "cx", "cy",
    "world_x_m", "world_y_m",
    "velocity_x_mps", "velocity_y_mps", "speed_mps", "speed_kmh",
    "acceleration_x_mps2", "acceleration_y_mps2", "acceleration_mps2",
    "longitudinal_acceleration_mps2", "lateral_acceleration_mps2", "heading_deg",
    "estimated_length_m", "length_uncertainty_m", "length_measurement_count",
    "motion_state", "parked", "observed", "interpolated", "conf",
    "calibration_method", "calibration_confidence", "camera_motion_quality",
    "kinematics_valid", "kinematics_quality",
    "velocity_window_frames", "acceleration_window_frames",
    "interpolated_fraction_in_window",
]


def save(df: pd.DataFrame, report: dict, cal, paths: dict, write_csv=True):
    cols = [c for c in L2_COLUMNS if c in df.columns]
    out = df[cols].sort_values(["frame", "display_id"])
    out = out.rename(columns={"conf": "detection_confidence"})
    try:
        out.to_parquet(paths["final"], index=False)
    except Exception:
        pass
    if write_csv:
        out.to_csv(paths["csv"], index=False, float_format="%.4f",
                   na_rep="")            # empty, not 0 - unknown is not zero
    paths["report"].write_text(json.dumps(report, indent=2, default=str))
    calib.save(cal, paths["calib"])
