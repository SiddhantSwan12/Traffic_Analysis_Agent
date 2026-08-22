"""Track post-processing: gap interpolation, smoothing, metric calibration,
class voting with size refinement, speed, and stationary/parked detection.

Everything here is deliberately done *after* tracking rather than inside it,
because these decisions need a track's whole history. Voting a class over 300
frames is far more reliable than trusting any single frame, and "is this thing
parked?" is only answerable once you know where it went.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from .taxonomy import ScaleModel, map_mode, refine_by_size, MODES


def _interpolate_gaps(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Fill short holes in a single track so boxes do not blink during brief
    occlusion. Rows added here are marked interpolated=True."""
    out = []
    for tid, g in df.groupby("track_id", sort=False):
        g = g.sort_values("frame")
        frames = g["frame"].to_numpy()
        gaps = np.diff(frames)
        fill_at = np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1))
        if fill_at.size == 0:
            out.append(g)
            continue
        pieces = [g]
        cols = ["x1", "y1", "x2", "y2"]
        for i in fill_at:
            f0, f1 = frames[i], frames[i + 1]
            a = g.iloc[i]
            b = g.iloc[i + 1]
            n = int(f1 - f0) - 1
            w = np.arange(1, n + 1, dtype=float) / (n + 1)
            block = pd.DataFrame({
                "frame": np.arange(f0 + 1, f1, dtype=np.int64),
                "track_id": tid,
                "conf": float(min(a["conf"], b["conf"])) * 0.8,
                "cls_id": int(a["cls_id"]),
                "cls_name": a["cls_name"],
                "interpolated": True,
                "camera_motion_quality": a.get("camera_motion_quality", np.nan),
            })
            for c in cols:
                block[c] = a[c] + (b[c] - a[c]) * w
            pieces.append(block)
        out.append(pd.concat(pieces, ignore_index=True).sort_values("frame"))
    return pd.concat(out, ignore_index=True)


def _smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Centred moving average on box coordinates, per contiguous segment.

    Detector jitter of +/-2 px on a 40 px box is very visible when the box is
    drawn at 30 fps. Smoothing is the single cheapest quality win here.
    """
    if window < 3:
        return df
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    new_seg = df.groupby("track_id")["frame"].diff().fillna(1) > 1
    df["_seg"] = new_seg.groupby(df["track_id"]).cumsum()
    g = df.groupby(["track_id", "_seg"], sort=False)
    for c in ["x1", "y1", "x2", "y2"]:
        df[c] = g[c].transform(
            lambda s: s.rolling(window, center=True, min_periods=1).mean())
    return df.drop(columns=["_seg"])


def calibrate_scale(df: pd.DataFrame, car_length_m: float) -> ScaleModel:
    """Metres-per-pixel from the apparent size of cars, as a function of row."""
    cars = df[(df["cls_name"] == "car") & (~df["interpolated"])]
    if len(cars) < 200:
        cars = df[~df["interpolated"]]
    w = (cars["x2"] - cars["x1"]).to_numpy()
    h = (cars["y2"] - cars["y1"]).to_numpy()
    cy = (0.5 * (cars["y1"] + cars["y2"])).to_numpy()
    return ScaleModel.fit(cy, np.maximum(w, h), car_length_m)


def process(tracks: pd.DataFrame, fps: float, cfg) -> tuple[pd.DataFrame, dict]:
    p = cfg.post
    df = tracks.copy()
    df["interpolated"] = False

    # 1. drop flicker: too short, or too few real observations
    stats = df.groupby("track_id").agg(
        n=("frame", "size"), f0=("frame", "min"), f1=("frame", "max"))
    stats["life"] = stats["f1"] - stats["f0"] + 1
    good = stats[(stats["life"] >= p.min_track_len) &
                 (stats["n"] >= p.min_track_hits)].index
    dropped_short = int(len(stats) - len(good))
    df = df[df["track_id"].isin(good)]
    if df.empty:
        raise RuntimeError("no tracks survived filtering - loosen min_track_len")

    # 2. fill short occlusion gaps, then smooth
    df = _interpolate_gaps(df, p.max_gap_interp)
    df = _smooth(df, p.smooth_window)

    # 3. pixels -> metres
    scale = calibrate_scale(df, p.car_length_m)
    cy = 0.5 * (df["y1"] + df["y2"]).to_numpy()
    cx = 0.5 * (df["x1"] + df["x2"]).to_numpy()
    df["cx"], df["cy"] = cx, cy
    df["mpp"] = scale.m_per_px(cy)
    box_len = np.maximum((df["x2"] - df["x1"]).to_numpy(),
                         (df["y2"] - df["y1"]).to_numpy())
    df["length_m"] = box_len * df["mpp"]

    # 4. class decision per track: confidence-weighted vote, then size refinement
    obs = df[~df["interpolated"]]
    obs = obs.assign(mode0=obs["cls_name"].map(map_mode))
    votes = (obs.dropna(subset=["mode0"])
                .groupby(["track_id", "mode0"])["conf"].sum()
                .reset_index()
                .sort_values("conf", ascending=False)
                .drop_duplicates("track_id")
                .set_index("track_id")["mode0"])
    med_len = df.groupby("track_id")["length_m"].median()
    coarse = votes.reindex(med_len.index).fillna("car")
    final_mode = {tid: refine_by_size(coarse[tid], float(med_len[tid]))
                  for tid in med_len.index}
    df["mode"] = df["track_id"].map(final_mode)

    # 5. speed, in metres per second.
    #    Measured as displacement across a +/-w frame window, NOT as a
    #    frame-to-frame difference. Detector jitter is zero-mean, but taking
    #    |difference| first and averaging afterwards rectifies that noise into a
    #    positive bias - which makes a perfectly static object appear to crawl
    #    at a few km/h. Differencing over a window cancels it instead.
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    w = max(2, int(round(fps / 4)))
    g = df.groupby("track_id", sort=False)
    dx = (g["cx"].shift(-w) - g["cx"].shift(w)).to_numpy()
    dy = (g["cy"].shift(-w) - g["cy"].shift(w)).to_numpy()
    dt = (g["frame"].shift(-w) - g["frame"].shift(w)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        px_per_frame = np.hypot(dx, dy) / dt
    df["speed_mps"] = px_per_frame * df["mpp"].to_numpy() * fps
    # window is undefined at the ends of a track; carry the nearest value in
    df["speed_mps"] = (df.groupby("track_id", sort=False)["speed_mps"]
                         .transform(lambda s: s.ffill().bfill()))
    df["speed_mps"] = df["speed_mps"].fillna(0.0).clip(lower=0.0)
    df["speed_mps"] = (df.groupby("track_id", sort=False)["speed_mps"]
                         .transform(lambda s: s.rolling(p.speed_smooth,
                                                        center=True,
                                                        min_periods=1).mean()))
    df["speed_kmh"] = df["speed_mps"] * 3.6

    # 6. parked vs moving.
    #    A vehicle halted at a signal still has a large path length over its
    #    life, so it stays "moving". Only genuinely stationary objects - parked
    #    cars, and any static false positive on a rooftop or facade - satisfy
    #    all four conditions at once. This is why no hand-drawn exclusion
    #    rectangle is needed.
    #    Path length is measured on centres resampled every ~0.5 s for the same
    #    reason as the speed window: summing every frame's displacement would
    #    accumulate jitter into metres of phantom travel over a long track.
    k = max(1, int(round(fps / 2)))

    def _path_m(gr):
        s = gr.iloc[::k]
        if len(s) < 2:
            return 0.0
        cx = s["cx"].to_numpy()
        cy = s["cy"].to_numpy()
        return float((np.hypot(np.diff(cx), np.diff(cy))
                      * s["mpp"].to_numpy()[:-1]).sum())

    nwin = max(1, int(round(p.parked_window_s * fps / k)))

    def _win_disp_m(gr):
        """Farthest the object got from itself within any parked_window_s window."""
        s = gr.iloc[::k]
        cx = s["cx"].to_numpy()
        cy = s["cy"].to_numpy()
        mpp = s["mpp"].to_numpy()
        n = len(cx)
        if n < 2:
            return 0.0
        w = min(nwin, n - 1)
        d = np.hypot(cx[w:] - cx[:-w], cy[w:] - cy[:-w]) * mpp[:-w]
        return float(d.max()) if d.size else 0.0

    agg = df.groupby("track_id").agg(
        f0=("frame", "min"), f1=("frame", "max"),
        moving_frac=("speed_mps", lambda s: float((s > p.moving_speed_mps).mean())),
        x_first=("cx", "first"), y_first=("cy", "first"),
        x_last=("cx", "last"), y_last=("cy", "last"),
        mpp=("mpp", "median"),
    )
    gb = df.groupby("track_id")[["cx", "cy", "mpp"]]
    agg["path_m"] = gb.apply(_path_m)            # diagnostic only, not a gate
    agg["win_m"] = gb.apply(_win_disp_m)
    agg["life_s"] = (agg["f1"] - agg["f0"] + 1) / fps
    agg["net_m"] = np.hypot(agg["x_last"] - agg["x_first"],
                            agg["y_last"] - agg["y_first"]) * agg["mpp"]
    agg["parked"] = ((agg["life_s"] >= p.parked_min_life_s) &
                     (agg["moving_frac"] < p.parked_moving_frac) &
                     (agg["net_m"] < p.parked_net_disp_m) &
                     (agg["win_m"] < p.parked_window_disp_m))
    df["parked"] = df["track_id"].map(agg["parked"])

    # 7. compact display ids, ordered by first appearance among moving tracks
    moving = agg[~agg["parked"]].sort_values("f0")
    disp = {tid: i + 1 for i, tid in enumerate(moving.index)}
    df["display_id"] = df["track_id"].map(disp).astype("Int64")

    per_mode = (agg.join(df.groupby("track_id")["mode"].first())
                   .query("~parked")["mode"].value_counts().to_dict())
    report = {
        "fps": round(float(fps), 4),
        "frames": int(df["frame"].max() + 1),
        "tracks_total": int(len(agg)),
        "tracks_moving": int((~agg["parked"]).sum()),
        "tracks_parked": int(agg["parked"].sum()),
        "tracks_dropped_short": dropped_short,
        "rows": int(len(df)),
        "rows_interpolated": int(df["interpolated"].sum()),
        "median_track_seconds": round(float(agg["life_s"].median()), 2),
        "p90_track_seconds": round(float(agg["life_s"].quantile(0.9)), 2),
        "counts_by_mode": {m: int(per_mode.get(m, 0)) for m in MODES},
        "scale": scale.describe(),
    }
    return df, report


def save(df: pd.DataFrame, report: dict, paths: dict):
    cols = ["frame", "track_id", "display_id", "mode", "cls_name", "conf",
            "x1", "y1", "x2", "y2", "cx", "cy", "speed_kmh", "length_m",
            "parked", "interpolated"]
    out = df[cols].sort_values(["frame", "display_id"])
    try:
        out.to_parquet(paths["final"], index=False)
    except Exception:
        pass
    out.to_csv(paths["csv"], index=False, float_format="%.3f")
    paths["report"].write_text(json.dumps(report, indent=2))
