"""Time-space diagrams, Edie's generalised flow measures, and lane tracking.

These follow the pNEUMA/EPFL analysis vocabulary rather than inventing a
private one, because they are what traffic engineering already agrees on.

The time-space diagram is the centrepiece. Plotting distance along a corridor
against time, one polyline per vehicle, makes structure visible that no summary
statistic shows: a queue appears as a horizontal band, its discharge as a fan,
and a shockwave as a diagonal front whose slope IS the propagation speed. That
is the direct route to "where did this jam start", so it is built here rather
than left to L5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregate import project_to_centreline


# ------------------------------------------------------- time-space diagram
def time_space(df: pd.DataFrame, net, origin: str, destination: str,
               fps: float, max_offset_m: float = 12.0,
               sample_hz: float = 5.0, corridor_only: bool = False,
               vehicles_only: bool = True, min_progress_m: float = 25.0) -> dict:
    """Vehicle trajectories in the (time, distance-along-corridor) plane.

    By default EVERY vehicle whose path lies along the corridor is included,
    not only those with a complete origin-destination journey. A queue is
    largely made of vehicles that entered mid-clip or had not left by the end,
    and excluding them empties the diagram of exactly the structure it exists
    to show.
    """
    line = net.corridors.get((origin, destination))
    if line is None:
        return {}

    if corridor_only:
        tids = net.movements.index[(net.movements["origin"] == origin) &
                                   (net.movements["destination"] == destination)]
        sub = df[df["track_id"].isin(tids)]
    else:
        sub = df
    sub = sub[np.isfinite(sub["world_x_m"]) & np.isfinite(sub["world_y_m"])]
    if vehicles_only:
        sub = sub[sub["mode"] != "pedestrian"]
    if sub.empty:
        return {}

    s, off = project_to_centreline(sub["world_x_m"].to_numpy(),
                                   sub["world_y_m"].to_numpy(), line)
    sub = sub.assign(s_m=s, offset_m=off)
    sub = sub[np.isfinite(sub["s_m"]) & (np.abs(sub["offset_m"]) <= max_offset_m)]
    if sub.empty:
        return {}

    step = max(1, int(round(fps / sample_hz)))
    total_len = float(np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1])).sum())

    tracks = []
    for tid, g in sub.groupby("track_id", sort=False):
        g = g.sort_values("frame").iloc[::step]
        if len(g) < 3:
            continue
        # Something loitering beside the road is not a corridor trajectory, and
        # a vehicle that only clips the edge adds noise rather than signal.
        if float(g["s_m"].max() - g["s_m"].min()) < min_progress_m:
            continue
        sp = g["speed_kmh"].to_numpy()
        tracks.append({
            "id": int(g["display_id"].iloc[0]) if pd.notna(g["display_id"].iloc[0]) else -int(tid),
            "mode": str(g["mode"].iloc[0]),
            "t": [round(float(v), 2) for v in (g["frame"].to_numpy() / fps)],
            "s": [round(float(v), 1) for v in g["s_m"].to_numpy()],
            "v": [None if not np.isfinite(x) else round(float(x), 1) for x in sp],
        })
    return {"origin": origin, "destination": destination,
            "corridor_length_m": round(total_len, 1),
            "n_tracks": len(tracks), "tracks": tracks}


# --------------------------------------------------- Edie's generalised measures
def edie(df: pd.DataFrame, net, origin: str, destination: str, fps: float,
         segment_m: float = 50.0, interval_s: float = 20.0,
         max_offset_m: float = 12.0, lane_width_m: float = 3.2,
         vehicles_only: bool = True) -> pd.DataFrame:
    """Flow, density and speed by Edie's definitions over space-time regions.

    For a region A of extent (space x time):

        q = (total distance travelled inside A) / |A|
        k = (total time spent inside A)         / |A|
        v = q / k

    This is strictly better than counting vehicles present at sampled instants.
    A vehicle that crosses only part of the segment, or that enters halfway
    through the interval, contributes exactly its share; the naive count either
    includes it whole or not at all. Stopped vehicles also fall out correctly:
    they add time but no distance, which is what drives density up and speed
    down. q = k*v holds identically, by construction rather than by luck.

    Flow and density are PER LANE. A region spanning the whole corridor width
    sums several lanes, the opposing carriageway and the footway into one
    number, which here produced 750 veh/km and 8,800 veh/h -- both far beyond
    any physical jam density or lane capacity. The region is therefore split by
    lane, as pNEUMA's multilane analysis does, and each lane reported on its
    own.

    READ THE RESULT PER LOCATION, NOT POOLED. A fundamental diagram describes
    one place. Pooling cells from along a corridor mixes a queued segment with
    a free-flowing one and inverts the speed-density relationship by Simpson's
    paradox: measured here, the pooled correlation came out at +0.23, while
    16 of 19 individual (segment, lane) locations showed the required negative
    slope, several below -0.6. `fd_validity()` performs that check.
    """
    line = net.corridors.get((origin, destination))
    if line is None:
        return pd.DataFrame()
    sub = df[np.isfinite(df["world_x_m"]) & np.isfinite(df["world_y_m"])]
    if vehicles_only:
        # someone on the footway is not traffic on this carriageway
        sub = sub[sub["mode"] != "pedestrian"]
    if sub.empty:
        return pd.DataFrame()
    s, off = project_to_centreline(sub["world_x_m"].to_numpy(),
                                   sub["world_y_m"].to_numpy(), line)
    sub = sub.assign(s_m=s, offset_m=off)
    sub = sub[np.isfinite(sub["s_m"]) & (np.abs(sub["offset_m"]) <= max_offset_m)]
    if sub.empty:
        return pd.DataFrame()

    # Keep only traffic travelling WITH the corridor. The centreline is fitted
    # to one movement, so the opposing carriageway falls inside the same offset
    # band; mixing a free-flowing opposing stream with a congested one destroys
    # the speed-density relationship outright (it measured +0.03 here, where
    # every traffic-flow theory requires a negative slope).
    keep = np.ones(len(sub), bool)
    if "velocity_x_mps" in sub.columns:
        vx = sub["velocity_x_mps"].to_numpy()
        vy = sub["velocity_y_mps"].to_numpy()
        s_arr0 = sub["s_m"].to_numpy()
        # local corridor direction at each sample, from the polyline
        cum = np.r_[0, np.cumsum(np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1])))]
        seg = np.clip(np.searchsorted(cum, s_arr0) - 1, 0, len(line) - 2)
        dxc = line[seg + 1, 0] - line[seg, 0]
        dyc = line[seg + 1, 1] - line[seg, 1]
        dot = vx * dxc + vy * dyc
        moving = np.isfinite(dot) & (np.hypot(vx, vy) > 0.5)
        keep = (~moving) | (dot > 0)     # stopped vehicles keep their place
    sub = sub[keep]
    if sub.empty:
        return pd.DataFrame()
    sub = sub.assign(lane=np.floor(sub["offset_m"].to_numpy() / lane_width_m).astype(int))

    total_len = float(np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1])).sum())
    n_seg = max(1, int(total_len // segment_m))
    dt = 1.0 / fps
    rows = []

    sub = sub.sort_values(["track_id", "frame"])
    tid_arr = sub["track_id"].to_numpy()
    s_arr = sub["s_m"].to_numpy()
    f_arr = sub["frame"].to_numpy()

    # per-sample step in space and time, within a track only
    same = np.r_[False, tid_arr[1:] == tid_arr[:-1]]
    ds = np.zeros(len(sub))
    dtv = np.zeros(len(sub))
    ds[1:] = np.abs(np.diff(s_arr))
    dtv[1:] = np.diff(f_arr) * dt
    ds[~same] = 0.0
    dtv[~same] = 0.0
    # a gap in the track is not travel through this region
    broken = dtv > (5 * dt)
    ds[broken] = 0.0
    dtv[broken] = 0.0

    seg_idx = np.clip((s_arr // segment_m).astype(int), 0, n_seg - 1)
    iv_idx = (f_arr / fps // interval_s).astype(int)

    frame = pd.DataFrame({"seg": seg_idx, "iv": iv_idx, "ds": ds, "dt": dtv,
                          "tid": tid_arr, "lane": sub["lane"].to_numpy()})
    area = segment_m * interval_s          # |A| in metre-seconds, ONE lane wide
    for (sg, iv, ln), g in frame.groupby(["seg", "iv", "lane"]):
        dist = float(g["ds"].sum())
        time = float(g["dt"].sum())
        if time <= 0:
            continue
        q = dist / area * 3600.0                  # veh/h
        k = time / area * 1000.0                  # veh/km
        v = (dist / time) * 3.6 if time > 0 else np.nan   # km/h
        rows.append((int(sg), sg * segment_m, int(iv), iv * interval_s, int(ln),
                     q, k, v, int(g["tid"].nunique())))
    out = pd.DataFrame(rows, columns=[
        "segment", "segment_start_m", "interval", "interval_start_s", "lane",
        "flow_veh_per_h", "density_veh_per_km", "speed_kmh", "n_vehicles"])
    if len(out):
        out.attrs["identity_residual"] = float(np.nanmax(np.abs(
            out["flow_veh_per_h"] - out["density_veh_per_km"] * out["speed_kmh"])))
    return out.round({"flow_veh_per_h": 1, "density_veh_per_km": 2, "speed_kmh": 2})


def fd_validity(e: pd.DataFrame, min_cells: int = 6, min_vehicles: int = 3,
                lane_range: tuple = (-3, 2)) -> dict:
    """Check the fundamental diagram behaves the way flow theory requires.

    Speed must fall as density rises AT A GIVEN LOCATION. This reports the
    share of locations that satisfy it, which is a genuine correctness signal:
    if most locations came out positive, the calibration, the lane split or the
    direction filter would be wrong, not traffic.
    """
    sub = e[(e["n_vehicles"] >= min_vehicles) &
            (e["lane"].between(*lane_range))]
    rows = []
    for (sg, ln), g in sub.groupby(["segment", "lane"]):
        if len(g) < min_cells:
            continue
        c = np.corrcoef(g["density_veh_per_km"], g["speed_kmh"])[0, 1]
        if np.isfinite(c):
            rows.append({"segment": int(sg), "lane": int(ln), "cells": len(g),
                         "corr": round(float(c), 3),
                         "median_speed_kmh": round(float(g["speed_kmh"].median()), 1),
                         "p95_density": round(float(np.percentile(
                             g["density_veh_per_km"], 95)), 1)})
    if not rows:
        return {"locations": 0, "negative_share": None, "detail": []}
    neg = sum(1 for r in rows if r["corr"] < 0)
    return {"locations": len(rows), "negative": neg,
            "negative_share": round(neg / len(rows), 3),
            "median_corr": round(float(np.median([r["corr"] for r in rows])), 3),
            "detail": sorted(rows, key=lambda r: (r["segment"], r["lane"]))}


# --------------------------------------------------------------- lane tracking
def assign_lanes(df: pd.DataFrame, net, fps: float, lane_width_m: float = 3.2,
                 max_offset_m: float = 14.0, smooth_s: float = 0.7,
                 hysteresis_m: float = 0.7, dwell_s: float = 1.0) -> pd.DataFrame:
    """Per-frame lane index for every vehicle on a corridor.

    L4 asks for the correct lane "moment to moment", which a single lane per
    track cannot answer. Lateral offset from the corridor centreline is
    computed every frame, smoothed over half a second to suppress box jitter,
    and quantised into lanes. Smoothing matters: raw offsets straddle a lane
    boundary constantly, which would report a lane change several times a
    second where there is none.

    Returns a frame indexed like `df` with lane, lateral offset, and a
    lane-change flag.
    """
    out = pd.DataFrame(index=df.index)
    out["corridor"] = None
    out["lane"] = np.nan
    out["lateral_offset_m"] = np.nan

    win = max(3, int(round(smooth_s * fps)))
    for (o, d), line in net.corridors.items():
        tids = net.movements.index[(net.movements["origin"] == o) &
                                   (net.movements["destination"] == d)]
        m = df["track_id"].isin(tids) & np.isfinite(df["world_x_m"])
        if not m.any():
            continue
        sel = df[m]
        s, off = project_to_centreline(sel["world_x_m"].to_numpy(),
                                       sel["world_y_m"].to_numpy(), line)
        ok = np.isfinite(off) & (np.abs(off) <= max_offset_m)
        idx = sel.index[ok]
        if not len(idx):
            continue
        tmp = pd.DataFrame({"tid": sel["track_id"].to_numpy()[ok],
                            "frame": sel["frame"].to_numpy()[ok],
                            "off": off[ok]}, index=idx).sort_values(["tid", "frame"])
        sm = (tmp.groupby("tid")["off"]
                 .transform(lambda s: s.rolling(win, center=True, min_periods=1).median()))
        out.loc[tmp.index, "corridor"] = f"{o}->{d}"
        out.loc[tmp.index, "lateral_offset_m"] = sm.to_numpy()
        out.loc[tmp.index, "lane"] = np.floor(sm.to_numpy() / lane_width_m).astype(float)

    # A lane change has to be a committed move, not a wobble across a boundary.
    # Quantising even a smoothed offset flips whenever a vehicle tracks near a
    # lane edge, which reported 48 changes/km against a real urban rate of
    # 0.5-2. Hysteresis fixes it: the offset must clear the boundary by a
    # margin AND hold the new lane for `dwell_s` before the change counts.
    out["lane_change"] = False
    have = out["lane"].notna()
    if have.any():
        dwell = max(3, int(round(dwell_s * fps)))
        tmp = pd.DataFrame({"tid": df.loc[have, "track_id"].to_numpy(),
                            "frame": df.loc[have, "frame"].to_numpy(),
                            "off": out.loc[have, "lateral_offset_m"].to_numpy()},
                           index=out.index[have]).sort_values(["tid", "frame"])
        lanes_out = np.full(len(tmp), np.nan)
        change_out = np.zeros(len(tmp), bool)
        pos = 0
        for _, g in tmp.groupby("tid", sort=False):
            o = g["off"].to_numpy()
            n = len(o)
            cur = int(np.floor(o[0] / lane_width_m))
            seq = np.full(n, float(cur))
            pending, pend_lane = 0, cur
            for i in range(1, n):
                lo = cur * lane_width_m - hysteresis_m
                hi = (cur + 1) * lane_width_m + hysteresis_m
                if lo <= o[i] <= hi:
                    pending, pend_lane = 0, cur
                else:
                    cand = int(np.floor(o[i] / lane_width_m))
                    pending = pending + 1 if cand == pend_lane else 1
                    pend_lane = cand
                    if pending >= dwell:
                        cur = cand
                        change_out[pos + i] = True
                        pending = 0
                seq[i] = cur
            lanes_out[pos:pos + n] = seq
            pos += n
        out.loc[tmp.index, "lane"] = lanes_out
        out.loc[tmp.index, "lane_change"] = change_out
    return out


def lane_change_summary(df: pd.DataFrame, lanes: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Lane changes per vehicle and per kilometre, by corridor."""
    j = df[["track_id", "frame", "world_x_m", "world_y_m"]].join(lanes)
    j = j[j["lane"].notna()]
    if j.empty:
        return pd.DataFrame()
    rows = []
    for corr, g in j.groupby("corridor"):
        n_tracks = g["track_id"].nunique()
        n_changes = int(g["lane_change"].sum())
        dist_km = 0.0
        for _, t in g.groupby("track_id"):
            x = t["world_x_m"].to_numpy()
            y = t["world_y_m"].to_numpy()
            dist_km += float(np.nansum(np.hypot(np.diff(x), np.diff(y)))) / 1000.0
        rows.append((corr, n_tracks, n_changes,
                     round(n_changes / max(n_tracks, 1), 2),
                     round(n_changes / max(dist_km, 1e-6), 1),
                     round(dist_km, 2)))
    return pd.DataFrame(rows, columns=[
        "corridor", "n_vehicles", "lane_changes", "changes_per_vehicle",
        "changes_per_km", "vehicle_km"])
