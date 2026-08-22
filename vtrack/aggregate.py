"""L3: aggregate insight across many road users, time windows and regions.

Everything here summarises; nothing here decides anything about an individual
vehicle. The unit of analysis is a movement, an interval, an approach, a lane
or a segment.

Covers the six areas the brief names:
  1. classified counts by movement and interval
  2. origin-destination distributions
  3. speed profiles along a corridor, and where speeding concentrates
  4. lane volume and modal split by lane
  5. queue lengths
  6. density, occupancy, and the flow-density relationship
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .taxonomy import MODES

# Passenger-car equivalents, for capacity-style aggregates. Indian Roads
# Congress guidance for urban roads; a two-wheeler occupies far less road space
# than a car, and counting both as "one vehicle" overstates motorcycle demand.
PCE = {"pedestrian": 0.0, "motorcycle": 0.35, "car": 1.0, "LGV": 1.4,
       "truck": 2.2, "HGV": 3.7, "bus": 3.0}


# ------------------------------------------------- 1. counts by movement/interval
def movement_counts(net, fps: float, interval_s: float = 60.0,
                    by_mode: bool = True) -> pd.DataFrame:
    """Turning-movement counts per time interval.

    A movement is attributed to the interval in which the vehicle ENTERED the
    scene, so a single journey is counted once and cannot straddle two bins.
    """
    mv = net.movements[net.movements["turn"] != "internal"].copy()
    if mv.empty:
        return pd.DataFrame()
    mv["t_s"] = mv["f0"] / fps
    mv["interval"] = (mv["t_s"] // interval_s).astype(int)
    keys = ["interval", "origin", "destination", "turn"]
    if by_mode:
        keys.append("mode")
    out = mv.groupby(keys).size().reset_index(name="count")
    out["interval_start_s"] = out["interval"] * interval_s
    out["pce"] = out.apply(
        lambda r: r["count"] * PCE.get(r.get("mode", "car"), 1.0), axis=1) \
        if by_mode else out["count"]
    return out


def approach_volumes(net, fps: float, interval_s: float = 60.0) -> pd.DataFrame:
    """Directional volume entering and leaving each approach, per interval."""
    mv = net.movements[net.movements["turn"] != "internal"].copy()
    if mv.empty:
        return pd.DataFrame()
    mv["interval"] = (mv["f0"] / fps // interval_s).astype(int)
    ent = mv.groupby(["interval", "origin"]).size().rename("entering")
    ext = mv.groupby(["interval", "destination"]).size().rename("leaving")
    out = pd.concat([ent, ext], axis=1).fillna(0).astype(int)
    out.index.names = ["interval", "approach"]
    out = out.reset_index()
    out["interval_start_s"] = out["interval"] * interval_s
    span_h = interval_s / 3600.0
    out["entering_veh_per_h"] = (out["entering"] / span_h).round(0)
    # The last interval is usually truncated by the end of the clip; without a
    # flag its low count reads as a collapse in demand rather than less time.
    clip_end_s = float(net.movements["f1"].max()) / fps
    out["interval_end_s"] = out["interval_start_s"] + interval_s
    out["partial"] = out["interval_end_s"] > clip_end_s
    out.loc[out["partial"], "entering_veh_per_h"] = np.nan
    return out


# --------------------------------------------- 2. origin-destination distribution
def od_summary(net) -> pd.DataFrame:
    mv = net.movements[net.movements["turn"] != "internal"]
    if mv.empty:
        return pd.DataFrame()
    tot = len(mv)
    g = (mv.groupby(["origin", "destination", "turn"])
           .agg(count=("f0", "size"),
                median_travel_m=("travel_m", "median")).reset_index())
    g["share_pct"] = (100.0 * g["count"] / tot).round(1)
    return g.sort_values("count", ascending=False)


def desire_lines(net, geo=None) -> dict:
    """Origin-destination flows as map-native GeoJSON.

    Uses each movement's observed centreline where one exists, so a desire line
    follows the path vehicles actually took rather than a straight chord between
    two gates.
    """
    feats = []
    by_name = {g.name: g for g in net.gates}
    summ = od_summary(net)
    for r in summ.itertuples(index=False):
        line = net.corridors.get((r.origin, r.destination))
        if line is None:
            o, d = by_name.get(r.origin), by_name.get(r.destination)
            if o is None or d is None:
                continue
            line = np.array([[o.x, o.y], [d.x, d.y]])
        if geo is not None:
            lat, lon = geo.to_wgs84(line[:, 0], line[:, 1])
            coords = [[round(float(a), 8), round(float(b), 8)]
                      for a, b in zip(lon, lat)]
        else:
            coords = [[round(float(a), 2), round(float(b), 2)] for a, b in line]
        feats.append({"type": "Feature",
                      "geometry": {"type": "LineString", "coordinates": coords},
                      "properties": {"origin": r.origin, "destination": r.destination,
                                     "turn": r.turn, "count": int(r.count),
                                     "share_pct": float(r.share_pct)}})
    return {"type": "FeatureCollection", "features": feats}


# ------------------------------------------------------- 3. speed along a corridor
def project_to_centreline(X, Y, line: np.ndarray):
    """Signed distance along a polyline, and lateral offset from it."""
    P = np.column_stack([np.asarray(X, float), np.asarray(Y, float)])
    seg_a = line[:-1]
    seg_b = line[1:]
    d = seg_b - seg_a
    L = np.hypot(d[:, 0], d[:, 1])
    L = np.where(L < 1e-9, 1e-9, L)
    cum = np.r_[0, np.cumsum(L)]

    best_s = np.full(len(P), np.nan)
    best_off = np.full(len(P), np.nan)
    best_d2 = np.full(len(P), np.inf)
    for i in range(len(seg_a)):
        w = P - seg_a[i]
        t = np.clip((w[:, 0] * d[i, 0] + w[:, 1] * d[i, 1]) / (L[i] ** 2), 0, 1)
        proj = seg_a[i] + t[:, None] * d[i]
        diff = P - proj
        d2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
        m = d2 < best_d2
        best_d2[m] = d2[m]
        best_s[m] = cum[i] + t[m] * L[i]
        # sign the offset so left and right of the centreline are separable
        cross = d[i, 0] * diff[m, 1] - d[i, 1] * diff[m, 0]
        best_off[m] = np.sign(cross) * np.sqrt(d2[m])
    return best_s, best_off


def speed_profile(df: pd.DataFrame, net, origin: str, destination: str,
                  bin_m: float = 10.0, speed_limit_kmh: float | None = None,
                  max_offset_m: float = 12.0) -> pd.DataFrame:
    """Speed as a function of distance along one movement's corridor."""
    line = net.corridors.get((origin, destination))
    if line is None:
        return pd.DataFrame()
    mv = net.movements
    tids = mv.index[(mv["origin"] == origin) & (mv["destination"] == destination)]
    sub = df[df["track_id"].isin(tids)]
    sub = sub[np.isfinite(sub["speed_kmh"]) & np.isfinite(sub["world_x_m"])]
    if sub.empty:
        return pd.DataFrame()

    s, off = project_to_centreline(sub["world_x_m"].to_numpy(),
                                   sub["world_y_m"].to_numpy(), line)
    keep = np.isfinite(s) & (np.abs(off) <= max_offset_m)
    sub = sub.assign(s_m=s, offset_m=off)[keep]
    if sub.empty:
        return pd.DataFrame()

    sub["bin"] = (sub["s_m"] // bin_m).astype(int)
    agg = sub.groupby("bin").agg(
        distance_m=("s_m", "mean"),
        n_samples=("speed_kmh", "size"),
        n_vehicles=("track_id", "nunique"),
        mean_speed_kmh=("speed_kmh", "mean"),
        median_speed_kmh=("speed_kmh", "median"),
        p15_speed_kmh=("speed_kmh", lambda s: float(np.percentile(s, 15))),
        p85_speed_kmh=("speed_kmh", lambda s: float(np.percentile(s, 85))),
    ).reset_index()
    if speed_limit_kmh:
        over = sub["speed_kmh"] > speed_limit_kmh
        agg = agg.merge(
            sub.assign(over=over).groupby("bin")["over"].mean()
               .rename("exceeding_share").reset_index(), on="bin", how="left")
        agg["exceeding_share"] = (100 * agg["exceeding_share"]).round(1)
    return agg.round(2)


# ------------------------------------------------ 4. lane volume and modal split
def lane_volumes(net) -> pd.DataFrame:
    mv = net.movements
    if "lane_in" not in mv.columns:
        return pd.DataFrame()
    sub = mv[mv["lane_in"].notna() & (mv["turn"] != "internal")].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["lane_in"] = sub["lane_in"].astype(int)
    out = (sub.groupby(["origin", "lane_in", "mode"]).size()
              .rename("count").reset_index())
    tot = out.groupby(["origin", "lane_in"])["count"].transform("sum")
    out["lane_share_pct"] = (100 * out["count"] / tot).round(1)
    out["pce"] = [c * PCE.get(m, 1.0) for c, m in zip(out["count"], out["mode"])]
    return out.sort_values(["origin", "lane_in", "count"], ascending=[True, True, False])


def modal_split(net) -> pd.DataFrame:
    mv = net.movements[net.movements["turn"] != "internal"]
    if mv.empty:
        return pd.DataFrame()
    out = mv.groupby("mode").size().rename("count").reindex(MODES).fillna(0).astype(int)
    res = out.to_frame()
    res["share_pct"] = (100 * res["count"] / max(res["count"].sum(), 1)).round(1)
    res["pce_total"] = [c * PCE.get(m, 1.0) for m, c in zip(res.index, res["count"])]
    return res.reset_index().rename(columns={"index": "mode"})


# ------------------------------------------------------------- 5. queue lengths
def queue_lengths(df: pd.DataFrame, net, fps: float, gate_name: str,
                  stop_speed_kmh: float = 5.0, max_offset_m: float = 12.0,
                  link_gap_m: float = 12.0, sample_s: float = 1.0) -> pd.DataFrame:
    """Queue extent on one approach, sampled through time.

    A queue is the contiguous run of slow vehicles reaching back from the stop
    line. Contiguity matters: two stopped vehicles 60 m apart are not one queue,
    they are a queue and an unrelated stopped vehicle, so the run is broken
    wherever the gap between consecutive members exceeds `link_gap_m`.
    """
    gate = next((g for g in net.gates if g.name == gate_name), None)
    if gate is None:
        return pd.DataFrame()
    line = None
    for (o, d), c in net.corridors.items():
        if o == gate_name:
            line = c
            break
    if line is None:
        return pd.DataFrame()

    # Every vehicle physically on the approach counts toward its queue, not
    # only those whose complete origin-destination journey was observed.
    # Restricting to assigned movements discards vehicles that entered mid-clip
    # or were still queued when the clip ended -- exactly the ones a long queue
    # is made of -- and understates queue length badly.
    sub = df[np.isfinite(df["world_x_m"]) & np.isfinite(df["speed_kmh"])]
    if sub.empty:
        return pd.DataFrame()
    s, off = project_to_centreline(sub["world_x_m"].to_numpy(),
                                   sub["world_y_m"].to_numpy(), line)
    sub = sub.assign(s_m=s, offset_m=off)
    sub = sub[np.isfinite(sub["s_m"]) & (np.abs(sub["offset_m"]) <= max_offset_m)]
    if sub.empty:
        return pd.DataFrame()

    step = max(1, int(round(sample_s * fps)))
    rows = []
    for f, g in sub[sub["frame"] % step == 0].groupby("frame"):
        slow = g[g["speed_kmh"] <= stop_speed_kmh].sort_values("s_m")
        if slow.empty:
            rows.append((f, f / fps, 0, 0.0, 0.0))
            continue
        d = slow["s_m"].to_numpy()
        # walk back from the stop line, breaking at the first oversized gap
        n, tail = 1, d[0]
        for i in range(1, len(d)):
            if d[i] - d[i - 1] > link_gap_m:
                break
            n += 1
            tail = d[i]
        rows.append((f, f / fps, n, float(tail - d[0]), float(d[0])))
    out = pd.DataFrame(rows, columns=["frame", "t_s", "queue_vehicles",
                                      "queue_length_m", "queue_head_s_m"])
    out["approach"] = gate_name
    return out


# ---------------------------------------- 6. density, occupancy, flow relationship
def fundamental_diagram(df: pd.DataFrame, net, origin: str, destination: str,
                        segment_m: float = 60.0, interval_s: float = 15.0,
                        fps: float = 29.97, max_offset_m: float = 12.0) -> pd.DataFrame:
    """Flow, density and space-mean speed on one corridor segment.

    Density is measured as a true spatial average (vehicles present on the
    segment, averaged over the interval), and speed as the space-mean speed,
    which is what makes q = k*v hold. Using the arithmetic mean of spot speeds
    instead would bias the identity, so the residual is reported as a check.
    """
    line = net.corridors.get((origin, destination))
    if line is None:
        return pd.DataFrame()
    mv = net.movements
    tids = mv.index[(mv["origin"] == origin) & (mv["destination"] == destination)]
    sub = df[df["track_id"].isin(tids)]
    sub = sub[np.isfinite(sub["world_x_m"]) & np.isfinite(sub["speed_kmh"])]
    if sub.empty:
        return pd.DataFrame()
    s, off = project_to_centreline(sub["world_x_m"].to_numpy(),
                                   sub["world_y_m"].to_numpy(), line)
    sub = sub.assign(s_m=s, offset_m=off)
    total_len = float(np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1])).sum())
    lo = max(0.0, total_len / 2 - segment_m / 2)
    hi = lo + segment_m
    seg = sub[np.isfinite(sub["s_m"]) & (np.abs(sub["offset_m"]) <= max_offset_m)
              & (sub["s_m"] >= lo) & (sub["s_m"] < hi)]
    if seg.empty:
        return pd.DataFrame()

    seg = seg.copy()
    seg["interval"] = (seg["frame"] / fps // interval_s).astype(int)
    rows = []
    for iv, g in seg.groupby("interval"):
        n_frames = g["frame"].nunique()
        if n_frames == 0:
            continue
        # density: mean number present, over segment length
        present = g.groupby("frame")["track_id"].nunique().mean()
        k = float(present / (segment_m / 1000.0))                    # veh/km
        v_space = float(len(g) / np.sum(1.0 / np.clip(g["speed_kmh"], 1e-3, None)))
        q = k * v_space                                              # veh/h
        # occupancy: share of interval during which any vehicle is on the segment
        occ = float(min(1.0, n_frames / max(interval_s * fps, 1)))
        rows.append((iv, iv * interval_s, k, v_space, q, 100 * occ,
                     int(g["track_id"].nunique())))
    out = pd.DataFrame(rows, columns=[
        "interval", "interval_start_s", "density_veh_per_km",
        "space_mean_speed_kmh", "flow_veh_per_h", "occupancy_pct",
        "distinct_vehicles"])
    # q = k*v holds exactly by construction; the check belongs on unrounded
    # values, so rounding happens only on the way out
    out.attrs["identity_residual"] = float(np.max(np.abs(
        out["flow_veh_per_h"] - out["density_veh_per_km"] * out["space_mean_speed_kmh"])))
    return out.round({"density_veh_per_km": 1, "space_mean_speed_kmh": 1,
                      "flow_veh_per_h": 0, "occupancy_pct": 1})
