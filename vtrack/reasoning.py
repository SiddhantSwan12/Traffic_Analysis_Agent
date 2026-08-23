"""L5: reasoning across space and time, with the evidence attached.

Every function here answers a question a traffic engineer would actually ask,
and each returns the evidence it reasoned from rather than a bare verdict. That
matters because most of these questions have no ground truth in the footage: a
claim like "this jam started at 132 m at t=205 s" is only useful if the reader
can see the cells it came from.

Where a question cannot be answered from the data, that is reported as such
instead of being filled in with a plausible-looking number. Signal performance
is the sharp case: if the junction is unsignalised, there are no cycles to
measure, and inventing a saturation flow rate for it would be worse than
returning nothing.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .aggregate import project_to_centreline
from . import trafficflow as tflow


# ============================================================ 1. congestion
def congestion_origins(edie: pd.DataFrame, congested_kmh: float = 15.0,
                       min_cells: int = 4) -> dict:
    """Trace each congested region back to where and when it began.

    A jam is visible everywhere at once, which is exactly why the question is
    hard. The answer is in the space-time plane: congestion forms a connected
    region there, and its earliest cell is its origin. The upstream boundary of
    that region moves at the shockwave speed, and the sign of that speed says
    whether the jam is growing backwards into approaching traffic or clearing.

    Positive shockwave speed means the congested region's upstream edge moves
    downstream, i.e. the queue is discharging. Negative means it is propagating
    upstream against the traffic, which is the dangerous case.
    """
    if edie is None or not len(edie):
        return {"regions": [], "note": "no flow cells available"}

    # aggregate lanes: a jam is a property of the carriageway, not one lane
    cell = (edie.groupby(["segment", "interval"])
                .agg(speed_kmh=("speed_kmh", "mean"),
                     density=("density_veh_per_km", "mean"),
                     flow=("flow_veh_per_h", "sum"),
                     seg_m=("segment_start_m", "first"),
                     t_s=("interval_start_s", "first"),
                     n=("n_vehicles", "sum"))
                .reset_index())
    slow = cell[cell["speed_kmh"] < congested_kmh]
    if slow.empty:
        return {"regions": [],
                "note": f"no cell below {congested_kmh} km/h; free flow throughout"}

    # connected components over the (segment, interval) lattice
    key = {(int(r.segment), int(r.interval)): i for i, r in enumerate(slow.itertuples())}
    seen, comps = set(), []
    for k in key:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            s, t = stack.pop()
            comp.append((s, t))
            for ns, nt in ((s + 1, t), (s - 1, t), (s, t + 1), (s, t - 1)):
                if (ns, nt) in key and (ns, nt) not in seen:
                    seen.add((ns, nt))
                    stack.append((ns, nt))
        if len(comp) >= min_cells:
            comps.append(comp)

    clip_start = float(cell["t_s"].min())
    idx = slow.reset_index(drop=True)
    lut = {(int(r.segment), int(r.interval)): r for r in idx.itertuples()}
    out = []
    for comp in comps:
        rows = [lut[c] for c in comp]
        t0 = min(r.t_s for r in rows)
        first = [r for r in rows if r.t_s == t0]
        # among the earliest cells, the origin is the most severely congested
        origin = min(first, key=lambda r: r.speed_kmh)

        # upstream edge over time -> shockwave speed
        back = defaultdict(list)
        for r in rows:
            back[r.t_s].append(r.seg_m)
        ts = np.array(sorted(back))
        edge = np.array([min(back[t]) for t in ts])

        # A queue has TWO shockwaves, not one: a stopping wave that carries the
        # tail upstream while the queue grows, and a starting wave that carries
        # it back downstream as the queue discharges. Fitting one line across
        # both averages them to roughly zero and reports "stationary" for a
        # queue that plainly moved, so the series is split at its furthest
        # upstream extent and each phase fitted separately.
        shock = grow = clear = np.nan
        if len(ts) >= 3:
            shock = float(np.polyfit(ts, edge, 1)[0])       # m/s, whole life
            k = int(np.argmin(edge))
            if k >= 2:
                grow = float(np.polyfit(ts[:k + 1], edge[:k + 1], 1)[0])
            if len(ts) - k >= 3:
                clear = float(np.polyfit(ts[k:], edge[k:], 1)[0])

        def _dir(v):
            if not np.isfinite(v):
                return None
            return "upstream" if v < -0.2 else "downstream" if v > 0.2 else "stationary"

        # A jam already under way in the first interval did not start inside the
        # footage, so its "origin" is only where it was first SEEN. Saying
        # otherwise would attribute a cause to a moment the camera never saw.
        predates = bool(abs(t0 - clip_start) < 1e-6)
        out.append({
            "predates_clip": predates,
            "origin_qualifier": ("first observed at the start of the recording; "
                                 "formation happened before it")
                                if predates else "formation observed",
            "origin_distance_m": round(float(origin.seg_m), 1),
            "origin_time_s": round(float(origin.t_s), 1),
            "origin_speed_kmh": round(float(origin.speed_kmh), 1),
            "start_s": round(float(t0), 1),
            "end_s": round(float(max(r.t_s for r in rows)), 1),
            "duration_s": round(float(max(r.t_s for r in rows) - t0), 1),
            "extent_m": round(float(max(r.seg_m for r in rows) -
                                    min(r.seg_m for r in rows)), 1),
            "cells": len(rows),
            "min_speed_kmh": round(float(min(r.speed_kmh for r in rows)), 1),
            "peak_density_veh_km": round(float(max(r.density for r in rows)), 1),
            "shockwave_mps": None if not np.isfinite(shock) else round(shock, 2),
            "propagating": _dir(shock),
            "stopping_wave_mps": None if not np.isfinite(grow) else round(grow, 2),
            "starting_wave_mps": None if not np.isfinite(clear) else round(clear, 2),
            "growth_direction": _dir(grow),
            "discharge_direction": _dir(clear),
            "max_extent_at_s": (round(float(ts[int(np.argmin(edge))]), 1)
                                if len(ts) >= 3 else None),
        })
    out.sort(key=lambda r: -r["cells"])
    return {"regions": out, "congested_below_kmh": congested_kmh,
            "note": f"{len(out)} congested region(s) found"}


# ======================================================== 2. signal behaviour
def stopline_crossings(df: pd.DataFrame, net, origin: str, destination: str,
                       fps: float, at_m: float | None = None,
                       max_offset_m: float = 12.0) -> pd.DataFrame:
    """Time at which each vehicle crosses a chosen point on the corridor."""
    line = net.corridors.get((origin, destination))
    if line is None:
        return pd.DataFrame()
    sub = df[np.isfinite(df["world_x_m"]) & (df["mode"] != "pedestrian")]
    if sub.empty:
        return pd.DataFrame()
    s, off = project_to_centreline(sub["world_x_m"].to_numpy(),
                                   sub["world_y_m"].to_numpy(), line)
    sub = sub.assign(s_m=s, offset_m=off)
    sub = sub[np.isfinite(sub["s_m"]) & (np.abs(sub["offset_m"]) <= max_offset_m)]
    if sub.empty:
        return pd.DataFrame()
    total = float(np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1])).sum())
    x0 = total * 0.5 if at_m is None else at_m

    rows = []
    for tid, g in sub.groupby("track_id", sort=False):
        g = g.sort_values("frame")
        sm = g["s_m"].to_numpy()
        if sm.min() > x0 or sm.max() < x0:
            continue
        i = int(np.argmax(sm >= x0))
        if i == 0:
            continue
        # linear interpolation to the exact crossing instant
        f0, f1 = g["frame"].to_numpy()[i - 1], g["frame"].to_numpy()[i]
        s0, s1 = sm[i - 1], sm[i]
        w = 0.0 if s1 == s0 else (x0 - s0) / (s1 - s0)
        rows.append((int(tid), float((f0 + w * (f1 - f0)) / fps),
                     str(g["mode"].iloc[0]),
                     float(np.nanmedian(g["speed_kmh"].to_numpy()))))
    out = pd.DataFrame(rows, columns=["track_id", "t_s", "mode", "median_speed_kmh"])
    return out.sort_values("t_s").reset_index(drop=True)


def signal_performance(df: pd.DataFrame, net, origin: str, destination: str,
                       fps: float, red_gap_s: float = 12.0,
                       min_cycles: int = 3, max_cycle_cov: float = 0.25,
                       max_green_split: float = 0.80) -> dict:
    """Discharge headways, saturation flow and cycle structure at a stop line.

    Signal timings are not in the data, so the cycle has to be inferred from
    behaviour: a red period shows up as a gap in stop-line crossings much
    longer than any headway inside a discharging platoon.

    If no such structure exists, the junction is very likely unsignalised and
    this says so. Reporting a saturation flow rate for an uncontrolled junction
    would be a fabrication, so nothing is reported instead.
    """
    cr = stopline_crossings(df, net, origin, destination, fps)
    if len(cr) < 8:
        return {"signalised": None, "note": "too few stop-line crossings to judge",
                "crossings": int(len(cr))}

    t = cr["t_s"].to_numpy()
    h = np.diff(t)
    breaks = np.flatnonzero(h > red_gap_s)
    if len(breaks) + 1 < min_cycles:
        return {
            "signalised": False,
            "note": ("no repeated long interruptions in the discharge stream; "
                     "the approach behaves as uncontrolled"),
            "crossings": int(len(cr)),
            "median_headway_s": round(float(np.median(h)), 2),
            "longest_gap_s": round(float(h.max()), 1),
            "gaps_over_threshold": int(len(breaks)),
            "threshold_s": red_gap_s,
            "flow_veh_per_h": round(3600.0 / float(np.median(h)), 0),
        }

    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(t) - 1]

    # Long gaps alone do NOT mean a signal. Light minor-road traffic produces
    # them naturally, and calling that a red phase invents cycle times that
    # were never there. A fixed-time signal has a near-constant cycle length
    # and a green split well under 100%; both are checked before claiming one.
    cyc_lengths = np.diff(t[starts]) if len(starts) > 2 else np.array([])
    cov = (float(np.std(cyc_lengths) / max(np.mean(cyc_lengths), 1e-6))
           if cyc_lengths.size >= 2 else np.inf)
    greens = t[ends] - t[starts]
    reds_all = (t[starts[1:]] - t[ends[:-1]]) if len(starts) > 1 else np.array([])
    split = (float(np.sum(greens[:-1]) / max(np.sum(greens[:-1]) + np.sum(reds_all), 1e-6))
             if reds_all.size else 1.0)
    if cov > max_cycle_cov or split > max_green_split:
        return {
            "signalised": False,
            "note": ("interruptions are irregular, so they are gaps in a light "
                     "stream rather than red phases"),
            "crossings": int(len(cr)),
            "cycle_length_cov": None if not np.isfinite(cov) else round(cov, 2),
            "cycle_cov_threshold": max_cycle_cov,
            "apparent_green_split": round(split, 2),
            "median_headway_s": round(float(np.median(h)), 2),
            "gaps_over_threshold": int(len(breaks)),
        }

    cycles, sats, startups = [], [], []
    for a, b in zip(starts, ends):
        n = b - a + 1
        if n < 3:
            continue
        hh = np.diff(t[a:b + 1])
        # the first few headways carry start-up lost time; saturation flow is
        # measured on the steady part of the platoon that follows
        steady = hh[2:] if len(hh) > 3 else hh
        sat = 3600.0 / float(np.mean(steady)) if len(steady) else np.nan
        cycles.append({
            "start_s": round(float(t[a]), 1), "end_s": round(float(t[b]), 1),
            "vehicles": int(n),
            "green_s": round(float(t[b] - t[a]), 1),
            "first_headways_s": [round(float(x), 2) for x in hh[:3]],
            "saturation_flow_veh_h": None if not np.isfinite(sat) else round(sat, 0),
        })
        if np.isfinite(sat):
            sats.append(sat)
        if len(hh) >= 3:
            startups.append(float(hh[0] - np.median(hh[2:]) if len(hh) > 3 else hh[0]))

    reds = [float(t[s] - t[e]) for s, e in zip(starts[1:], ends[:-1])]
    return {
        "signalised": True,
        "note": f"{len(cycles)} discharge cycles inferred from crossing gaps",
        "crossings": int(len(cr)),
        "cycles": cycles,
        "n_cycles": len(cycles),
        "median_saturation_flow_veh_h": (round(float(np.median(sats)), 0)
                                         if sats else None),
        "median_startup_lost_time_s": (round(float(np.median(startups)), 2)
                                       if startups else None),
        "median_red_s": round(float(np.median(reds)), 1) if reds else None,
        "median_green_s": round(float(np.median([c["green_s"] for c in cycles])), 1),
        "green_utilisation": (round(float(np.mean(
            [c["vehicles"] / max(c["green_s"], 1e-6) for c in cycles])), 2)
            if cycles else None),
    }


# ==================================================== 3. gap acceptance
def gap_acceptance(df: pd.DataFrame, net, minor: tuple, major: tuple,
                   fps: float, conflict_m: float | None = None,
                   wait_speed_kmh: float = 6.0) -> dict:
    """Gaps in the major stream that minor-road drivers accepted or rejected.

    A minor-road vehicle waits at the give-way line, watches gaps go by, and
    eventually takes one. The gap it takes is ACCEPTED; every gap that passed
    while it was still waiting was REJECTED. The critical gap sits between the
    two distributions, and the honest estimate is where they overlap: the
    largest rejected gaps should be comparable to the smallest accepted ones.
    """
    maj = stopline_crossings(df, net, major[0], major[1], fps, at_m=conflict_m)
    mnr = stopline_crossings(df, net, minor[0], minor[1], fps, at_m=conflict_m)
    if len(maj) < 6 or len(mnr) < 3:
        return {"note": "not enough conflicting traffic to measure gap acceptance",
                "major_crossings": int(len(maj)), "minor_crossings": int(len(mnr))}

    mt = maj["t_s"].to_numpy()
    gaps = np.diff(mt)
    accepted, rejected = [], []

    # when did each minor vehicle start waiting?
    line = net.corridors.get((minor[0], minor[1]))
    waits = {}
    if line is not None:
        sub = df[df["track_id"].isin(mnr["track_id"])]
        for tid, g in sub.groupby("track_id", sort=False):
            g = g.sort_values("frame")
            sp = g["speed_kmh"].to_numpy()
            slow = sp < wait_speed_kmh
            if slow.any():
                waits[int(tid)] = float(g["frame"].to_numpy()[np.argmax(slow)] / fps)

    per_driver = []
    for r in mnr.itertuples():
        entry = r.t_s
        i = int(np.searchsorted(mt, entry))
        if i == 0 or i >= len(mt):
            continue
        accepted.append(float(mt[i] - mt[i - 1]))
        # gaps that opened and closed while this driver was still waiting
        w = waits.get(int(r.track_id))
        n_rej = 0
        if w is not None and w < entry:
            for j in range(len(gaps)):
                if mt[j] >= w and mt[j + 1] <= entry:
                    rejected.append(float(gaps[j]))
                    n_rej += 1
        per_driver.append(n_rej)

    if not accepted:
        return {"note": "no conflicting crossings could be paired"}
    acc = np.array(accepted)
    rej = np.array(rejected) if rejected else np.array([])
    crit = None
    if rej.size >= 3:
        # a simple, defensible estimator: midpoint of the overlap between the
        # accepted and rejected distributions
        crit = float(0.5 * (np.percentile(acc, 15) + np.percentile(rej, 85)))
    return {
        "note": ("gaps measured at the conflict point on the major stream; in a "
                 "dense stream one waiting driver rejects many sub-second gaps, "
                 "so the rejected pool is reported per driver as well as in "
                 "total"),
        "n_accepted": int(acc.size),
        "n_rejected": int(rej.size),
        "median_rejected_per_driver": (round(float(np.median(per_driver)), 1)
                                       if per_driver else None),
        "median_accepted_gap_s": round(float(np.median(acc)), 2),
        "p15_accepted_gap_s": round(float(np.percentile(acc, 15)), 2),
        "median_rejected_gap_s": (round(float(np.median(rej)), 2) if rej.size else None),
        "p85_rejected_gap_s": (round(float(np.percentile(rej, 85)), 2) if rej.size else None),
        "critical_gap_s": None if crit is None else round(crit, 2),
        "major_flow_veh_h": round(3600.0 / float(np.median(np.diff(mt))), 0),
    }


# ================================================ 4. desire lines vs geometry
def desire_deviation(df: pd.DataFrame, net, fps: float,
                     lane_width_m: float = 3.2,
                     straddle_band_m: float = 0.6) -> pd.DataFrame:
    """Where people actually drive against where the layout assumed they would.

    Two measurable symptoms. Lane straddling: a vehicle sitting astride a lane
    boundary rather than within a lane. Corner cutting: on a turn, taking a
    tighter path than the movement's own median line, which shows up as a
    consistently signed lateral offset rather than a symmetric spread.
    """
    rows = []
    for (o, d), line in net.corridors.items():
        tids = net.movements.index[(net.movements["origin"] == o) &
                                   (net.movements["destination"] == d)]
        sub = df[df["track_id"].isin(tids) & np.isfinite(df["world_x_m"])]
        if len(sub) < 50:
            continue
        s, off = project_to_centreline(sub["world_x_m"].to_numpy(),
                                       sub["world_y_m"].to_numpy(), line)
        ok = np.isfinite(off) & (np.abs(off) <= 14.0)
        off = off[ok]
        if off.size < 50:
            continue
        # distance from the nearest lane boundary
        # `frac` is the distance to the nearest lane BOUNDARY, in lane widths.
        # Straddling means sitting close to a boundary, so the test is frac
        # SMALL. Testing frac large measures the opposite -- vehicles neatly on
        # a lane centre -- and scored disciplined traffic as undisciplined.
        frac = np.abs((off / lane_width_m) - np.round(off / lane_width_m))
        straddle = float(np.mean(frac < (straddle_band_m / lane_width_m)))
        # Uniformly distributed traffic straddles this often by chance alone.
        # Without the baseline a raw share is uninterpretable: 2*band/width of
        # every lane is within the band whether drivers keep lane or not.
        baseline = min(1.0, 2.0 * straddle_band_m / lane_width_m)
        turn = net.movements.loc[list(tids), "turn"].mode()
        rows.append({
            "corridor": f"{o}->{d}",
            "turn": str(turn.iloc[0]) if len(turn) else "unknown",
            "samples": int(off.size),
            "median_offset_m": round(float(np.median(off)), 2),
            "offset_spread_m": round(float(np.percentile(off, 85) -
                                           np.percentile(off, 15)), 2),
            "straddle_share": round(straddle, 3),
            "straddle_expected_by_chance": round(baseline, 3),
            "lane_discipline": ("none" if straddle >= baseline * 0.95
                                else "partial" if straddle >= baseline * 0.6
                                else "good"),
            # a signed median well away from zero means the stream systematically
            # favours one side of its own centreline - corner cutting on a turn
            "systematic_bias": abs(float(np.median(off))) > 0.9,
        })
    return pd.DataFrame(rows)


# ======================================================= 5. obstruction census
def obstruction_census(df_all: pd.DataFrame, net, fps: float,
                       carriageway_m: float = 7.0,
                       min_dwell_s: float = 20.0) -> pd.DataFrame:
    """Stationary vehicles sitting ON the carriageway, with dwell duration.

    The parked set already contains everything that never moved. What makes a
    parked vehicle an OBSTRUCTION is where it stopped: inside the travelled way
    rather than beside it. Each parked track is projected onto every corridor
    and kept when it falls within the carriageway half-width.
    """
    parked = df_all[df_all["parked"]]
    if parked.empty:
        return pd.DataFrame()
    rows = []
    for tid, g in parked.groupby("track_id", sort=False):
        X = g["world_x_m"].to_numpy()
        Y = g["world_y_m"].to_numpy()
        if not np.isfinite(X).any():
            continue
        dwell = float((g["frame"].max() - g["frame"].min() + 1) / fps)
        if dwell < min_dwell_s:
            continue
        best = None
        for (o, d), line in net.corridors.items():
            s, off = project_to_centreline(X, Y, line)
            m = np.isfinite(off)
            if not m.any():
                continue
            mo = float(np.median(np.abs(off[m])))
            if best is None or mo < best[0]:
                best = (mo, f"{o}->{d}", float(np.median(s[m])))
        if best is None:
            continue
        lateral, corr, along = best
        if lateral > carriageway_m:
            continue                     # beside the road, not blocking it
        rows.append({
            "track_id": int(tid),
            "mode": str(g["mode"].iloc[0]),
            "corridor": corr,
            "distance_along_m": round(along, 1),
            "lateral_offset_m": round(lateral, 2),
            "dwell_s": round(dwell, 1),
            "start_s": round(float(g["frame"].min() / fps), 1),
            "obstruction": ("in-lane" if lateral <= carriageway_m * 0.5
                            else "kerbside"),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("dwell_s", ascending=False) if len(out) else out


# ================================================================== assemble
def analyse(df_all: pd.DataFrame, net, fps: float, top_corridors: int = 3) -> dict:
    """Run every L5 question over the busiest corridors."""
    moving = df_all[~df_all["parked"]]
    from .aggregate import od_summary
    od = od_summary(net)
    pairs = [(r.origin, r.destination) for r in od.itertuples()][:top_corridors]

    congestion, signals = {}, {}
    for (o, d) in pairs:
        e = tflow.edie(moving, net, o, d, fps)
        if len(e):
            congestion[f"{o}->{d}"] = congestion_origins(e)
        signals[f"{o}->{d}"] = signal_performance(moving, net, o, d, fps)

    gaps = {}
    if len(pairs) >= 2:
        major = pairs[0]
        minor = next(((o, d) for (o, d) in pairs[1:] if o != major[0]), None)
        if minor:
            gaps[f"{minor[0]}->{minor[1]} vs {major[0]}->{major[1]}"] = \
                gap_acceptance(moving, net, minor, major, fps)

    dev = desire_deviation(moving, net, fps)
    obs = obstruction_census(df_all, net, fps)
    lanes = tflow.assign_lanes(moving, net, fps)
    lc = tflow.lane_change_summary(moving, lanes, fps)

    return {
        "congestion": congestion,
        "signals": signals,
        "gap_acceptance": gaps,
        "desire_deviation": (dev.to_dict(orient="records") if len(dev) else []),
        "obstructions": (obs.to_dict(orient="records") if len(obs) else []),
        "lane_changes": (lc.to_dict(orient="records") if len(lc) else []),
    }
