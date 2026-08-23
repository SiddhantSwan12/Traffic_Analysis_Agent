"""Stronger estimators for the four L5 questions.

The first versions were made correct; these make them rigorous. Each replaces
an ad-hoc rule with the estimator the field already agrees on, and each reports
enough for a reader to disagree with it.

  1. shockwave speed   geometric edge-tracking  ->  Rankine-Hugoniot jump
  2. signal detection  gap thresholding         ->  periodicity + phase complementarity
  3. lane discipline   assumed 3.2 m grid       ->  lane structure inferred from behaviour
  4. critical gap      midpoint of percentiles  ->  Raff's method, per mode
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .reasoning import stopline_crossings


# ================================================ 1. Rankine-Hugoniot shockwave
def shockwave_rankine_hugoniot(edie: pd.DataFrame, congested_kmh: float = 15.0,
                               min_pairs: int = 4) -> dict:
    """Shockwave speed from the jump condition, not from tracking an edge.

    Tracking the geometric boundary of a congested region estimates how fast
    the *picture* changes. The physical quantity is the jump condition across
    the front:

        u_shock = (q2 - q1) / (k2 - k1)

    for flow q and density k on either side. It follows directly from
    conservation of vehicles, needs no edge to be located, and is immune to a
    single spurious cell shifting the boundary. Both estimates are returned so
    they can be compared: they should agree in sign and rough magnitude, and a
    disagreement is a signal that the congested region was misidentified.
    """
    if edie is None or not len(edie):
        return {"note": "no flow cells"}

    cell = (edie.groupby(["segment", "interval"])
                .agg(q=("flow_veh_per_h", "sum"),
                     k=("density_veh_per_km", "sum"),
                     v=("speed_kmh", "mean"),
                     seg_m=("segment_start_m", "first"),
                     t_s=("interval_start_s", "first"))
                .reset_index())
    lut = {(int(r.segment), int(r.interval)): r for r in cell.itertuples()}

    speeds, evidence = [], []
    for (sg, iv), r in lut.items():
        nxt = lut.get((sg + 1, iv))
        if nxt is None:
            continue
        a, b = r, nxt
        # a front exists only where one side is congested and the other is not
        if (a.v < congested_kmh) == (b.v < congested_kmh):
            continue
        dk = b.k - a.k
        if abs(dk) < 1e-6:
            continue
        u_kmh = (b.q - a.q) / dk          # veh/h per veh/km = km/h
        u = u_kmh / 3.6
        if not np.isfinite(u) or abs(u) > 30.0:
            continue                       # beyond any physical wave speed
        speeds.append(u)
        evidence.append({
            "at_m": round(float(a.seg_m), 1), "t_s": round(float(a.t_s), 1),
            "q_up": round(float(a.q), 0), "q_down": round(float(b.q), 0),
            "k_up": round(float(a.k), 1), "k_down": round(float(b.k), 1),
            "u_mps": round(float(u), 2),
        })
    if len(speeds) < min_pairs:
        return {"note": f"only {len(speeds)} usable fronts; not enough to estimate",
                "n_fronts": len(speeds)}
    arr = np.array(speeds)
    return {
        "note": "shockwave speed from the Rankine-Hugoniot jump condition",
        "n_fronts": int(arr.size),
        "median_mps": round(float(np.median(arr)), 2),
        "p25_mps": round(float(np.percentile(arr, 25)), 2),
        "p75_mps": round(float(np.percentile(arr, 75)), 2),
        "upstream_share": round(float(np.mean(arr < 0)), 2),
        "direction": ("upstream" if np.median(arr) < -0.2
                      else "downstream" if np.median(arr) > 0.2 else "stationary"),
        "evidence": sorted(evidence, key=lambda e: e["t_s"])[:12],
    }


# ============================================ 2. periodicity-based signal test
def signal_periodicity(df: pd.DataFrame, net, origin: str, destination: str,
                       fps: float, bin_s: float = 1.0,
                       min_cycle_s: float = 25.0, max_cycle_s: float = 180.0,
                       min_prominence: float = 0.25) -> dict:
    """Detect a signal from periodicity in the discharge, not from gap sizes.

    A gap threshold asks "is there a long pause?", which light traffic answers
    yes to by accident. The defining property of a signal is that the pauses
    RECUR at a fixed interval, so the right test is periodicity: bin the
    stop-line crossings into a time series and autocorrelate it. A signal puts a
    sharp peak at the cycle length; unsignalised traffic produces none.

    This also handles actuated signals, whose cycle length varies a little and
    which a strict fixed-cycle test would reject: they still give a peak, only
    broader.
    """
    cr = stopline_crossings(df, net, origin, destination, fps)
    if len(cr) < 20:
        return {"signalised": None, "method": "autocorrelation",
                "note": "too few crossings to test for periodicity",
                "crossings": int(len(cr))}

    t = cr["t_s"].to_numpy()
    span = float(t.max() - t.min())
    if span < 3 * min_cycle_s:
        return {"signalised": None, "method": "autocorrelation",
                "note": "clip too short to contain several cycles",
                "span_s": round(span, 1)}

    nb = int(span / bin_s) + 1
    counts = np.histogram(t, bins=nb, range=(t.min(), t.min() + nb * bin_s))[0]
    x = counts.astype(float) - counts.mean()
    if not np.any(x):
        return {"signalised": False, "method": "autocorrelation",
                "note": "no variation in the crossing rate"}

    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    lo = int(min_cycle_s / bin_s)
    hi = min(len(ac) - 1, int(max_cycle_s / bin_s))
    if hi <= lo + 2:
        return {"signalised": None, "method": "autocorrelation",
                "note": "search window empty"}
    band = ac[lo:hi]
    k = int(np.argmax(band))
    peak = float(band[k])
    cycle = (lo + k) * bin_s
    # prominence against the local background, so a slowly decaying
    # autocorrelation is not mistaken for a periodic peak
    prominence = peak - float(np.median(band))

    signalised = bool(prominence >= min_prominence and peak > 0.2)
    return {
        "signalised": signalised,
        "method": "autocorrelation of stop-line crossing rate",
        "note": ("a recurring cycle is present in the discharge"
                 if signalised else
                 "no recurring cycle in the discharge; pauses are not periodic"),
        "crossings": int(len(cr)),
        "cycle_length_s": round(cycle, 1) if signalised else None,
        "peak_autocorrelation": round(peak, 3),
        "peak_prominence": round(prominence, 3),
        "prominence_threshold": min_prominence,
    }


def signal_complementarity(df: pd.DataFrame, net, pairs: list, fps: float,
                           bin_s: float = 2.0) -> dict:
    """Do the approaches take turns?

    This is the strongest single discriminator available, and it needs no
    assumption about cycle length. A signal enforces that conflicting approaches
    discharge at different times, so their crossing-rate series are negatively
    correlated. An uncontrolled junction has them independent, since each
    approach simply flows when it has traffic.
    """
    series, names = [], []
    for (o, d) in pairs:
        cr = stopline_crossings(df, net, o, d, fps)
        if len(cr) < 15:
            continue
        series.append(cr["t_s"].to_numpy())
        names.append(f"{o}->{d}")
    if len(series) < 2:
        return {"note": "need at least two approaches with traffic"}

    lo = min(s.min() for s in series)
    hi = max(s.max() for s in series)
    nb = max(4, int((hi - lo) / bin_s))
    mats = [np.histogram(s, bins=nb, range=(lo, hi))[0].astype(float) for s in series]

    out = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            a, b = mats[i], mats[j]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            out.append({"pair": f"{names[i]} vs {names[j]}",
                        "correlation": round(r, 3),
                        "interpretation": ("alternating (signal-like)" if r < -0.15
                                           else "independent (uncontrolled)"
                                           if abs(r) <= 0.15 else "simultaneous")})
    if not out:
        return {"note": "no comparable approach pairs"}
    med = float(np.median([o["correlation"] for o in out]))
    return {
        "note": ("conflicting approaches discharge at different times, as a "
                 "signal enforces" if med < -0.15 else
                 "approaches discharge independently, as at an uncontrolled "
                 "junction"),
        "median_correlation": round(med, 3),
        "signal_like": bool(med < -0.15),
        "pairs": out,
    }


# ================================================ 3. observed lane structure
def lane_structure(df: pd.DataFrame, net, lane_width_hint: float = 3.2,
                   max_offset_m: float = 10.0, bw_m: float = 0.35) -> pd.DataFrame:
    """Infer whether lane structure exists at all, from where traffic sits.

    Measuring straddling against an assumed grid answers the wrong question: it
    presumes lanes are where a 3.2 m ruler says they are. What matters is
    whether the lateral distribution has MODES at all. Clear peaks mean drivers
    are keeping to lanes, wherever those lanes happen to be; a flat distribution
    means there is no lane behaviour to keep to.

    Reported as a modality index: the amplitude of the smoothed lateral density
    relative to its mean. Zero means uniform, high means sharply lane-structured.
    """
    from .aggregate import project_to_centreline

    rows = []
    for (o, d), line in net.corridors.items():
        tids = net.movements.index[(net.movements["origin"] == o) &
                                   (net.movements["destination"] == d)]
        sub = df[df["track_id"].isin(tids) & np.isfinite(df["world_x_m"])]
        if len(sub) < 200:
            continue
        _, off = project_to_centreline(sub["world_x_m"].to_numpy(),
                                       sub["world_y_m"].to_numpy(), line)
        off = off[np.isfinite(off) & (np.abs(off) <= max_offset_m)]
        if off.size < 200:
            continue

        grid = np.arange(-max_offset_m, max_offset_m + bw_m, bw_m)
        # Gaussian kernel density, computed directly to avoid a scipy dependency
        dens = np.exp(-0.5 * ((grid[:, None] - off[None, :]) / bw_m) ** 2).sum(1)
        dens /= dens.sum() * bw_m

        # peaks that stand clear of their neighbourhood
        peaks = []
        for i in range(2, len(dens) - 2):
            w = dens[max(0, i - 4):i + 5]
            if dens[i] == w.max() and dens[i] > dens.mean() * 1.15:
                peaks.append(float(grid[i]))
        merged = []
        for p in peaks:
            if not merged or abs(p - merged[-1]) > lane_width_hint * 0.55:
                merged.append(p)

        modality = float((dens.max() - dens.min()) / max(dens.mean(), 1e-9))
        spacing = (float(np.median(np.diff(merged))) if len(merged) > 1 else None)
        # A high modality index with ONE mode means traffic is concentrated in a
        # single band -- which is the road, not a lane. Lane structure requires
        # several modes separated by roughly a lane width; anything else is one
        # undifferentiated stream, however sharply peaked it is.
        lane_like = (len(merged) >= 2 and spacing is not None
                     and 2.4 <= spacing <= 4.2)
        rows.append({
            "corridor": f"{o}->{d}",
            "samples": int(off.size),
            "modality_index": round(modality, 2),
            "n_modes": len(merged),
            "mode_offsets_m": [round(m, 2) for m in merged],
            "observed_lane_spacing_m": (None if spacing is None else round(spacing, 2)),
            "lane_like_spacing": bool(lane_like),
            # concentration alone is not lane structure; separation into bands a
            # lane apart is
            "lane_structure": ("clear" if lane_like and modality >= 2.2 else
                               "weak" if lane_like else
                               "single undifferentiated stream" if modality >= 2.2
                               else "absent"),
        })
    return pd.DataFrame(rows)


# ==================================================== 4. Raff's critical gap
def raff_critical_gap(accepted: np.ndarray, rejected: np.ndarray,
                      grid_s: np.ndarray | None = None) -> dict:
    """Critical gap by Raff's method.

    The standard graphical estimator, and the reason it is standard is that it
    needs no distributional assumption. Plot the cumulative share of ACCEPTED
    gaps at most t against the cumulative share of REJECTED gaps greater than t;
    the value of t where the two curves cross is the critical gap. Below it most
    drivers refuse, above it most accept.

    Replaces a midpoint of two percentiles, which had no basis beyond looking
    reasonable.
    """
    a = np.asarray(accepted, float)
    r = np.asarray(rejected, float)
    a = a[np.isfinite(a) & (a > 0)]
    r = r[np.isfinite(r) & (r > 0)]
    if a.size < 5 or r.size < 5:
        return {"critical_gap_s": None,
                "note": "too few accepted or rejected gaps for Raff's method",
                "n_accepted": int(a.size), "n_rejected": int(r.size)}

    if grid_s is None:
        hi = float(min(np.percentile(a, 99), 20.0))
        grid_s = np.linspace(0.1, max(hi, 2.0), 400)
    Fa = np.array([(a <= t).mean() for t in grid_s])      # accepted <= t
    Fr = np.array([(r > t).mean() for t in grid_s])       # rejected > t
    d = Fa - Fr
    sign = np.sign(d)
    # A crossing only means something where BOTH curves are actually present.
    # Disjoint distributions (every rejection tiny, every acceptance large)
    # leave a stretch where Fa and Fr are both zero; the difference touches zero
    # there and a naive sign test reads it as a crossing, returning a critical
    # gap for populations that never overlap at all.
    live = (Fa > 0.02) & (Fr > 0.02)
    flip = (np.diff(sign) != 0) & live[:-1] & live[1:]
    cross = np.flatnonzero(flip)
    if cross.size == 0:
        return {"critical_gap_s": None,
                "note": ("accepted and rejected gap distributions do not "
                         "overlap, so no critical gap can be identified"),
                "n_accepted": int(a.size), "n_rejected": int(r.size),
                "median_accepted_s": round(float(np.median(a)), 2),
                "median_rejected_s": round(float(np.median(r)), 2)}
    i = int(cross[0])
    # linear interpolation onto the exact crossing
    t0, t1 = grid_s[i], grid_s[i + 1]
    d0, d1 = d[i], d[i + 1]
    tc = float(t0 - d0 * (t1 - t0) / (d1 - d0)) if d1 != d0 else float(t0)
    return {
        "critical_gap_s": round(tc, 2),
        "method": "Raff",
        "note": "crossing of accepted-below-t and rejected-above-t curves",
        "n_accepted": int(a.size), "n_rejected": int(r.size),
        "median_accepted_s": round(float(np.median(a)), 2),
        "median_rejected_s": round(float(np.median(r)), 2),
    }


def gap_acceptance_by_mode(df: pd.DataFrame, net, minor: tuple, major: tuple,
                           fps: float, wait_speed_kmh: float = 6.0) -> dict:
    """Critical gap per transport mode, with lags separated from gaps.

    Two refinements that matter in mixed traffic. A motorcycle accepts a far
    smaller gap than a car, so one pooled critical gap describes nobody. And the
    FIRST interval a driver sees on arrival is a lag, not a gap -- the driver
    did not watch it open -- so pooling lags with gaps biases the estimate.
    """
    maj = stopline_crossings(df, net, major[0], major[1], fps)
    mnr = stopline_crossings(df, net, minor[0], minor[1], fps)
    if len(maj) < 10 or len(mnr) < 5:
        return {"note": "not enough conflicting traffic",
                "major": int(len(maj)), "minor": int(len(mnr))}
    mt = maj["t_s"].to_numpy()
    gaps = np.diff(mt)

    waits = {}
    sub = df[df["track_id"].isin(mnr["track_id"])]
    for tid, g in sub.groupby("track_id", sort=False):
        g = g.sort_values("frame")
        sp = g["speed_kmh"].to_numpy()
        slow = sp < wait_speed_kmh
        if slow.any():
            waits[int(tid)] = float(g["frame"].to_numpy()[int(np.argmax(slow))] / fps)

    by_mode = {}
    lags = []
    for r in mnr.itertuples():
        i = int(np.searchsorted(mt, r.t_s))
        if i == 0 or i >= len(mt):
            continue
        acc = float(mt[i] - mt[i - 1])
        w = waits.get(int(r.track_id))
        rej = []
        if w is not None and w < r.t_s:
            rej = [float(gaps[j]) for j in range(len(gaps))
                   if mt[j] >= w and mt[j + 1] <= r.t_s]
        else:
            # took the very first interval it saw: that is a lag
            lags.append(acc)
        e = by_mode.setdefault(r.mode, {"accepted": [], "rejected": []})
        e["accepted"].append(acc)
        e["rejected"].extend(rej)

    out = {"by_mode": {}, "n_lags": len(lags),
           "median_lag_s": (round(float(np.median(lags)), 2) if lags else None),
           "note": ("critical gap estimated by Raff's method per mode; the first "
                    "interval a driver sees on arrival is counted as a lag, not "
                    "a gap")}
    for m, e in by_mode.items():
        if len(e["accepted"]) < 5:
            continue
        out["by_mode"][m] = raff_critical_gap(np.array(e["accepted"]),
                                              np.array(e["rejected"]))
    allacc = np.concatenate([np.array(e["accepted"]) for e in by_mode.values()]) \
        if by_mode else np.array([])
    allrej = np.concatenate([np.array(e["rejected"]) for e in by_mode.values()]) \
        if by_mode else np.array([])
    out["pooled"] = raff_critical_gap(allacc, allrej)
    return out
