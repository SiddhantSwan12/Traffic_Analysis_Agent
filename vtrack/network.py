"""L4b: recover the road network from the trajectories themselves.

No external map is used. The geometry is inferred from where vehicles actually
went, which matters for two reasons: it works anywhere, with no dependency on
OSM coverage or accuracy; and L5's desire-line analysis needs precisely this
"observed" layout to compare against the "assumed" one.

The inference proceeds:

    endpoints  ->  gates       where traffic enters and leaves the scene
    gates      ->  approaches  a gate plus the bearing traffic uses there
    tracks     ->  movements   (origin gate, destination gate) per vehicle
    approach   ->  lanes       lateral clustering across the carriageway
    movements  ->  corridors   a centreline per origin-destination pair

Every stage is clustering with an explicit metric threshold rather than a fixed
cluster count, so a scene with three approaches and a scene with six both work
without retuning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# --------------------------------------------------------------------- helpers
def circular_mean_deg(a) -> float:
    a = np.radians(np.asarray(a, float))
    return float(np.degrees(math.atan2(float(np.mean(np.sin(a))),
                                       float(np.mean(np.cos(a))))) % 360.0)


def angdiff(a, b):
    """Smallest signed difference a-b, in degrees, wrapped to [-180, 180)."""
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


def _cluster_points(P: np.ndarray, threshold_m: float) -> np.ndarray:
    """Average-linkage clustering with a metric cut. Returns 0-based labels."""
    if len(P) == 0:
        return np.zeros(0, int)
    if len(P) == 1:
        return np.zeros(1, int)
    Z = linkage(pdist(P), method="average")
    return fcluster(Z, t=threshold_m, criterion="distance") - 1


@dataclass
class Gate:
    """A place on the scene boundary where traffic enters or leaves."""
    id: int
    x: float
    y: float
    bearing_in: float | None     # local-frame bearing of traffic ENTERING here
    bearing_out: float | None    # ... and LEAVING here
    n_in: int
    n_out: int
    name: str = ""
    lanes: list = field(default_factory=list)

    def as_dict(self):
        return {"id": self.id, "name": self.name,
                "x_m": round(self.x, 2), "y_m": round(self.y, 2),
                "bearing_in_deg": None if self.bearing_in is None else round(self.bearing_in, 1),
                "bearing_out_deg": None if self.bearing_out is None else round(self.bearing_out, 1),
                "entering": self.n_in, "leaving": self.n_out,
                "n_lanes": len(self.lanes)}


@dataclass
class RoadNetwork:
    gates: list
    movements: pd.DataFrame        # per-track origin/destination assignment
    od: pd.DataFrame               # origin-destination count matrix
    corridors: dict                # (o, d) -> centreline as an (N, 2) array
    min_travel_m: float

    def gate(self, gid):
        return next(g for g in self.gates if g.id == gid)


# ----------------------------------------------------------------------- gates
def _endpoint_table(df: pd.DataFrame, min_travel_m: float,
                    frame_wh: tuple | None = None,
                    edge_margin_px: float = 45.0) -> pd.DataFrame:
    """First and last observed state of every track.

    `on_edge_in` / `on_edge_out` record whether the endpoint sits against the
    image border. That distinction is essential: a track beginning in the
    middle of the frame did not enter the scene through an approach, it simply
    became visible there (occlusion recovery, or a parked vehicle pulling out).
    Treating those as gate crossings invents approaches that do not exist.
    """
    g = df.sort_values("frame").groupby("track_id")
    first = g.first()
    last = g.last()
    out = pd.DataFrame({
        "x0": first["world_x_m"], "y0": first["world_y_m"],
        "x1": last["world_x_m"], "y1": last["world_y_m"],
        "f0": first["frame"], "f1": last["frame"],
        "mode": first["mode"], "display_id": first["display_id"],
    })
    if "cx" in first.columns:
        out["px0"], out["py0"] = first["cx"], first["cy"]
        out["px1"], out["py1"] = last["cx"], last["cy"]
    out = out.dropna(subset=["x0", "y0", "x1", "y1"])
    out["travel_m"] = np.hypot(out["x1"] - out["x0"], out["y1"] - out["y0"])

    if frame_wh is not None and "px0" in out.columns:
        W, H = frame_wh
        m = edge_margin_px
        def on_edge(px, py):
            return ((px <= m) | (px >= W - 1 - m) | (py <= m) | (py >= H - 1 - m))
        out["on_edge_in"] = on_edge(out["px0"].to_numpy(), out["py0"].to_numpy())
        out["on_edge_out"] = on_edge(out["px1"].to_numpy(), out["py1"].to_numpy())
    else:
        out["on_edge_in"] = True
        out["on_edge_out"] = True
    # bearing of the first and last few metres, which is what defines an approach
    b_in, b_out = [], []
    for tid in out.index:
        t = df[df["track_id"] == tid].sort_values("frame")
        X = t["world_x_m"].to_numpy(); Y = t["world_y_m"].to_numpy()
        ok = np.isfinite(X) & np.isfinite(Y)
        X, Y = X[ok], Y[ok]
        k = max(2, min(len(X) // 4, 45))
        b_in.append(math.degrees(math.atan2(X[k - 1] - X[0], Y[k - 1] - Y[0])) % 360
                    if len(X) >= k else np.nan)
        b_out.append(math.degrees(math.atan2(X[-1] - X[-k], Y[-1] - Y[-k])) % 360
                     if len(X) >= k else np.nan)
    out["bearing_in"] = b_in
    out["bearing_out"] = b_out
    return out[out["travel_m"] >= min_travel_m]


def infer_gates(df: pd.DataFrame, min_travel_m: float = 25.0,
                cluster_m: float = 25.0, min_tracks: int = 12,
                frame_wh: tuple | None = None) -> tuple[list, pd.DataFrame]:
    """Cluster trajectory endpoints into entry/exit gates.

    Entry and exit points are clustered together, not separately: a road is
    normally both, and forcing them into one gate is what makes the
    origin-destination matrix square and interpretable.
    """
    ep = _endpoint_table(df, min_travel_m, frame_wh)
    if ep.empty:
        return [], ep

    # Only endpoints on the image border define approaches.
    pts_all = np.vstack([ep[["x0", "y0"]].to_numpy(), ep[["x1", "y1"]].to_numpy()])
    kind_all = np.r_[np.zeros(len(ep), int), np.ones(len(ep), int)]
    edge_all = np.r_[ep["on_edge_in"].to_numpy(), ep["on_edge_out"].to_numpy()]
    sel = np.flatnonzero(edge_all)
    if sel.size < min_tracks:
        return [], ep
    pts = pts_all[sel]
    kind = kind_all[sel]
    lab_sel = _cluster_points(pts, cluster_m)
    lab = np.full(len(pts_all), -1, int)
    lab[sel] = lab_sel

    gates, remap = [], {}
    for c in np.unique(lab_sel):
        m = lab == c
        n_in = int((m[:len(ep)]).sum())
        n_out = int((m[len(ep):]).sum())
        if n_in + n_out < min_tracks:
            continue
        gi = np.flatnonzero(lab == c)                 # indices into pts_all
        ein = gi[gi < len(ep)]
        eout = gi[gi >= len(ep)] - len(ep)
        bi = circular_mean_deg(ep["bearing_in"].to_numpy()[ein]) if len(ein) else None
        bo = circular_mean_deg(ep["bearing_out"].to_numpy()[eout]) if len(eout) else None
        remap[c] = len(gates)
        gates.append(Gate(id=len(gates), x=float(pts_all[m, 0].mean()),
                          y=float(pts_all[m, 1].mean()),
                          bearing_in=bi, bearing_out=bo,
                          n_in=n_in, n_out=n_out))

    ep = ep.copy()
    ep["gate_in"] = [remap.get(l, -1) for l in lab[:len(ep)]]
    ep["gate_out"] = [remap.get(l, -1) for l in lab[len(ep):]]
    return gates, ep


def name_gates(gates: list, geo=None) -> None:
    """Label gates by the compass direction traffic arrives FROM.

    A gate where vehicles enter heading east is the *west* approach, which is
    how a traffic engineer would name it, so the inbound bearing is reversed
    before naming.
    """
    COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
               "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    for g in gates:
        b = g.bearing_in if g.bearing_in is not None else g.bearing_out
        if b is None:
            g.name = f"G{g.id}"
            continue
        if geo is not None:
            b = float(geo.local_bearing_to_compass(b))
        from_b = (b + 180.0) % 360.0       # direction traffic comes FROM
        g.name = f"{COMPASS[int((from_b + 11.25) % 360 // 22.5)]}"
    # disambiguate duplicates
    seen = {}
    for g in gates:
        seen[g.name] = seen.get(g.name, 0) + 1
        if seen[g.name] > 1:
            g.name = f"{g.name}{seen[g.name]}"


# ------------------------------------------------------------------- movements
def _turn_type(b_in: float, b_out: float) -> str:
    """Classify a movement from the change in heading."""
    d = float(angdiff(b_out, b_in))
    if abs(d) <= 35:
        return "through"
    if abs(d) >= 145:
        return "u-turn"
    return "right" if d > 0 else "left"


def assign_movements(ep: pd.DataFrame, gates: list) -> pd.DataFrame:
    mv = ep[(ep["gate_in"] >= 0) & (ep["gate_out"] >= 0)].copy()
    by_id = {g.id: g for g in gates}
    mv["origin"] = [by_id[i].name for i in mv["gate_in"]]
    mv["destination"] = [by_id[i].name for i in mv["gate_out"]]
    mv["turn"] = [
        _turn_type(bi, bo) if np.isfinite(bi) and np.isfinite(bo) else "unknown"
        for bi, bo in zip(mv["bearing_in"], mv["bearing_out"])]
    # a movement that starts and ends at one gate without a u-turn heading is
    # almost always a track that never really left: mark it, do not count it
    same = mv["gate_in"] == mv["gate_out"]
    mv.loc[same & (mv["turn"] != "u-turn"), "turn"] = "internal"
    return mv


def od_matrix(mv: pd.DataFrame) -> pd.DataFrame:
    real = mv[mv["turn"] != "internal"]
    return (real.groupby(["origin", "destination"]).size()
                .unstack(fill_value=0).sort_index(axis=0).sort_index(axis=1))


# ----------------------------------------------------------------------- lanes
def infer_lanes(df: pd.DataFrame, ep: pd.DataFrame, gate: Gate,
                window_m: float = 30.0, lane_width_m: float = 3.2,
                min_tracks: int = 5) -> list:
    """Cluster lateral offsets near a gate into lanes.

    Positions within `window_m` of the gate are projected onto the axis
    perpendicular to the approach bearing. Vehicles keeping lane produce
    distinct modes in that 1-D distribution; the clustering threshold is set to
    a little under one lane width so adjacent lanes stay separable.
    """
    ids = ep.index[(ep["gate_in"] == gate.id)]
    if len(ids) < min_tracks:
        return []
    b = gate.bearing_in if gate.bearing_in is not None else gate.bearing_out
    if b is None:
        return []
    th = math.radians(b)
    # unit vector across the approach (perpendicular to direction of travel)
    px, py = math.cos(th), -math.sin(th)

    offs, per_track = [], {}
    sub = df[df["track_id"].isin(ids)]
    for tid, g in sub.groupby("track_id"):
        X = g["world_x_m"].to_numpy(); Y = g["world_y_m"].to_numpy()
        d = np.hypot(X - gate.x, Y - gate.y)
        m = np.isfinite(d) & (d <= window_m)
        if m.sum() < 3:
            continue
        o = float(np.median((X[m] - gate.x) * px + (Y[m] - gate.y) * py))
        per_track[tid] = o
        offs.append(o)
    if len(offs) < min_tracks:
        return []

    arr = np.array(offs).reshape(-1, 1)
    lab = _cluster_points(arr, lane_width_m * 0.75)
    lanes = []
    for c in np.unique(lab):
        m = lab == c
        if m.sum() < min_tracks:
            continue
        lanes.append({"offset_m": float(arr[m, 0].mean()),
                      "n_tracks": int(m.sum()),
                      "spread_m": float(arr[m, 0].std())})
    lanes.sort(key=lambda l: l["offset_m"])
    for i, l in enumerate(lanes):
        l["lane"] = i + 1
    # map every track to its lane index
    if lanes:
        centres = np.array([l["offset_m"] for l in lanes])
        for tid, o in per_track.items():
            per_track[tid] = int(np.argmin(np.abs(centres - o))) + 1
    return [lanes, per_track]


# ------------------------------------------------------------------- corridors
def movement_centreline(df: pd.DataFrame, tids, n_nodes: int = 24):
    """Median path of a set of trajectories, resampled to a common length.

    Each trajectory is resampled by arc length so that vehicles which paused do
    not drag the median toward wherever they stopped; the result is a geometric
    centreline, not a time average.
    """
    paths = []
    for tid in tids:
        g = df[df["track_id"] == tid].sort_values("frame")
        X = g["world_x_m"].to_numpy(); Y = g["world_y_m"].to_numpy()
        ok = np.isfinite(X) & np.isfinite(Y)
        X, Y = X[ok], Y[ok]
        if X.size < 4:
            continue
        s = np.r_[0, np.cumsum(np.hypot(np.diff(X), np.diff(Y)))]
        if s[-1] < 1e-6:
            continue
        u = np.linspace(0, s[-1], n_nodes)
        paths.append(np.column_stack([np.interp(u, s, X), np.interp(u, s, Y)]))
    if len(paths) < 3:
        return None
    return np.median(np.stack(paths), axis=0)


def build(df: pd.DataFrame, geo=None, min_travel_m: float = 25.0,
          cluster_m: float = 25.0, lane_width_m: float = 3.2,
          frame_wh: tuple | None = None) -> RoadNetwork:
    gates, ep = infer_gates(df, min_travel_m, cluster_m, frame_wh=frame_wh)
    name_gates(gates, geo)
    mv = assign_movements(ep, gates)

    for g in gates:
        res = infer_lanes(df, ep, g, lane_width_m=lane_width_m)
        if res:
            g.lanes = res[0]
            for tid, ln in res[1].items():
                if tid in mv.index:
                    mv.loc[tid, "lane_in"] = ln

    corridors = {}
    real = mv[mv["turn"] != "internal"]
    for (o, d), grp in real.groupby(["origin", "destination"]):
        if len(grp) < 5:
            continue
        c = movement_centreline(df, grp.index)
        if c is not None:
            corridors[(o, d)] = c

    return RoadNetwork(gates=gates, movements=mv, od=od_matrix(mv),
                       corridors=corridors, min_travel_m=min_travel_m)


def describe(net: RoadNetwork) -> dict:
    return {
        "gates": [g.as_dict() for g in net.gates],
        "n_movements_assigned": int(len(net.movements)),
        "n_corridors": len(net.corridors),
        "min_travel_m": net.min_travel_m,
        "turn_counts": net.movements["turn"].value_counts().to_dict(),
    }
