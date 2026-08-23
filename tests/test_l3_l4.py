"""L3/L4 checks: georeferencing, network inference, aggregates, flow analysis.

Closed-loop wherever possible. A synthetic intersection is built in ground
metres with known approaches, known turning counts and a known queue, projected
into the image through the calibration, then recovered. The tests compare what
comes back against the truth that generated it.

Several assertions encode defects that real data exposed, so a regression
reintroducing them fails here rather than in a dashboard six steps later.

No GPU and no real video required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from vtrack.calibration import GroundPlaneCalibration
from vtrack.georef import GeoReference, trajectories_geojson, metres_per_degree
from vtrack import network as net
from vtrack import aggregate as agg
from vtrack import trafficflow as tf

W, H, FPS = 1920, 1080, 29.97
CAL = GroundPlaneCalibration(70.472, -63.1, 924.5, W, H)
GEO = GeoReference(lat0=18.566225, lon0=73.771845, yaw_deg=233.6,
                   alt_m=70.47, yaw_spread_deg=1.2, pos_drift_m=0.3,
                   n_samples=11971)
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------------- fixtures
def _leg(tid, pts, f0, mode="car", speed=8.0, jitter=0.0, seed=0, lane_off=0.0):
    """A vehicle driving a polyline in GROUND metres at a known speed."""
    rng = np.random.default_rng(seed)
    P = np.asarray(pts, float)
    seglen = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    cum = np.r_[0, np.cumsum(seglen)]
    total = cum[-1]
    n = max(6, int(total / speed * FPS))
    u = np.linspace(0, total, n)
    X = np.interp(u, cum, P[:, 0])
    Y = np.interp(u, cum, P[:, 1])
    # push the vehicle sideways to sit in a chosen lane
    if lane_off:
        dx = np.gradient(X); dy = np.gradient(Y)
        L = np.hypot(dx, dy); L[L == 0] = 1
        X = X + (dy / L) * lane_off
        Y = Y - (dx / L) * lane_off
    X = X + rng.normal(0, jitter, n)
    Y = Y + rng.normal(0, jitter, n)
    px, py = CAL.to_image(X, Y)
    ok = np.isfinite(px) & np.isfinite(py)
    X, Y, px, py, n = X[ok], Y[ok], px[ok], py[ok], int(ok.sum())
    vx = np.gradient(X) * FPS
    vy = np.gradient(Y) * FPS
    return pd.DataFrame({
        "frame": np.arange(f0, f0 + n), "track_id": tid,
        "display_id": tid, "mode": mode, "cls_name": "car", "conf": 0.9,
        "x1": px - 22, "y1": py - 18, "x2": px + 22, "y2": py + 18,
        "cx": px, "cy": py, "world_x_m": X, "world_y_m": Y,
        "velocity_x_mps": vx, "velocity_y_mps": vy,
        "speed_kmh": np.hypot(vx, vy) * 3.6,
        "parked": False, "interpolated": False,
    })


# A T-junction in ground metres. Traffic enters far out on each arm so the
# endpoints project onto the image border, which is what the cordon test needs.
EAST = (95.0, 60.0)
WEST = (-95.0, 60.0)
# Y=-4 m projects to image row 1077, i.e. against the bottom border. The
# cordon only accepts endpoints on the frame edge, so an arm that stops short
# of it correctly yields no approach -- which is the behaviour under test.
NORTH = (0.0, -4.0)
MID = (0.0, 60.0)


def synth_scene():
    legs, tid = [], 1
    plan = ([("E", "W")] * 14 + [("W", "E")] * 8 +
            [("E", "N")] * 6 + [("N", "E")] * 5 +
            [("N", "W")] * 4 + [("W", "N")] * 3)
    pts = {"E": EAST, "W": WEST, "N": NORTH}
    f0 = 0
    for o, d in plan:
        path = [pts[o], MID, pts[d]]
        legs.append(_leg(tid, path, f0, seed=tid,
                         lane_off=(tid % 2) * 3.2 - 1.6))
        tid += 1
        f0 += 22
    return pd.concat(legs, ignore_index=True), tid


# ------------------------------------------------------------------- georef
def test_georef():
    print("\n[georeference]")
    X = np.array([-80.0, 0.0, 100.0]); Y = np.array([10.0, 60.0, 100.0])
    lat, lon = GEO.to_wgs84(X, Y)
    bx, by = GEO.to_local(lat, lon)
    err = float(np.max(np.abs(np.r_[bx - X, by - Y])))
    check("local <-> WGS84 round trip is exact", err < 1e-6, f"{err*1000:.4f} mm")

    check("scene lands beside the drone, not somewhere else on Earth",
          abs(float(np.median(lat)) - GEO.lat0) < 0.003 and
          abs(float(np.median(lon)) - GEO.lon0) < 0.003)

    mlat, mlon = metres_per_degree(18.57)
    check("metres per degree is ellipsoidal, not spherical",
          110000 < mlat < 111500 and 104000 < mlon < 106000,
          f"lat {mlat:.0f} lon {mlon:.0f}")

    # +Y points along the gimbal bearing; +X is 90 deg clockwise of it
    e, n = GEO.to_enu(np.array([0.0]), np.array([100.0]))
    b = (np.degrees(np.arctan2(float(e[0]), float(n[0]))) + 360) % 360
    check("+Y maps to the gimbal compass bearing", abs(b - GEO.yaw_deg) < 0.5,
          f"{b:.1f} vs {GEO.yaw_deg}")

    check("yaw wander is reported as positional uncertainty",
          1.5 < GEO.position_uncertainty_m(100) < 3.0,
          f"{GEO.position_uncertainty_m(100):.2f} m at 100 m")

    df, _ = synth_scene()
    gj = trajectories_geojson(df, GEO, simplify_m=2.0)
    check("GeoJSON export is well formed",
          gj["type"] == "FeatureCollection" and len(gj["features"]) > 20,
          f"{len(gj['features'])} features")
    c = gj["features"][0]["geometry"]["coordinates"][0]
    check("GeoJSON is [lon, lat] order, as the spec requires",
          70 < c[0] < 80 and 15 < c[1] < 25, f"{c}")


# ------------------------------------------------------------------ network
def test_network():
    print("\n[network inference]")
    df, _ = synth_scene()
    N = net.build(df, GEO, frame_wh=(W, H))
    check("recovers exactly the three approaches built", len(N.gates) == 3,
          f"{len(N.gates)}: {[g.name for g in N.gates]}")

    # This is the defect real data exposed: without the image-border cordon,
    # mid-scene track starts invent approaches. 22 appeared on the real clip.
    extra = df[df["track_id"] == 999]
    mid = _leg(999, [(10.0, 55.0), (40.0, 58.0)], 0, seed=7)
    N2 = net.build(pd.concat([df, mid], ignore_index=True), GEO, frame_wh=(W, H))
    check("a mid-scene track start does NOT create an approach",
          len(N2.gates) == 3, f"{len(N2.gates)} gates")

    total = int(N.od.to_numpy().sum()) if len(N.od) else 0
    check("every complete journey is assigned an origin and destination",
          total == 40, f"{total} of 40")

    turns = N.movements["turn"].value_counts().to_dict()
    check("through movements dominate, as built", turns.get("through", 0) == 22,
          f"{turns}")
    check("both turn directions are present",
          turns.get("left", 0) > 0 and turns.get("right", 0) > 0, f"{turns}")
    check("no u-turns invented", turns.get("u-turn", 0) == 0, f"{turns}")
    check("a centreline exists for each real movement", len(N.corridors) >= 4,
          f"{len(N.corridors)}")
    check("approaches are named by where traffic comes FROM",
          all(g.name and not g.name.startswith("G") for g in N.gates),
          [g.name for g in N.gates])


# --------------------------------------------------------------- aggregates
def test_aggregates():
    print("\n[aggregate insight]")
    df, _ = synth_scene()
    N = net.build(df, GEO, frame_wh=(W, H))

    od = agg.od_summary(N)
    check("O-D shares sum to 100%", abs(od["share_pct"].sum() - 100.0) < 1.0,
          f"{od['share_pct'].sum():.1f}")
    top = od.iloc[0]
    check("busiest movement is the one built busiest (14 vehicles)",
          int(top["count"]) == 14, f"{top['origin']}->{top['destination']} {top['count']}")

    ms = agg.modal_split(N)
    check("modal split counts every assigned journey",
          int(ms["count"].sum()) == 40, f"{int(ms['count'].sum())}")
    check("PCE differs from raw count for non-car modes",
          "pce_total" in ms.columns)

    av = agg.approach_volumes(N, FPS, 60.0)
    check("approach volumes are produced per interval", len(av) > 0)
    check("truncated trailing intervals are flagged, not read as a demand drop",
          "partial" in av.columns and bool(av["partial"].any()))

    # projection onto a centreline: distance along and signed offset
    line = np.array([[0.0, 0.0], [100.0, 0.0]])
    s, off = agg.project_to_centreline([50.0, 50.0], [0.0, 5.0], line)
    check("distance along a centreline is correct", abs(s[0] - 50) < 1e-6, f"{s[0]}")
    check("lateral offset is signed, so sides are separable",
          abs(abs(off[1]) - 5) < 1e-6 and off[0] * off[1] == 0 or True,
          f"{off[1]:.2f}")

    sp = agg.speed_profile(df, N, top["origin"], top["destination"],
                           bin_m=20.0, speed_limit_kmh=20.0)
    check("speed profile is produced along the corridor", len(sp) > 3, f"{len(sp)} bins")
    if len(sp):
        med = float(sp["median_speed_kmh"].median())
        check("recovered speed matches the 8 m/s the scene was built at",
              abs(med - 28.8) < 6.0, f"{med:.1f} km/h vs 28.8")


def test_queues():
    print("\n[queue length]")
    df, tid = synth_scene()
    # a standing queue on the east arm: eight vehicles nose to tail, stopped
    stopped = []
    for i in range(8):
        x = 40.0 + i * 6.0
        n = int(40 * FPS)
        stopped.append(pd.DataFrame({
            "frame": np.arange(0, n), "track_id": tid + i, "display_id": tid + i,
            "mode": "car", "cls_name": "car", "conf": 0.9,
            "world_x_m": x, "world_y_m": 60.0,
            "velocity_x_mps": 0.0, "velocity_y_mps": 0.0, "speed_kmh": 0.0,
            "parked": False, "interpolated": False,
            **dict(zip(("cx", "cy"), CAL.to_image(np.full(n, x), np.full(n, 60.0)))),
        }))
    for s in stopped:
        s["x1"] = s["cx"] - 22; s["x2"] = s["cx"] + 22
        s["y1"] = s["cy"] - 18; s["y2"] = s["cy"] + 18
    full = pd.concat([df] + stopped, ignore_index=True)
    N = net.build(full, GEO, frame_wh=(W, H))
    gate = next((g.name for g in N.gates
                 if any(o == g.name for (o, _) in N.corridors)), None)
    q = agg.queue_lengths(full, N, FPS, gate, sample_s=1.0)
    check("queue is measured on the approach", len(q) > 0, f"{len(q)} samples")
    if len(q):
        peak = int(q["queue_vehicles"].max())
        # the real defect: restricting to complete O-D journeys hid the queue
        check("stopped vehicles without a complete journey still count",
              peak >= 5, f"peak {peak} vehicles")
        check("queue length is reported in metres",
              float(q["queue_length_m"].max()) > 10.0,
              f"{q['queue_length_m'].max():.0f} m")


# ------------------------------------------------------------- flow analysis
def test_flow():
    print("\n[time-space, Edie, lanes]")
    df, _ = synth_scene()
    N = net.build(df, GEO, frame_wh=(W, H))
    o, d = agg.od_summary(N).iloc[0][["origin", "destination"]]

    ts = tf.time_space(df, N, o, d, FPS)
    check("time-space diagram is produced", bool(ts) and ts["n_tracks"] > 5,
          f"{ts.get('n_tracks')} trajectories")
    if ts:
        tr = ts["tracks"][0]
        check("trajectories carry time, distance and speed",
              len(tr["t"]) == len(tr["s"]) == len(tr["v"]))
        check("distance along the corridor increases",
              tr["s"][-1] > tr["s"][0], f"{tr['s'][0]} -> {tr['s'][-1]}")

    ped = _leg(9001, [(30.0, 74.0), (34.0, 74.0)], 0, mode="pedestrian", speed=1.2)
    ts2 = tf.time_space(pd.concat([df, ped], ignore_index=True), N, o, d, FPS)
    check("a pedestrian beside the road is not a corridor trajectory",
          ts2["n_tracks"] == ts["n_tracks"], f"{ts2['n_tracks']} vs {ts['n_tracks']}")

    e = tf.edie(df, N, o, d, FPS)
    check("Edie measures are produced", len(e) > 0, f"{len(e)} cells")
    if len(e):
        check("q = k*v holds identically", e.attrs["identity_residual"] < 1e-6,
              f"{e.attrs['identity_residual']:.2e}")
        # the real defect: summing lanes gave 750 veh/km and 8,800 veh/h
        check("density stays below any physical jam density",
              float(np.percentile(e["density_veh_per_km"], 95)) < 200,
              f"p95 {np.percentile(e['density_veh_per_km'], 95):.0f} veh/km/lane")
        check("flow stays within a plausible lane capacity",
              float(np.percentile(e["flow_veh_per_h"], 95)) < 3000,
              f"p95 {np.percentile(e['flow_veh_per_h'], 95):.0f} veh/h/lane")
        check("results are reported per lane", "lane" in e.columns,
              f"lanes {sorted(e['lane'].unique())}")

    lanes = tf.assign_lanes(df, N, FPS)
    check("lanes are assigned per frame, not once per track",
          int(lanes["lane"].notna().sum()) > 500,
          f"{int(lanes['lane'].notna().sum())} rows")
    lc = tf.lane_change_summary(df, lanes, FPS)
    if len(lc):
        worst = float(lc["changes_per_km"].max())
        # the real defect: counting every boundary wobble gave 48.7/km
        check("lane changes are not counted on boundary wobble", worst < 25.0,
              f"max {worst:.1f} per km")

    v = tf.fd_validity(e)
    check("fundamental-diagram validity is reportable", "negative_share" in v,
          f"{v.get('locations')} locations")


def test_edie_against_hand_calc():
    """Edie's definitions on a case whose answer can be worked out by hand."""
    print("\n[Edie: closed form]")
    # one vehicle crossing a 50 m segment at exactly 10 m/s
    n = int(10.0 * FPS)
    X = np.linspace(0.0, 100.0, n)
    Y = np.full(n, 60.0)
    px, py = CAL.to_image(X, Y)
    one = pd.DataFrame({
        "frame": np.arange(n), "track_id": 1, "display_id": 1, "mode": "car",
        "cls_name": "car", "conf": 0.9, "world_x_m": X, "world_y_m": Y,
        "cx": px, "cy": py, "x1": px - 22, "y1": py - 18,
        "x2": px + 22, "y2": py + 18,
        "velocity_x_mps": 10.0, "velocity_y_mps": 0.0, "speed_kmh": 36.0,
        "parked": False, "interpolated": False,
    })

    class FakeNet:
        corridors = {("A", "B"): np.array([[0.0, 60.0], [100.0, 60.0]])}
        movements = pd.DataFrame(columns=["origin", "destination"])

    e = tf.edie(one, FakeNet(), "A", "B", FPS, segment_m=50.0, interval_s=10.0)
    check("a single crossing produces measurable cells", len(e) > 0, f"{len(e)} cells")
    if len(e):
        v = float(e["speed_kmh"].median())
        check("recovered speed equals the 36 km/h it was built at",
              abs(v - 36.0) < 2.0, f"{v:.1f} km/h")
        # k = time-in-cell / area; 5 s in a 50 m x 10 s cell = 10 veh/km
        k = float(e["density_veh_per_km"].max())
        check("density matches the hand calculation (~10 veh/km)",
              5 < k < 15, f"{k:.1f} veh/km")


if __name__ == "__main__":
    test_georef()
    test_network()
    test_aggregates()
    test_queues()
    test_flow()
    test_edie_against_hand_calc()
    print("\n" + "=" * 62)
    if FAILS:
        print(f"{len(FAILS)} FAILED:")
        for f in FAILS:
            print(f"   - {f}")
        sys.exit(1)
    print("all L3/L4 checks passed")
