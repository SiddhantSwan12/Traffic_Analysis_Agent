"""L5 checks: congestion origination, signal inference, gap acceptance,
desire-line deviation, obstruction census.

L5 makes claims rather than measurements -- "this jam started here", "this
approach is signalised" -- so the tests build scenes where the claim has a known
right answer and check it comes back. Two of them check the opposite direction:
that a confident answer is NOT produced where the data cannot support one, which
is where reasoning code normally goes wrong.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from vtrack.calibration import GroundPlaneCalibration
from vtrack import reasoning as R
from vtrack import trafficflow as tf

W, H, FPS = 1920, 1080, 29.97
CAL = GroundPlaneCalibration(70.472, -63.1, 924.5, W, H)
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


class FakeNet:
    """One straight 200 m corridor along Y = 60 m."""
    corridors = {("A", "B"): np.array([[0.0, 60.0], [200.0, 60.0]])}
    movements = pd.DataFrame({"origin": [], "destination": [], "turn": []})


def vehicle(tid, t_start, s_of_t, mode="car"):
    """A vehicle whose position along the corridor is given by s(t)."""
    n = len(s_of_t)
    X = np.asarray(s_of_t, float)
    Y = np.full(n, 60.0)
    px, py = CAL.to_image(X, Y)
    f0 = int(t_start * FPS)
    v = np.gradient(X) * FPS
    return pd.DataFrame({
        "frame": np.arange(f0, f0 + n), "track_id": tid, "display_id": tid,
        "mode": mode, "cls_name": "car", "conf": 0.9,
        "world_x_m": X, "world_y_m": Y, "cx": px, "cy": py,
        "x1": px - 22, "y1": py - 18, "x2": px + 22, "y2": py + 18,
        "velocity_x_mps": v, "velocity_y_mps": 0.0,
        "speed_kmh": np.abs(v) * 3.6, "parked": False, "interpolated": False,
    })


# ------------------------------------------------------- congestion origins
def test_congestion():
    print("\n[congestion origination]")

    # Free flow everywhere: no region should be claimed at all.
    free = []
    for i in range(25):
        t0 = i * 2.0
        n = int(14 * FPS)
        free.append(vehicle(i + 1, t0, np.linspace(0, 200, n)))
    e = tf.edie(pd.concat(free, ignore_index=True), FakeNet(), "A", "B", FPS)
    c = R.congestion_origins(e)
    check("free flow yields no congested region", len(c["regions"]) == 0, c["note"])

    # A bottleneck switching on at t=100 s at 120 m: vehicles entering after
    # that crawl through the 120-160 m band and run free elsewhere.
    veh = []
    for i in range(60):
        t0 = i * 4.0
        pos, s, t = [], 0.0, 0.0
        while s < 200 and t < 200:
            jam = (t0 + t) > 100 and 120 <= s <= 160
            s += (1.2 if jam else 12.0) / FPS
            pos.append(s)
            t += 1 / FPS
        veh.append(vehicle(200 + i, t0, pos))
    df = pd.concat(veh, ignore_index=True)
    e2 = tf.edie(df, FakeNet(), "A", "B", FPS, segment_m=20.0, interval_s=10.0)
    c2 = R.congestion_origins(e2, congested_kmh=15.0)
    check("a bottleneck produces a congested region", len(c2["regions"]) >= 1,
          c2["note"])
    if c2["regions"]:
        r = c2["regions"][0]
        check("origin is located at the bottleneck (120-160 m)",
              100 <= r["origin_distance_m"] <= 170, f"{r['origin_distance_m']} m")
        check("onset time is near when the bottleneck switched on (100 s)",
              80 <= r["origin_time_s"] <= 140, f"{r['origin_time_s']} s")
        check("a jam formed inside the clip is not flagged as predating it",
              r["predates_clip"] is False, f"start {r['start_s']}s")
        check("shockwave speed is reported", r["shockwave_mps"] is not None)


def test_shockwave_direction():
    print("\n[shockwave direction]")
    # A queue whose tail moves UPSTREAM: each later vehicle stops further back.
    veh = []
    for i in range(40):
        t0 = i * 3.0
        stop_at = 170.0 - i * 3.0            # tail recedes at ~1 m/s
        pos, s, t = [], 0.0, 0.0
        while s < 200 and t < 240:
            s += (0.4 if s >= stop_at and (t0 + t) > 30 else 11.0) / FPS
            pos.append(s)
            t += 1 / FPS
        veh.append(vehicle(400 + i, t0, pos))
    e = tf.edie(pd.concat(veh, ignore_index=True), FakeNet(), "A", "B", FPS,
                segment_m=20.0, interval_s=10.0)
    c = R.congestion_origins(e, congested_kmh=15.0)
    check("a receding queue tail is detected", len(c["regions"]) >= 1, c["note"])
    if c["regions"]:
        r = max(c["regions"], key=lambda x: x["cells"])
        # a queue that grows then clears has two waves; a single fit across
        # both averages to zero, so each phase is checked on its own
        check("the stopping wave is reported as propagating upstream",
              r["growth_direction"] == "upstream",
              f"{r['stopping_wave_mps']} m/s -> {r['growth_direction']}")
        check("its magnitude is near the 1 m/s the queue was built to recede at",
              r["stopping_wave_mps"] is not None
              and 0.4 < abs(r["stopping_wave_mps"]) < 2.5,
              f"{r['stopping_wave_mps']} m/s")
        check("the starting wave is reported as clearing downstream",
              r["discharge_direction"] == "downstream",
              f"{r['starting_wave_mps']} m/s -> {r['discharge_direction']}")


# ------------------------------------------------------------ signal inference
def _stream(tids, times, speed=11.0):
    veh = []
    n = int((200.0 / speed) * FPS)
    for tid, t0 in zip(tids, times):
        veh.append(vehicle(tid, t0, np.linspace(0, 200, n)))
    return pd.concat(veh, ignore_index=True)


def test_signal():
    print("\n[signal inference]")
    # A REGULAR 60 s cycle: 30 s of discharge at 2 s headways, then 30 s red.
    times, tid, t = [], 700, 0.0
    ids = []
    for cycle in range(7):
        base = cycle * 60.0
        for k in range(15):
            times.append(base + k * 2.0)
            ids.append(tid); tid += 1
    df = _stream(ids, times)
    s = R.signal_performance(df, FakeNet(), "A", "B", FPS)
    check("a regular cycle is recognised as signalised", s.get("signalised") is True,
          s.get("note"))
    if s.get("signalised"):
        check("cycle count is about right", 5 <= s["n_cycles"] <= 8, s["n_cycles"])
        check("red duration is near the 30 s built",
              s["median_red_s"] is not None and 20 <= s["median_red_s"] <= 40,
              f"{s['median_red_s']} s")
        sat = s["median_saturation_flow_veh_h"]
        check("saturation flow matches 2 s headways (~1800 veh/h)",
              sat is not None and 1500 <= sat <= 2100, f"{sat} veh/h")

    # IRREGULAR gaps: the real failure mode. Long interruptions exist, but they
    # are not periodic, so this must NOT be called a signal.
    times, ids = [], []
    tid = 900
    for base, count in ((0, 12), (95, 8), (150, 14), (330, 9)):
        for k in range(count):
            times.append(base + k * 1.6)
            ids.append(tid); tid += 1
    s2 = R.signal_performance(_stream(ids, times), FakeNet(), "A", "B", FPS)
    check("irregular interruptions are NOT called a signal",
          s2.get("signalised") is False, s2.get("note"))
    check("the cycle-regularity evidence is reported",
          s2.get("cycle_length_cov") is not None,
          f"CoV {s2.get('cycle_length_cov')} vs {s2.get('cycle_cov_threshold')}")

    s3 = R.signal_performance(_stream([1, 2, 3], [0, 5, 10]), FakeNet(), "A", "B", FPS)
    check("too little traffic yields no verdict, not a guess",
          s3.get("signalised") is None, s3.get("note"))


# --------------------------------------------------------------- obstructions
def test_obstructions():
    print("\n[obstruction census]")
    moving = _stream(list(range(1, 16)), [i * 3.0 for i in range(15)])
    n = int(120 * FPS)

    def still(tid, X, Y, mode="car"):
        px, py = CAL.to_image(np.full(n, X), np.full(n, Y))
        return pd.DataFrame({
            "frame": np.arange(n), "track_id": tid, "display_id": pd.NA,
            "mode": mode, "cls_name": "car", "conf": 0.9,
            "world_x_m": X, "world_y_m": Y, "cx": px, "cy": py,
            "x1": px - 22, "y1": py - 18, "x2": px + 22, "y2": py + 18,
            "velocity_x_mps": 0.0, "velocity_y_mps": 0.0, "speed_kmh": 0.0,
            "parked": True, "interpolated": False,
        })

    blocking = still(9001, 90.0, 61.5)      # in the running lane
    kerb = still(9002, 110.0, 65.5)         # at the kerb
    offroad = still(9003, 130.0, 92.0)      # well off the carriageway
    allrows = pd.concat([moving, blocking, kerb, offroad], ignore_index=True)

    obs = R.obstruction_census(allrows, FakeNet(), FPS)
    ids = set(obs["track_id"]) if len(obs) else set()
    check("a vehicle stopped in the running lane is an obstruction", 9001 in ids)
    check("a kerbside vehicle is recorded too", 9002 in ids)
    check("a vehicle well off the carriageway is NOT an obstruction",
          9003 not in ids, f"found {sorted(ids)}")
    if 9001 in ids:
        r = obs[obs["track_id"] == 9001].iloc[0]
        check("it is classified as in-lane", r["obstruction"] == "in-lane",
              f"{r['obstruction']} at {r['lateral_offset_m']} m")
        check("dwell duration is reported and correct",
              abs(float(r["dwell_s"]) - 120.0) < 3.0, f"{r['dwell_s']} s")
    check("moving vehicles are never counted as obstructions",
          not ({1, 2, 3} & ids))


# ------------------------------------------------------------- desire lines
def test_desire():
    print("\n[desire-line deviation]")

    class Net2:
        corridors = {("A", "B"): np.array([[0.0, 60.0], [200.0, 60.0]])}
        movements = pd.DataFrame(
            {"origin": ["A"] * 24, "destination": ["B"] * 24, "turn": ["through"] * 24},
            index=range(1, 25))

    # disciplined: every vehicle holds a lane centre
    veh = []
    nn = int(18 * FPS)
    for i in range(1, 25):
        X = np.linspace(0, 200, nn)
        Y = np.full(nn, 60.0 + (1.6 if i % 2 else -1.6))
        px, py = CAL.to_image(X, Y)
        veh.append(pd.DataFrame({
            "frame": np.arange(nn), "track_id": i, "display_id": i, "mode": "car",
            "cls_name": "car", "conf": 0.9, "world_x_m": X, "world_y_m": Y,
            "cx": px, "cy": py, "x1": px - 22, "y1": py - 18,
            "x2": px + 22, "y2": py + 18, "velocity_x_mps": 11.0,
            "velocity_y_mps": 0.0, "speed_kmh": 40.0,
            "parked": False, "interpolated": False}))
    d1 = R.desire_deviation(pd.concat(veh, ignore_index=True), Net2(), FPS)
    check("lane-keeping traffic is scored as good discipline",
          len(d1) and d1.iloc[0]["lane_discipline"] == "good",
          d1.iloc[0]["lane_discipline"] if len(d1) else "no rows")

    # Undisciplined: continuous lateral weaving across the carriageway. Drawing
    # one fixed offset per vehicle would give only 24 samples of the statistic
    # and make the result a coin flip; real absence of lane discipline is
    # vehicles drifting across boundaries as they travel.
    rng = np.random.default_rng(3)
    veh = []
    for i in range(1, 25):
        X = np.linspace(0, 200, nn)
        # A triangle wave, not a sinusoid: a sinusoid has an arcsine marginal
        # and lingers at its extremes, which really is PARTIAL discipline. A
        # triangle wave is uniformly distributed across the carriageway, which
        # is what an absence of lane discipline means.
        u = np.linspace(0, 3, nn) + rng.uniform(0, 1)
        tri = 2 * np.abs(2 * (u - np.floor(u + 0.5))) - 1
        Y = 60.0 + 5.0 * tri
        px, py = CAL.to_image(X, Y)
        veh.append(pd.DataFrame({
            "frame": np.arange(nn), "track_id": i, "display_id": i, "mode": "car",
            "cls_name": "car", "conf": 0.9, "world_x_m": X, "world_y_m": Y,
            "cx": px, "cy": py, "x1": px - 22, "y1": py - 18,
            "x2": px + 22, "y2": py + 18, "velocity_x_mps": 11.0,
            "velocity_y_mps": 0.0, "speed_kmh": 40.0,
            "parked": False, "interpolated": False}))
    d2 = R.desire_deviation(pd.concat(veh, ignore_index=True), Net2(), FPS)
    check("uniformly spread traffic is scored as no discipline",
          len(d2) and d2.iloc[0]["lane_discipline"] == "none",
          d2.iloc[0]["lane_discipline"] if len(d2) else "no rows")
    check("the chance baseline is reported alongside the raw share",
          len(d2) and "straddle_expected_by_chance" in d2.columns,
          f"{d2.iloc[0]['straddle_expected_by_chance']:.2f}" if len(d2) else "")


# ==================================================== improved estimators (L5+)
def test_rankine_hugoniot():
    print("\n[shockwave: Rankine-Hugoniot]")
    from vtrack import reasoning2 as R2

    # Two cells either side of a front, with q and k chosen so the jump
    # condition has an answer that can be worked out by hand:
    #   u = (q2-q1)/(k2-k1) = (600-1800)/(90-30) = -20 km/h = -5.56 m/s
    e = pd.DataFrame({
        "segment": [0, 1], "interval": [0, 0], "lane": [0, 0],
        "flow_veh_per_h": [1800.0, 600.0],
        "density_veh_per_km": [30.0, 90.0],
        "speed_kmh": [60.0, 6.7],
        "segment_start_m": [0.0, 50.0], "interval_start_s": [0.0, 0.0],
        "n_vehicles": [8, 8],
    })
    r = R2.shockwave_rankine_hugoniot(e, min_pairs=1)
    check("the jump condition is evaluated across the front",
          r.get("median_mps") is not None, r.get("note"))
    if r.get("median_mps") is not None:
        check("it matches the hand calculation (-5.56 m/s)",
              abs(r["median_mps"] + 5.56) < 0.2, f"{r['median_mps']} m/s")
        check("direction is reported as upstream", r["direction"] == "upstream")
        check("the q and k it reasoned from are attached",
              bool(r.get("evidence")) and "k_up" in r["evidence"][0])

    same = e.copy()
    same["speed_kmh"] = [60.0, 60.0]           # no front at all
    r2 = R2.shockwave_rankine_hugoniot(same, min_pairs=1)
    check("no front means no shockwave is claimed",
          r2.get("median_mps") is None, r2.get("note"))


def test_signal_periodicity():
    print("\n[signal: periodicity and complementarity]")
    from vtrack import reasoning2 as R2

    ids, times, tid = [], [], 5000
    for cycle in range(8):
        for k in range(16):
            times.append(cycle * 60.0 + k * 1.8)
            ids.append(tid); tid += 1
    s = R2.signal_periodicity(_stream(ids, times), FakeNet(), "A", "B", FPS)
    check("a 60 s cycle is detected by autocorrelation",
          s.get("signalised") is True, s.get("note"))
    if s.get("signalised"):
        check("the recovered cycle length is near 60 s",
              abs(s["cycle_length_s"] - 60.0) < 8.0, f"{s['cycle_length_s']} s")

    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0, 480, 220))
    s2 = R2.signal_periodicity(_stream(list(range(6000, 6220)), list(t)),
                               FakeNet(), "A", "B", FPS)
    check("Poisson arrivals are NOT called periodic",
          s2.get("signalised") is False,
          f"prominence {s2.get('peak_prominence')} vs {s2.get('prominence_threshold')}")


def test_raff():
    print("\n[critical gap: Raff's method]")
    from vtrack.reasoning2 import raff_critical_gap
    rng = np.random.default_rng(5)
    # accepted ~ U(3,10), rejected ~ U(0,5) cross where (t-3)/7 = (5-t)/5,
    # i.e. at t = 50/12 = 4.17 s
    acc = rng.uniform(3, 10, 4000)
    rej = rng.uniform(0, 5, 4000)
    r = raff_critical_gap(acc, rej)
    check("Raff returns a critical gap", r.get("critical_gap_s") is not None,
          r.get("note"))
    if r.get("critical_gap_s") is not None:
        check("it matches the analytic crossing at 4.17 s",
              abs(r["critical_gap_s"] - 4.17) < 0.35, f"{r['critical_gap_s']} s")
    check("too little data yields no estimate, not a guess",
          raff_critical_gap(acc[:3], rej[:3]).get("critical_gap_s") is None)
    # disjoint distributions never cross, and that must be said rather than faked
    r3 = raff_critical_gap(rng.uniform(20, 30, 500), rng.uniform(0, 1, 500))
    check("non-overlapping distributions report no crossing",
          r3.get("critical_gap_s") is None, r3.get("note"))


def test_lane_structure():
    print("\n[lane structure from behaviour]")
    from vtrack import reasoning2 as R2

    class Net3:
        corridors = {("A", "B"): np.array([[0.0, 60.0], [200.0, 60.0]])}
        movements = pd.DataFrame(
            {"origin": ["A"] * 40, "destination": ["B"] * 40, "turn": ["through"] * 40},
            index=range(1, 41))

    def build(offsets):
        veh, nn = [], int(16 * FPS)
        for i, off in enumerate(offsets, start=1):
            X = np.linspace(0, 200, nn)
            Y = np.full(nn, 60.0) + off
            px, py = CAL.to_image(X, Y)
            veh.append(pd.DataFrame({
                "frame": np.arange(nn), "track_id": i, "display_id": i,
                "mode": "car", "cls_name": "car", "conf": 0.9,
                "world_x_m": X, "world_y_m": Y, "cx": px, "cy": py,
                "x1": px - 22, "y1": py - 18, "x2": px + 22, "y2": py + 18,
                "velocity_x_mps": 12.0, "velocity_y_mps": 0.0, "speed_kmh": 43.0,
                "parked": False, "interpolated": False}))
        return pd.concat(veh, ignore_index=True)

    rng = np.random.default_rng(2)
    # three lanes 3.2 m apart, drivers holding them
    lanes = np.array([-3.2, 0.0, 3.2])[rng.integers(0, 3, 40)] + rng.normal(0, 0.25, 40)
    d1 = R2.lane_structure(build(lanes), Net3())
    check("three separated lanes are recognised as lane structure",
          len(d1) and d1.iloc[0]["lane_structure"] == "clear",
          f"{d1.iloc[0]['lane_structure']}, {d1.iloc[0]['n_modes']} modes at "
          f"{d1.iloc[0]['observed_lane_spacing_m']} m" if len(d1) else "no rows")
    if len(d1):
        check("the observed spacing is close to the real 3.2 m",
              d1.iloc[0]["observed_lane_spacing_m"] is not None
              and abs(d1.iloc[0]["observed_lane_spacing_m"] - 3.2) < 0.9,
              f"{d1.iloc[0]['observed_lane_spacing_m']} m")

    # one tight band: concentrated, but NOT lane-separated
    d2 = R2.lane_structure(build(rng.normal(0, 0.5, 40)), Net3())
    check("a single tight band is not mistaken for lane structure",
          len(d2) and d2.iloc[0]["lane_structure"] == "single undifferentiated stream",
          d2.iloc[0]["lane_structure"] if len(d2) else "no rows")


if __name__ == "__main__":
    test_congestion()
    test_shockwave_direction()
    test_signal()
    test_obstructions()
    test_desire()
    test_rankine_hugoniot()
    test_signal_periodicity()
    test_raff()
    test_lane_structure()
    print("\n" + "=" * 62)
    if FAILS:
        print(f"{len(FAILS)} FAILED:")
        for f in FAILS:
            print(f"   - {f}")
        sys.exit(1)
    print("all L5 checks passed")
