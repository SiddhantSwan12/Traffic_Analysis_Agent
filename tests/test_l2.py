"""L2 checks: calibration, kinematics, classification, motion state.

Every kinematic test is closed-loop. A trajectory is defined in ground metres
with known speed and acceleration, projected into the image through the
calibration, then run back through the pipeline. Recovered values are compared
against the truth that generated them, so the test measures the estimator
rather than restating it.

No GPU and no real video required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from vtrack.config import Config
from vtrack import kinematics as kin
from vtrack.calibration import GroundPlaneCalibration, RowScaleCalibration, describe
from vtrack.classification import classify_track
from vtrack.taxonomy import MODES

W, H, FPS = 1920, 1080, 29.97
CAL = GroundPlaneCalibration(70.472, -63.1, 924.5, W, H)
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


class _ClsCfg:
    min_length_samples = 5
    length_uncertainty_floor_m = 0.35


def make_track(tid, X0, Y0, vx, vy, ax=0.0, ay=0.0, n=200, f0=0,
               cls_name="car", conf=0.9, box_px=44, hide=None, jitter_px=0.0,
               seed=0):
    """Build a track whose GROUND motion is exactly known."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if hide and hide[0] <= i <= hide[1]:
            continue
        t = i / FPS
        X = X0 + vx * t + 0.5 * ax * t * t
        Y = Y0 + vy * t + 0.5 * ay * t * t
        x, y = CAL.to_image(X, Y)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        x = float(x) + rng.normal(0, jitter_px)
        y = float(y) + rng.normal(0, jitter_px)
        rows.append((f0 + i, tid, x - box_px / 2, y - box_px / 2,
                     x + box_px / 2, y + box_px / 2, conf, 3, cls_name))
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2",
                                       "y2", "conf", "cls_id", "cls_name"])


def run_kin(df, cfg=None, cal=CAL):
    cfg = cfg or Config()
    df = df.copy()
    if "interpolated" not in df:
        df["interpolated"] = False
    df["cx"] = 0.5 * (df["x1"] + df["x2"])
    df["cy"] = 0.5 * (df["y1"] + df["y2"])
    return kin.compute(df, cal, FPS, cfg)


def mid(series):
    v = series.to_numpy(float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


# ------------------------------------------------------------------ calibration
def test_calibration():
    print("\n[calibration]")
    check("s_y = (f/h) * s_x^2 holds exactly",
          all(abs(float(CAL.scales(y)[1]) - (CAL.f / CAL.h) * float(CAL.scales(y)[0]) ** 2)
              < 1e-12 for y in (100, 540, 900)))
    a_top, a_bot = float(CAL.anisotropy(0)), float(CAL.anisotropy(1079))
    check("oblique view is anisotropic (not 1.0)", a_top > 1.4 and a_bot < 0.95,
          f"top {a_top:.2f} bottom {a_bot:.2f}")
    x, y = CAL.to_image(*CAL.to_ground(np.array([700.]), np.array([600.])))
    check("ground<->image round trip is exact",
          abs(float(x[0]) - 700) < 1e-6 and abs(float(y[0]) - 600) < 1e-6)
    check("rows above the horizon are out of range",
          not bool(np.all(CAL.in_range(np.array([float(CAL.horizon_row()) - 5])))))

    # recovering a known focal length from synthetic car sizes
    ys = np.linspace(50, 1050, 4000)
    sx, sy = CAL.scales(ys)
    px = 4.3 / np.sqrt(sx * sy)
    fit = GroundPlaneCalibration.fit(ys, px, 70.472, -63.1, W, H, 4.3)
    check("fit recovers the true focal length within 1%",
          fit is not None and abs(fit.f - CAL.f) / CAL.f < 0.01,
          f"{fit.f:.1f} vs {CAL.f}")

    rs = RowScaleCalibration.fit(ys, px, 4.3)
    check("row-scale fallback reports lower confidence than homography",
          rs.quality.confidence < fit.quality.confidence,
          f"{rs.quality.confidence:.2f} vs {fit.quality.confidence:.2f}")
    check("row-scale is isotropic by construction",
          abs(float(rs.scales(500)[0]) - float(rs.scales(500)[1])) < 1e-12)


# ------------------------------------------------------------------- kinematics
def test_constant_speed():
    print("\n[kinematics: constant speed]")
    for truth in (5.0, 12.0, 20.0):
        df = run_kin(make_track(1, -40, 60, truth, 0.0, n=220))
        got = mid(df["speed_mps"])
        err = abs(got - truth) / truth
        check(f"across-view speed {truth} m/s recovered within 10%", err < 0.10,
              f"got {got:.2f} ({err*100:.1f}% err)")

    # travelling along the view direction is where an isotropic scale fails
    df = run_kin(make_track(2, 0, 45, 0.0, 12.0, n=220))
    got = mid(df["speed_mps"])
    check("along-view speed 12 m/s recovered within 10%", abs(got - 12) / 12 < 0.10,
          f"got {got:.2f}")

    df = run_kin(make_track(3, -30, 55, 8.0, 8.0, n=220))
    truth = float(np.hypot(8, 8))
    got = mid(df["speed_mps"])
    check("diagonal speed recovered within 10%", abs(got - truth) / truth < 0.10,
          f"got {got:.2f} vs {truth:.2f}")


def test_isotropic_bias():
    """The concrete cost of the assumption the L2 spec originally made."""
    print("\n[kinematics: isotropic scale is direction-biased]")
    ys = np.linspace(50, 1050, 4000)
    sx, sy = CAL.scales(ys)
    rs = RowScaleCalibration.fit(ys, 4.3 / np.sqrt(sx * sy), 4.3)

    # The bias grows with distance from the camera, because the anisotropy does:
    # ~1.1x mid-frame, ~1.6x near the top. Both regions are checked.
    for label, (Xa, Ya), (Xb, Yb), floor in (
            ("mid-frame", (-40, 60), (0, 45), 1.06),
            ("far field", (-40, 110), (0, 95), 1.10)):
        across = mid(run_kin(make_track(1, Xa, Ya, 12.0, 0.0, n=220), cal=rs)["speed_mps"])
        along = mid(run_kin(make_track(2, Xb, Yb, 0.0, 12.0, n=220), cal=rs)["speed_mps"])
        ratio = max(across, along) / min(across, along)
        check(f"isotropic model biases speed by direction of travel ({label})",
              ratio > floor, f"across {across:.1f} vs along {along:.1f} m/s "
                             f"(ratio {ratio:.2f}) for an identical true 12 m/s")

    g_across = mid(run_kin(make_track(1, -40, 60, 12.0, 0.0, n=220))["speed_mps"])
    g_along = mid(run_kin(make_track(2, 0, 45, 0.0, 12.0, n=220))["speed_mps"])
    g_ratio = max(g_across, g_along) / min(g_across, g_along)
    check("ground-plane model does not", g_ratio < 1.06,
          f"across {g_across:.1f} vs along {g_along:.1f} (ratio {g_ratio:.2f})")


def test_acceleration():
    print("\n[kinematics: acceleration and braking]")
    for truth in (1.5, 3.0):
        df = run_kin(make_track(1, -60, 60, 4.0, 0.0, ax=truth, n=260))
        got = mid(df["longitudinal_acceleration_mps2"])
        check(f"uniform accel {truth} m/s^2 recovered within 0.5",
              abs(got - truth) < 0.5, f"got {got:.2f}")

    df = run_kin(make_track(2, -70, 60, 16.0, 0.0, ax=-3.0, n=150))
    got = mid(df["longitudinal_acceleration_mps2"])
    check("braking yields NEGATIVE longitudinal acceleration", got < -0.5,
          f"got {got:.2f} m/s^2")
    check("braking magnitude within 0.5 m/s^2", abs(got + 3.0) < 0.5, f"got {got:.2f}")

    df = run_kin(make_track(3, -40, 60, 10.0, 0.0, n=220))
    got = mid(df["longitudinal_acceleration_mps2"])
    check("constant speed yields near-zero acceleration", abs(got) < 0.3,
          f"got {got:.3f}")


def test_camera_motion_not_velocity():
    """A static object under a moving camera must not acquire velocity.

    The tracker's compensation is what removes this; here we assert the
    downstream consequence, that compensated coordinates give zero speed while
    uncompensated ones do not -- so a regression in CMC shows up as motion.
    """
    print("\n[kinematics: camera motion is not object velocity]")
    still = make_track(1, 0.0, 50.0, 0.0, 0.0, n=200)
    got = mid(run_kin(still)["speed_mps"])
    check("stationary object measures ~0 m/s", got < 0.15, f"got {got:.4f}")

    drift = still.copy()
    shift = np.arange(len(drift)) * 1.5          # 1.5 px/frame of uncorrected pan
    for c in ("x1", "x2"):
        drift[c] = drift[c].to_numpy() + shift
    got_drift = mid(run_kin(drift)["speed_mps"])
    check("uncompensated camera pan WOULD read as motion", got_drift > 2.0,
          f"{got_drift:.2f} m/s of phantom speed")


def test_nulls_not_zeros():
    print("\n[quality: unavailable is null, not zero]")
    df = run_kin(make_track(1, -40, 60, 10.0, 0.0, n=200))
    edge = df.iloc[:3]
    check("velocity window shoulders are NaN, not 0",
          bool(edge["speed_mps"].isna().all()))
    short = run_kin(make_track(2, -40, 60, 10.0, 0.0, n=6))
    check("a track shorter than the window yields no speed at all",
          bool(short["speed_mps"].isna().all()), f"{len(short)} rows")
    check("speed column is float (nullable), never filled with 0",
          not (df["speed_mps"].fillna(-1) == 0).any())

    cfg = Config()
    fast = make_track(3, -300, 60, 90.0, 0.0, n=120)     # ~324 km/h
    out = run_kin(fast, cfg)
    check("implausible speed is nulled and flagged, not clamped",
          bool(out["implausible_speed"].any()) and bool(out["speed_mps"].isna().any()))

    seg = pd.concat([make_track(4, -40, 60, 10.0, 0.0, n=60, f0=0),
                     make_track(4, 40, 60, 10.0, 0.0, n=60, f0=400)])
    out = run_kin(seg)
    check("no kinematics are computed across a segment discontinuity",
          out.groupby("segment").ngroups == 2 and
          bool(out.groupby("segment")["speed_mps"].apply(
              lambda s: s.isna().iloc[0]).all()))


def test_path_and_window():
    print("\n[motion measures]")
    df = run_kin(make_track(1, 0.0, 50.0, 0.0, 0.0, n=1200, jitter_px=1.2, seed=3))
    path, net = kin.resample_path(df, FPS)
    win = kin.max_window_displacement(df, FPS, 10.0)
    check("jitter still accumulates path length over a long static track",
          path > 6.0, f"path {path:.1f} m over 40 s -- a fixed path budget fails here")
    check("net displacement of a static object stays small", net < 2.5,
          f"net {net:.2f} m")
    check("windowed displacement is immune to that accumulation", win < 3.0,
          f"max 10 s displacement {win:.2f} m")

    mv = run_kin(make_track(2, -60, 55, 9.0, 0.0, n=400))
    check("a moving object exceeds the windowed threshold",
          kin.max_window_displacement(mv, FPS, 10.0) > 3.0)


# --------------------------------------------------------------- classification
def _cls(cls_name, length_m, n=120, conf=0.9):
    """A track whose apparent size corresponds to a known physical length."""
    y = 600.0
    sx, sy = CAL.scales(y)
    box = float(length_m / np.sqrt(float(sx) * float(sy)))
    df = make_track(1, 0.0, 0.0, 0.0, 0.0, n=n, cls_name=cls_name, conf=conf)
    df["y1"] = y - box / 2
    df["y2"] = y + box / 2
    df["x1"] = 900 - box / 2
    df["x2"] = 900 + box / 2
    df["interpolated"] = False
    return df


def test_classification():
    print("\n[classification: track-level, length-refined]")
    cases = [("car", 4.3, "car"), ("car", 6.9, "LGV"), ("van", 5.0, "LGV"),
             ("van", 8.4, "truck"), ("truck", 12.0, "HGV"), ("truck", 8.0, "truck"),
             ("bus", 11.0, "bus"), ("bus", 5.0, "LGV")]
    for raw, length, want in cases:
        r = classify_track(_cls(raw, length), CAL, W, H, _ClsCfg)
        check(f"{raw} at {length} m -> {want}", r["final_mode"] == want,
              f"got {r['final_mode']} ({r['classification_source']}, "
              f"L={r['estimated_length_m']:.1f}m)")

    r = classify_track(_cls("pedestrian", 4.0), CAL, W, H, _ClsCfg)
    check("a pedestrian is never promoted to a vehicle class by size",
          r["final_mode"] == "pedestrian", f"got {r['final_mode']}")
    r = classify_track(_cls("motor", 5.0), CAL, W, H, _ClsCfg)
    check("a motorcycle is never promoted to a vehicle class by size",
          r["final_mode"] == "motorcycle", f"got {r['final_mode']}")

    # a length sitting on a threshold must not force an unstable override
    r = classify_track(_cls("car", 6.2), CAL, W, H, _ClsCfg)
    check("a length on the threshold keeps the appearance class",
          r["final_mode"] == "car", f"got {r['final_mode']}")
    check("...and reports reduced confidence instead", r["mode_confidence"] < 0.95,
          f"conf {r['mode_confidence']:.2f}")

    # flickering per-frame labels must not flicker the track label
    df = _cls("car", 4.3, n=150)
    names = (["car"] * 100) + (["van"] * 30) + (["truck"] * 20)
    df["cls_name"] = names
    r = classify_track(df, CAL, W, H, _ClsCfg)
    check("flickering frame labels still give one stable track class",
          r["final_mode"] == "car", f"got {r['final_mode']}")
    check("vote margin is reported", r["vote_margin"] > 0.3,
          f"margin {r['vote_margin']:.2f}")
    check("every emitted mode is one of the 7", r["final_mode"] in MODES)


# --------------------------------------------------------------- motion states
def test_motion_states():
    print("\n[motion state: moving / temporarily_stopped / parked]")
    from vtrack.analytics import analyze

    cfg = Config()
    cfg.post.min_track_len = 20
    cfg.post.min_track_hits = 10

    # a vehicle that drives, waits 12 s at a signal, then drives on
    parts = []
    n1 = 150
    parts.append(make_track(1, -80, 55, 10.0, 0.0, n=n1, seed=1))
    Xs = -80 + 10.0 * (n1 - 1) / FPS
    wait = make_track(1, Xs, 55, 0.0, 0.0, n=360, f0=n1, jitter_px=0.6, seed=2)
    parts.append(wait)
    parts.append(make_track(1, Xs, 55, 10.0, 0.0, n=150, f0=n1 + 360, seed=3))
    waiting = pd.concat(parts, ignore_index=True)

    parked = make_track(2, 30, 40, 0.0, 0.0, n=660, jitter_px=1.0, seed=4)
    fp = make_track(3, -25, 35, 0.0, 0.0, n=660, jitter_px=1.4, seed=5)
    mover = make_track(4, -70, 65, 9.0, 0.0, n=660, seed=6)

    tracks = pd.concat([waiting, parked, fp, mover], ignore_index=True)
    df, rep, cal = analyze(tracks, FPS, W, H,
                           {"rel_alt": 70.472, "gb_pitch": -63.1}, cfg)

    pk = df.groupby("track_id")["parked"].first()
    check("a vehicle waiting 12 s at a signal is NOT parked", not bool(pk[1]))
    check("a genuinely parked vehicle IS parked", bool(pk[2]))
    check("a static false positive IS suppressed as parked", bool(pk[3]))
    check("a continuously moving vehicle is NOT parked", not bool(pk[4]))

    w = df[df["track_id"] == 1]
    states = set(w["motion_state"].unique())
    check("the waiting vehicle shows both moving and temporarily_stopped",
          {"moving", "temporarily_stopped"} <= states, sorted(states))
    check("it is never labelled parked at frame level",
          "parked" not in states, sorted(states))
    check("parked tracks get no display id",
          bool(df[df["parked"]]["display_id"].isna().all()))
    check("moving tracks all get display ids",
          bool(df[~df["parked"]]["display_id"].notna().all()))
    check("report counts parked separately",
          rep["tracks_parked"] == 2 and rep["tracks_moving"] == 2,
          f"parked {rep['tracks_parked']} moving {rep['tracks_moving']}")
    check("calibration method recorded in the report",
          rep["calibration"]["method"] == "homography")


if __name__ == "__main__":
    test_calibration()
    test_constant_speed()
    test_isotropic_bias()
    test_acceleration()
    test_camera_motion_not_velocity()
    test_nulls_not_zeros()
    test_path_and_window()
    test_classification()
    test_motion_states()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILED:")
        for f in FAILS:
            print(f"   - {f}")
        sys.exit(1)
    print("all L2 checks passed")
