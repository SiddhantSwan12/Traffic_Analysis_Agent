"""L2 kinematics: metric position, velocity, acceleration.

Three rules govern everything here.

1. Never difference consecutive frames. Detector jitter is zero-mean, but any
   estimator that squares or rectifies it before averaging turns that noise
   into signal. Velocity uses a centred temporal baseline; acceleration uses a
   least-squares slope over a wider window still.

2. Never estimate across a discontinuity. Each track is split into contiguous
   segments, and no window is allowed to span a gap the tracker could not
   bridge. A window that would cross one yields NaN.

3. Never substitute zero for unknown. A vehicle with no measurable speed and a
   vehicle measured at zero speed are different facts, and the output has to
   preserve the difference or every downstream statistic is quietly wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
    _HAVE_SAVGOL = True
except Exception:                                    # pragma: no cover
    _HAVE_SAVGOL = False


def assign_segments(df: pd.DataFrame) -> pd.Series:
    """Contiguous-run id within each track. Any frame gap starts a new segment."""
    gap = df.groupby("track_id")["frame"].diff().fillna(1) > 1
    return gap.groupby(df["track_id"]).cumsum().astype(int)


def _centered_slope(v: np.ndarray, half: int, dt: float) -> np.ndarray:
    """Least-squares slope of v over a centred window of 2*half+1 samples.

    Equivalent to a Savitzky-Golay first derivative with polyorder 1, but
    written as a convolution so it stays fast over millions of rows. Uses every
    sample in the window rather than just the endpoints, which is roughly 1.7x
    less noisy than an endpoint difference of the same span.
    """
    n = 2 * half + 1
    if v.size < n:
        return np.full(v.size, np.nan)
    k = np.arange(half, -half - 1, -1, dtype=float)   # reversed for convolution
    denom = dt * float((k ** 2).sum())
    out = np.convolve(v, k, mode="same") / denom
    out[:half] = np.nan                               # window would run off the end
    out[-half:] = np.nan
    return out


def _centered_diff(x: np.ndarray, half: int, dt: float) -> np.ndarray:
    """Displacement across +/- `half` frames, divided by the elapsed time."""
    n = x.size
    out = np.full(n, np.nan)
    if n <= 2 * half:
        return out
    out[half:n - half] = (x[2 * half:] - x[:n - 2 * half]) / (2 * half * dt)
    return out


def compute(df: pd.DataFrame, cal, fps: float, cfg) -> pd.DataFrame:
    """Attach metric position, velocity and acceleration to a track table.

    `df` must be sorted by (track_id, frame) and carry cx, cy, interpolated.
    """
    k = cfg.kinematics
    dt = 1.0 / float(fps)
    half_v = max(1, int(round(k.velocity_window_s * fps / 2)))
    half_a = max(2, int(round(k.acceleration_window_s * fps / 2)))

    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    df["segment"] = assign_segments(df)

    # --- metric position, closed form (accumulating displacements would drift)
    X, Y = cal.to_ground(df["cx"].to_numpy(), df["cy"].to_numpy())
    df["world_x_m"] = X
    df["world_y_m"] = Y
    in_range = cal.in_range(df["cy"].to_numpy())

    n = len(df)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    ax = np.full(n, np.nan)
    ay = np.full(n, np.nan)
    interp_frac = np.full(n, np.nan)
    vwin = np.zeros(n, np.int16)
    awin = np.zeros(n, np.int16)

    interpolated = df["interpolated"].to_numpy().astype(float)
    use_savgol = (k.velocity_method == "savgol") and _HAVE_SAVGOL

    for _, idx in df.groupby(["track_id", "segment"], sort=False).indices.items():
        idx = np.asarray(idx)
        m = idx.size
        if m < 2 * half_v + 1:
            continue                                  # too short to measure at all
        sx = X[idx]
        sy = Y[idx]

        if use_savgol:
            wl = 2 * half_v + 1
            gx = savgol_filter(sx, wl, 1, deriv=1, delta=dt, mode="interp")
            gy = savgol_filter(sy, wl, 1, deriv=1, delta=dt, mode="interp")
        else:
            gx = _centered_diff(sx, half_v, dt)
            gy = _centered_diff(sy, half_v, dt)
        vx[idx] = gx
        vy[idx] = gy
        vwin[idx] = 2 * half_v + 1

        # fraction of the velocity window that came from interpolation
        ker = np.ones(2 * half_v + 1) / (2 * half_v + 1)
        f_int = np.convolve(interpolated[idx], ker, mode="same")
        f_int[:half_v] = np.nan
        f_int[-half_v:] = np.nan
        interp_frac[idx] = f_int

        # --- acceleration: slope of velocity over a wider window
        if m >= 2 * half_a + 1:
            gvx = np.where(np.isfinite(gx), gx, np.nan)
            gvy = np.where(np.isfinite(gy), gy, np.nan)
            # interpolate the NaN shoulders so the slope kernel stays defined,
            # then re-mask; this avoids losing the whole segment to its edges
            ok = np.isfinite(gvx)
            if ok.sum() >= 2 * half_a + 1:
                pos = np.arange(m)
                fx = np.interp(pos, pos[ok], gvx[ok])
                fy = np.interp(pos, pos[ok], gvy[ok])
                sax = _centered_slope(fx, half_a, dt)
                say = _centered_slope(fy, half_a, dt)
                bad = ~ok
                sax[bad] = np.nan
                say[bad] = np.nan
                ax[idx] = sax
                ay[idx] = say
                awin[idx] = 2 * half_a + 1

    speed = np.hypot(vx, vy)
    df["velocity_x_mps"] = vx
    df["velocity_y_mps"] = vy
    df["speed_mps"] = speed
    df["speed_kmh"] = speed * 3.6
    df["acceleration_x_mps2"] = ax
    df["acceleration_y_mps2"] = ay
    df["acceleration_mps2"] = np.hypot(ax, ay)

    # --- signed longitudinal / lateral acceleration.
    # Undefined when the object is nearly stationary: the direction of travel is
    # what gives the sign its meaning, and below a few km/h that direction is
    # noise. Reporting "braking" for a parked car would be nonsense.
    fast = speed >= k.min_speed_for_heading_mps
    denom = np.where(fast, np.maximum(speed, 1e-6), np.nan)
    df["longitudinal_acceleration_mps2"] = (ax * vx + ay * vy) / denom
    df["lateral_acceleration_mps2"] = (vx * ay - vy * ax) / denom
    heading = np.degrees(np.arctan2(vy, vx))
    df["heading_deg"] = np.where(fast, heading, np.nan)

    # --- quality
    df["velocity_window_frames"] = vwin
    df["acceleration_window_frames"] = awin
    df["interpolated_fraction_in_window"] = interp_frac
    cal_conf = float(getattr(getattr(cal, "quality", None), "confidence", 0.0) or 0.0)
    df["calibration_confidence"] = cal_conf
    df["calibration_method"] = cal.method

    implausible = speed > k.max_plausible_speed_mps
    too_invented = np.nan_to_num(interp_frac, nan=1.0) > k.max_interp_fraction
    valid = (np.isfinite(speed) & in_range & (~implausible) & (~too_invented)
             & (cal_conf > 0))
    df["kinematics_valid"] = valid
    q = np.where(valid, cal_conf, 0.0)
    q = q * (1.0 - np.nan_to_num(interp_frac, nan=1.0) * 0.5)
    df["kinematics_quality"] = np.where(valid, np.clip(q, 0.0, 1.0), np.nan)

    # a speed above the plausible ceiling is a measurement failure, not a fast
    # vehicle: null it rather than clamping it into a believable-looking number
    for c in ("speed_mps", "speed_kmh", "velocity_x_mps", "velocity_y_mps",
              "acceleration_x_mps2", "acceleration_y_mps2", "acceleration_mps2",
              "longitudinal_acceleration_mps2", "lateral_acceleration_mps2",
              "heading_deg"):
        df.loc[implausible, c] = np.nan
    df["implausible_speed"] = implausible

    # An acceleration beyond physical plausibility, or one computed mostly from
    # interpolated positions, is not a measurement. Null it rather than let an
    # extreme-value statistic downstream be defined by it.
    acc_cols = ("acceleration_x_mps2", "acceleration_y_mps2", "acceleration_mps2",
                "longitudinal_acceleration_mps2", "lateral_acceleration_mps2")
    bad_acc = ((np.abs(df["acceleration_mps2"].to_numpy()) > k.max_plausible_accel_mps2)
               | too_invented)
    for c in acc_cols:
        df.loc[bad_acc, c] = np.nan
    df["implausible_accel"] = bad_acc
    return df


def resample_path(g: pd.DataFrame, fps: float, interval_s: float = 0.5):
    """Path length and net displacement from centres resampled at `interval_s`.

    Summing every frame's displacement would accumulate jitter into metres of
    phantom travel; sampling twice a second averages it out while still
    following real turns.
    """
    step = max(1, int(round(interval_s * fps)))
    out = []
    for _, seg in g.groupby("segment", sort=False):
        s = seg.iloc[::step]
        if len(s) < 2:
            continue
        x = s["world_x_m"].to_numpy()
        y = s["world_y_m"].to_numpy()
        d = np.hypot(np.diff(x), np.diff(y))
        out.append(d[np.isfinite(d)])
    path = float(np.concatenate(out).sum()) if out else 0.0
    fx, fy = g["world_x_m"].iloc[0], g["world_y_m"].iloc[0]
    lx, ly = g["world_x_m"].iloc[-1], g["world_y_m"].iloc[-1]
    net = float(np.hypot(lx - fx, ly - fy)) if np.isfinite([fx, fy, lx, ly]).all() else np.nan
    return path, net


def max_window_displacement(g: pd.DataFrame, fps: float, window_s: float = 10.0,
                            interval_s: float = 0.5) -> float:
    """Farthest the object travelled from itself within any `window_s` window.

    This is the duration-independent replacement for a total-path-length test.
    Path length grows with track lifetime, so residual jitter alone pushes a
    long-lived parked vehicle past any fixed budget -- measured at 55-67 m over
    a 400 s stationary track. A sliding-window displacement cannot accumulate.
    """
    step = max(1, int(round(interval_s * fps)))
    w = max(1, int(round(window_s / interval_s)))
    best = 0.0
    for _, seg in g.groupby("segment", sort=False):
        s = seg.iloc[::step]
        x = s["world_x_m"].to_numpy()
        y = s["world_y_m"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size < 2:
            continue
        ww = min(w, x.size - 1)
        d = np.hypot(x[ww:] - x[:-ww], y[ww:] - y[:-ww])
        if d.size:
            best = max(best, float(d.max()))
    return best
