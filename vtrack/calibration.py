"""Metric calibration: image pixels -> ground-plane metres.

Two models, sharing one interface.

`GroundPlaneCalibration` is the real one. It builds a pinhole camera looking at
a flat ground plane from a known height at a known depression angle, both read
from DJI telemetry, and solves for the single remaining unknown -- effective
focal length in pixels -- against observed vehicle sizes. That yields a closed
form image->ground map, so positions never drift the way accumulated
frame-to-frame displacements do.

`RowScaleCalibration` is the isotropic fallback for footage with no usable
telemetry. It is deliberately marked lower confidence, because a single scale
per row is only correct for a nadir view.

WHY ANISOTROPY MATTERS
----------------------
For a camera at height h, depression angle t, focal length f (pixels), a pixel
offset p below the principal point maps to ground range with

    A(p) = p*cos(t) + f*sin(t)
    s_x  = dX/dp_x = h / A            (across the view)
    s_y  = |dY/dp_y| = h*f / A**2     (along the view)

so exactly

    s_y = (f/h) * s_x**2

The two scales are equal only when f/h * s_x == 1. On this dataset the ratio
runs from about 0.85 at the bottom of frame to 1.63 at the top, so treating the
scale as isotropic makes a vehicle's measured speed depend on its direction of
travel -- by up to ~60%. Every velocity and acceleration downstream inherits
that error, which is why the ground-plane model is preferred wherever telemetry
allows it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CalibrationQuality:
    method: str                 # homography | row_scale | uncalibrated
    confidence: float           # 0..1
    n_samples: int
    residual_rel: float         # median |predicted-observed| / observed
    row_min: float
    row_max: float
    notes: str = ""

    def as_dict(self):
        d = asdict(self)
        d["residual_rel"] = round(float(self.residual_rel), 5)
        d["confidence"] = round(float(self.confidence), 3)
        return d


class BaseCalibration:
    method = "uncalibrated"

    def scales(self, y):
        """(s_x, s_y) in metres/pixel at image row y."""
        raise NotImplementedError

    def to_ground(self, x, y):
        """Image (x, y) -> ground (X, Y) in metres. Closed form, no drift."""
        raise NotImplementedError

    def length_m(self, px_len, y):
        """Convert an apparent length in pixels near row y to metres.

        A vehicle's long axis lies in the ground plane but its image direction
        is unknown, so the geometric mean of the two scales is the least-biased
        single estimator available without knowing heading.
        """
        sx, sy = self.scales(y)
        return np.asarray(px_len, float) * np.sqrt(np.asarray(sx) * np.asarray(sy))

    def in_range(self, y):
        return np.ones_like(np.asarray(y, float), dtype=bool)


class GroundPlaneCalibration(BaseCalibration):
    """Pinhole camera viewing a flat ground plane, from DJI telemetry."""

    method = "homography"

    def __init__(self, height_m, pitch_deg, f_px, W, H, quality=None):
        self.h = float(height_m)
        self.pitch = float(pitch_deg)          # negative = looking down
        self.t = math.radians(abs(float(pitch_deg)))
        self.f = float(f_px)
        self.W, self.H = int(W), int(H)
        self.cx, self.cy = W / 2.0, H / 2.0
        self.quality = quality

    # A(p) = p cos t + f sin t ; p is pixels below the principal point.
    def _A(self, y):
        p = np.asarray(y, float) - self.cy
        return p * math.cos(self.t) + self.f * math.sin(self.t)

    def scales(self, y):
        A = self._A(y)
        A = np.where(A <= 1e-6, np.nan, A)      # at/above the horizon: undefined
        sx = self.h / A
        sy = self.h * self.f / (A * A)
        return sx, sy

    def to_ground(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        A = self._A(y)
        A = np.where(A <= 1e-6, np.nan, A)
        p = y - self.cy
        t_scale = self.h / A
        X = (x - self.cx) * t_scale
        Y = t_scale * (self.f * math.cos(self.t) - p * math.sin(self.t))
        return X, Y

    def in_range(self, y):
        return self._A(y) > 1e-6

    def horizon_row(self):
        """Image row where the ground plane meets the horizon."""
        return self.cy - self.f * math.tan(self.t)

    def anisotropy(self, y):
        sx, sy = self.scales(y)
        return sy / sx

    @staticmethod
    def fit(ys, px_lens, height_m, pitch_deg, W, H, ref_length_m,
            f_bounds=(200.0, 12000.0)):
        """Solve for effective focal length against observed vehicle lengths.

        Telemetry fixes the geometry (height and depression angle); the SRT's
        `focal_len` field is a 35 mm-equivalent and does not convert reliably to
        pixels, so f is the one free parameter. Fitting it in log space against
        the median observed size makes the solve well conditioned and robust to
        the long tail of oversized boxes.
        """
        ys = np.asarray(ys, float)
        px = np.asarray(px_lens, float)
        ok = np.isfinite(ys) & np.isfinite(px) & (px > 4)
        ys, px = ys[ok], px[ok]
        if ys.size < 100:
            return None

        # Bin by row and take medians, so every part of the frame counts equally
        # rather than being dominated by wherever traffic is densest.
        nb = 14
        edges = np.linspace(ys.min(), ys.max(), nb + 1)
        by, bl, bw = [], [], []
        for i in range(nb):
            hi = ys <= edges[i + 1] if i == nb - 1 else ys < edges[i + 1]
            m = (ys >= edges[i]) & hi
            if m.sum() >= 25:
                by.append(0.5 * (edges[i] + edges[i + 1]))
                bl.append(np.median(px[m]))
                bw.append(math.sqrt(m.sum()))
        if len(by) < 4:
            return None
        by, bl, bw = np.array(by), np.array(bl), np.array(bw)

        def predicted(f):
            c = GroundPlaneCalibration(height_m, pitch_deg, f, W, H)
            sx, sy = c.scales(by)
            with np.errstate(invalid="ignore"):
                return ref_length_m / np.sqrt(sx * sy)

        def cost(f):
            p = predicted(f)
            if not np.all(np.isfinite(p)) or np.any(p <= 0):
                return 1e9
            return float(np.sum(bw * (np.log(p) - np.log(bl)) ** 2))

        lo, hi = math.log(f_bounds[0]), math.log(f_bounds[1])
        for _ in range(80):                       # golden-section on log f
            a = hi - (hi - lo) * 0.6180339887
            b = lo + (hi - lo) * 0.6180339887
            if cost(math.exp(a)) < cost(math.exp(b)):
                hi = b
            else:
                lo = a
        f = math.exp(0.5 * (lo + hi))

        pred = predicted(f)
        resid = float(np.median(np.abs(pred - bl) / bl))
        conf = float(np.clip(1.0 - resid / 0.25, 0.0, 1.0)) * \
            float(np.clip(len(by) / 10.0, 0.3, 1.0))
        q = CalibrationQuality(
            method="homography", confidence=conf, n_samples=int(ys.size),
            residual_rel=resid, row_min=float(ys.min()), row_max=float(ys.max()),
            notes=(f"f={f:.1f}px solved from {len(by)} row bins; "
                   f"h={height_m:.1f}m pitch={pitch_deg:.1f}deg"))
        return GroundPlaneCalibration(height_m, pitch_deg, f, W, H, q)


class RowScaleCalibration(BaseCalibration):
    """Isotropic metres-per-pixel as a linear function of image row.

    Fallback only. Correct for a nadir view; for an oblique one it cannot
    represent the difference between across-view and along-view scale, so
    speeds acquire a direction-dependent bias. Reported confidence is capped to
    reflect that.
    """

    method = "row_scale"

    def __init__(self, slope, intercept, ref_length_m, quality=None):
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.ref = float(ref_length_m)
        self.quality = quality

    def _px(self, y):
        return np.clip(self.slope * np.asarray(y, float) + self.intercept, 6.0, 600.0)

    def scales(self, y):
        s = self.ref / self._px(y)
        return s, s

    def to_ground(self, x, y):
        # No consistent global frame exists for an isotropic row model; this is
        # a local linearisation, adequate for short-baseline displacements.
        s = self.ref / self._px(y)
        return np.asarray(x, float) * s, np.asarray(y, float) * s

    @staticmethod
    def fit(ys, px_lens, ref_length_m):
        ys = np.asarray(ys, float)
        px = np.asarray(px_lens, float)
        ok = np.isfinite(ys) & np.isfinite(px) & (px > 4)
        ys, px = ys[ok], px[ok]
        if ys.size < 100:
            return RowScaleCalibration(
                0.0, 44.0, ref_length_m,
                CalibrationQuality("row_scale", 0.15, int(ys.size), 1.0, 0, 0,
                                   "insufficient samples; constant fallback"))
        nb = 12
        edges = np.linspace(ys.min(), ys.max(), nb + 1)
        bx, bl, bw = [], [], []
        for i in range(nb):
            hi = ys <= edges[i + 1] if i == nb - 1 else ys < edges[i + 1]
            m = (ys >= edges[i]) & hi
            if m.sum() >= 25:
                bx.append(0.5 * (edges[i] + edges[i + 1]))
                bl.append(np.median(px[m]))
                bw.append(math.sqrt(m.sum()))
        if len(bx) < 3:
            med = float(np.median(px))
            return RowScaleCalibration(
                0.0, med, ref_length_m,
                CalibrationQuality("row_scale", 0.2, int(ys.size), 1.0,
                                   float(ys.min()), float(ys.max()),
                                   "too few row bins; constant fallback"))
        bx, bl, bw = np.array(bx), np.array(bl), np.array(bw)
        A = np.stack([bx * bw, bw], 1)
        slope, intercept = np.linalg.lstsq(A, bl * bw, rcond=None)[0]
        if slope * bx.max() + intercept <= 5 or slope * bx.min() + intercept <= 5:
            slope, intercept = 0.0, float(np.median(px))
        pred = slope * bx + intercept
        resid = float(np.median(np.abs(pred - bl) / bl))
        q = CalibrationQuality(
            "row_scale", float(np.clip(0.6 - resid, 0.1, 0.6)), int(ys.size),
            resid, float(ys.min()), float(ys.max()),
            "isotropic; oblique view makes speed direction-dependent")
        return RowScaleCalibration(slope, intercept, ref_length_m, q)


class NoCalibration(BaseCalibration):
    method = "uncalibrated"

    def __init__(self):
        self.quality = CalibrationQuality("uncalibrated", 0.0, 0, float("nan"),
                                          0, 0, "no metric scale available")

    def scales(self, y):
        n = np.full(np.shape(y), np.nan, float) if np.ndim(y) else np.nan
        return n, n

    def to_ground(self, x, y):
        n = np.full(np.shape(x), np.nan, float) if np.ndim(x) else np.nan
        return n, n


def build(ys, px_lens, telemetry, W, H, ref_length_m, prefer="auto"):
    """Choose and fit the best calibration the available data supports."""
    tel_ok = bool(telemetry) and all(
        telemetry.get(k) is not None for k in ("rel_alt", "gb_pitch"))
    if tel_ok:
        h = float(telemetry["rel_alt"])
        pitch = float(telemetry["gb_pitch"])
        tel_ok = h > 3.0 and 5.0 < abs(pitch) < 89.5

    if prefer in ("auto", "homography") and tel_ok:
        cal = GroundPlaneCalibration.fit(ys, px_lens, h, pitch, W, H, ref_length_m)
        if cal is not None and cal.quality.confidence > 0.25:
            return cal
    if prefer in ("auto", "homography", "row_scale"):
        return RowScaleCalibration.fit(ys, px_lens, ref_length_m)
    return NoCalibration()


def describe(cal) -> dict:
    """Serialisable summary for the run report."""
    d = {"method": cal.method}
    if getattr(cal, "quality", None) is not None:
        d.update(cal.quality.as_dict())
    if isinstance(cal, GroundPlaneCalibration):
        d.update({
            "height_m": round(cal.h, 3),
            "gimbal_pitch_deg": round(cal.pitch, 2),
            "focal_px": round(cal.f, 1),
            "horizon_row": round(float(cal.horizon_row()), 1),
        })
        for label, y in (("top", 0.0), ("mid", cal.H / 2.0), ("bottom", cal.H - 1.0)):
            sx, sy = cal.scales(y)
            d[f"m_per_px_x_{label}"] = round(float(sx), 5)
            d[f"m_per_px_y_{label}"] = round(float(sy), 5)
            d[f"anisotropy_{label}"] = round(float(sy / sx), 3)
    elif isinstance(cal, RowScaleCalibration):
        d.update({"slope_px_per_row": round(cal.slope, 6),
                  "intercept_px": round(cal.intercept, 2)})
    return d


def save(cal, path):
    import pathlib
    pathlib.Path(path).write_text(json.dumps(describe(cal), indent=2))
