"""VisDrone -> hackathon 7-class taxonomy, plus metric scale calibration.

The detector speaks VisDrone's 10 classes. The brief asks for 7 transport
modes. Two of the target classes (LGV / HGV) are *weight* categories that no
aerial detector can read off appearance alone, so they are resolved from
physical vehicle length, recovered by calibrating pixels to metres against the
known length of a car.
"""
from __future__ import annotations
import numpy as np

# The 7 modes required by the brief.
MODES = ["pedestrian", "motorcycle", "car", "LGV", "truck", "HGV", "bus"]
MODE_IDX = {m: i for i, m in enumerate(MODES)}

# VisDrone label -> coarse mode. `van` and `truck` are provisional: the size
# refinement below can promote or demote them.
VISDRONE_TO_MODE = {
    "pedestrian":      "pedestrian",
    "people":          "pedestrian",
    "bicycle":         "motorcycle",   # no dedicated cycle class in the brief
    "motor":           "motorcycle",
    "tricycle":        "motorcycle",   # auto-rickshaw class of vehicle
    "awning-tricycle": "motorcycle",
    "car":             "car",
    "van":             "LGV",          # light goods vehicle / minibus
    "truck":           "truck",        # rigid truck, promoted to HGV if long
    "bus":             "bus",
    "ignored":         None,
    "others":          None,
}

# Display colours (BGR) per mode.
MODE_COLOR = {
    "pedestrian": (80, 220, 255),
    "motorcycle": (90, 255, 140),
    "car":        (255, 170, 60),
    "LGV":        (255, 110, 220),
    "truck":      (70, 130, 255),
    "HGV":        (40, 70, 235),
    "bus":        (60, 240, 240),
}

# Length thresholds in metres, applied to a track's *median* projected length.
LEN_CAR_MAX = 6.2    # above this a "car" is really a van/pickup
LEN_LGV_MAX = 7.5    # above this an "LGV" is a rigid truck
LEN_TRUCK_MAX = 9.5  # above this a "truck" is an HGV / articulated
LEN_BUS_MIN = 7.0    # below this a "bus" is probably a minibus -> LGV


def map_mode(visdrone_label: str) -> str | None:
    return VISDRONE_TO_MODE.get(visdrone_label)


def refine_by_size(mode: str, length_m: float | None) -> str:
    """Promote/demote a coarse mode using physical length."""
    if mode is None or length_m is None or not np.isfinite(length_m):
        return mode
    if mode == "car":
        return "LGV" if length_m > LEN_CAR_MAX else "car"
    if mode == "LGV":
        if length_m > LEN_TRUCK_MAX:
            return "HGV"
        return "truck" if length_m > LEN_LGV_MAX else "LGV"
    if mode in ("truck", "HGV"):
        return "HGV" if length_m > LEN_TRUCK_MAX else "truck"
    if mode == "bus":
        return "bus" if length_m >= LEN_BUS_MIN else "LGV"
    return mode


class ScaleModel:
    """Metres-per-pixel as a function of image row.

    The gimbal sits at ~-63 deg, so the view is oblique and scale varies down
    the frame. We fit projected car length (px) against y and invert it. Using
    max(w, h) of the axis-aligned box is deliberate: for a 4.3 x 1.8 m box that
    quantity stays within ~1% of 4.3 m at every rotation angle, which makes it
    an unusually stable ruler.
    """

    def __init__(self, slope: float, intercept: float, car_length_m: float,
                 y_range: tuple[float, float], n_samples: int, fallback: float):
        self.slope = slope
        self.intercept = intercept
        self.car_length_m = car_length_m
        self.y_range = y_range
        self.n_samples = n_samples
        self.fallback = fallback

    @classmethod
    def fit(cls, ys: np.ndarray, lengths_px: np.ndarray, car_length_m: float,
            n_bins: int = 12, min_per_bin: int = 25) -> "ScaleModel":
        ys = np.asarray(ys, float)
        lp = np.asarray(lengths_px, float)
        ok = np.isfinite(ys) & np.isfinite(lp) & (lp > 4)
        ys, lp = ys[ok], lp[ok]
        if ys.size < 200:
            med = float(np.median(lp)) if lp.size else 44.0
            return cls(0.0, med, car_length_m, (0, 1080), ys.size,
                       car_length_m / max(med, 1e-6))

        edges = np.linspace(ys.min(), ys.max(), n_bins + 1)
        bx, by, bw = [], [], []
        for i in range(n_bins):
            m = (ys >= edges[i]) & (ys < edges[i + 1] if i < n_bins - 1 else ys <= edges[i + 1])
            if m.sum() >= min_per_bin:
                bx.append(0.5 * (edges[i] + edges[i + 1]))
                by.append(np.median(lp[m]))
                bw.append(m.sum())
        if len(bx) < 3:
            med = float(np.median(lp))
            return cls(0.0, med, car_length_m, (float(ys.min()), float(ys.max())),
                       ys.size, car_length_m / max(med, 1e-6))

        bx, by, bw = np.array(bx), np.array(by), np.sqrt(np.array(bw, float))
        A = np.stack([bx * bw, bw], 1)
        slope, intercept = np.linalg.lstsq(A, by * bw, rcond=None)[0]
        # A negative slope steep enough to cross zero inside the frame means the
        # fit is garbage; fall back to a constant.
        if slope * ys.max() + intercept <= 5 or slope * ys.min() + intercept <= 5:
            slope, intercept = 0.0, float(np.median(lp))
        return cls(float(slope), float(intercept), car_length_m,
                   (float(ys.min()), float(ys.max())), int(ys.size),
                   car_length_m / max(float(np.median(lp)), 1e-6))

    def car_px_at(self, y):
        y = np.asarray(y, float)
        return np.clip(self.slope * y + self.intercept, 8.0, 400.0)

    def m_per_px(self, y):
        return self.car_length_m / self.car_px_at(y)

    def length_m(self, box_len_px, y):
        return np.asarray(box_len_px, float) * self.m_per_px(y)

    def describe(self) -> dict:
        y0, y1 = self.y_range
        return {
            "car_px_at_top": round(float(self.car_px_at(y0)), 1),
            "car_px_at_bottom": round(float(self.car_px_at(y1)), 1),
            "m_per_px_top": round(float(self.m_per_px(y0)), 5),
            "m_per_px_bottom": round(float(self.m_per_px(y1)), 5),
            "n_car_samples": self.n_samples,
        }
