"""L4a: bind the local ground frame to the Earth.

The calibration in `calibration.py` produces metres in a *camera-local* frame:
X across the view, Y away from the camera along the ground. That is enough for
speed and acceleration, but not for anything map-native. This module rotates
that frame by the gimbal's compass yaw and anchors it at the drone's GPS
position, giving WGS84 coordinates for every trajectory point.

CONVENTION AND ITS RISK
-----------------------
DJI reports `gb_yaw` as a compass bearing: 0 = true north, positive clockwise,
range -180..180. The local +Y axis (away from the camera) is taken to point
along that bearing. If a particular airframe or firmware reports yaw against a
different datum, every absolute coordinate rotates by the offset while all
*relative* geometry -- lane spacing, queue length, trajectory shape -- stays
exactly right. `yaw_offset_deg` exists to correct that without touching
anything else, and `GeoReference.describe()` always states the assumption so a
reader can check it against a map rather than trusting it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


def metres_per_degree(lat_deg: float) -> tuple[float, float]:
    """Local metres per degree of latitude and longitude at a given latitude.

    Uses the standard ellipsoidal radii rather than a spherical approximation;
    at Pune's latitude the difference is about 0.3%, which is 30 cm over a
    100 m scene -- small, but free to get right.
    """
    lat = math.radians(lat_deg)
    e2 = WGS84_F * (2 - WGS84_F)
    s = math.sin(lat)
    w = math.sqrt(1 - e2 * s * s)
    m_per_deg_lat = math.pi * WGS84_A * (1 - e2) / (180 * w ** 3)
    m_per_deg_lon = math.pi * WGS84_A * math.cos(lat) / (180 * w)
    return m_per_deg_lat, m_per_deg_lon


@dataclass
class GeoReference:
    lat0: float                 # drone latitude  (local frame origin)
    lon0: float                 # drone longitude
    yaw_deg: float              # compass bearing of the local +Y axis
    alt_m: float = 0.0
    yaw_spread_deg: float = 0.0   # observed wander over the clip
    pos_drift_m: float = 0.0
    n_samples: int = 0

    def __post_init__(self):
        self._mlat, self._mlon = metres_per_degree(self.lat0)
        psi = math.radians(self.yaw_deg)
        self._sin, self._cos = math.sin(psi), math.cos(psi)

    # ---- local <-> ENU ----------------------------------------------------
    def to_enu(self, X, Y):
        """Camera-local (X right, Y forward) -> (east, north) metres."""
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        east = X * self._cos + Y * self._sin
        north = -X * self._sin + Y * self._cos
        return east, north

    def from_enu(self, east, north):
        east = np.asarray(east, float)
        north = np.asarray(north, float)
        X = east * self._cos - north * self._sin
        Y = east * self._sin + north * self._cos
        return X, Y

    # ---- local <-> WGS84 --------------------------------------------------
    def to_wgs84(self, X, Y):
        east, north = self.to_enu(X, Y)
        return self.lat0 + north / self._mlat, self.lon0 + east / self._mlon

    def to_local(self, lat, lon):
        north = (np.asarray(lat, float) - self.lat0) * self._mlat
        east = (np.asarray(lon, float) - self.lon0) * self._mlon
        return self.from_enu(east, north)

    # ---- bearings ---------------------------------------------------------
    def local_bearing_to_compass(self, bearing_deg):
        """Bearing in the local frame (0 = +Y) -> true compass bearing."""
        return (np.asarray(bearing_deg, float) + self.yaw_deg) % 360.0

    def heading_to_compass(self, heading_deg):
        """`heading_deg` from kinematics is atan2(vy, vx), i.e. measured from
        the +X axis counter-clockwise. Convert to a compass bearing."""
        local = (90.0 - np.asarray(heading_deg, float)) % 360.0
        return self.local_bearing_to_compass(local)

    # ---- quality ----------------------------------------------------------
    def position_uncertainty_m(self, range_m: float) -> float:
        """Absolute positional uncertainty at a given range from the camera.

        Dominated by gimbal-yaw wander, which rotates the whole scene about the
        camera; relative geometry within the scene is unaffected.
        """
        return float(range_m * math.radians(self.yaw_spread_deg))

    @staticmethod
    def from_telemetry(entries, yaw_offset_deg: float = 0.0) -> "GeoReference | None":
        rec = [e for e in entries
               if e.get("lat") is not None and e.get("lon") is not None
               and e.get("gb_yaw") is not None]
        if not rec:
            return None
        lat = np.array([e["lat"] for e in rec], float)
        lon = np.array([e["lon"] for e in rec], float)
        # circular median, so a clip straddling +/-180 does not average to zero
        yaw = np.radians(np.array([e["gb_yaw"] for e in rec], float))
        yaw_med = math.degrees(math.atan2(float(np.median(np.sin(yaw))),
                                          float(np.median(np.cos(yaw)))))
        yaw_unwrapped = np.degrees(np.unwrap(yaw))
        alt = np.array([e.get("rel_alt") or 0.0 for e in rec], float)

        mlat, mlon = metres_per_degree(float(np.median(lat)))
        drift = float(math.hypot((lat.max() - lat.min()) * mlat,
                                 (lon.max() - lon.min()) * mlon))
        return GeoReference(
            lat0=float(np.median(lat)), lon0=float(np.median(lon)),
            yaw_deg=(yaw_med + yaw_offset_deg) % 360.0,
            alt_m=float(np.median(alt)),
            yaw_spread_deg=float(yaw_unwrapped.max() - yaw_unwrapped.min()),
            pos_drift_m=drift, n_samples=len(rec))

    def describe(self) -> dict:
        return {
            "origin_lat": round(self.lat0, 8),
            "origin_lon": round(self.lon0, 8),
            "local_plus_y_compass_bearing_deg": round(self.yaw_deg, 2),
            "camera_altitude_m": round(self.alt_m, 2),
            "gimbal_yaw_spread_deg": round(self.yaw_spread_deg, 3),
            "drone_position_drift_m": round(self.pos_drift_m, 3),
            "position_uncertainty_at_100m_m": round(self.position_uncertainty_m(100), 2),
            "telemetry_samples": self.n_samples,
            "assumption": ("gb_yaw is a true-north compass bearing, positive "
                           "clockwise; correct with yaw_offset_deg if a map "
                           "overlay shows a constant rotation"),
        }


# --------------------------------------------------------------------- GeoJSON
def _feature(geometry, props):
    return {"type": "Feature", "geometry": geometry, "properties": props}


def trajectories_geojson(df, geo: GeoReference, simplify_m: float = 1.0,
                         max_tracks: int | None = None) -> dict:
    """Every moving trajectory as a WGS84 LineString.

    Points closer together than `simplify_m` are dropped: a vehicle sampled at
    30 fps produces thousands of near-identical coordinates, which bloats the
    file without adding shape.
    """
    feats = []
    ids = df["display_id"].dropna().unique()
    if max_tracks:
        ids = ids[:max_tracks]
    for did in ids:
        g = df[df["display_id"] == did].sort_values("frame")
        X = g["world_x_m"].to_numpy()
        Y = g["world_y_m"].to_numpy()
        ok = np.isfinite(X) & np.isfinite(Y)
        X, Y = X[ok], Y[ok]
        if X.size < 2:
            continue
        keep = [0]
        for i in range(1, X.size):
            if math.hypot(X[i] - X[keep[-1]], Y[i] - Y[keep[-1]]) >= simplify_m:
                keep.append(i)
        if len(keep) < 2:
            continue
        lat, lon = geo.to_wgs84(X[keep], Y[keep])
        sp = g.loc[ok, "speed_kmh"].to_numpy()[keep]
        sp = sp[np.isfinite(sp)]        # a track may have no measurable speed
        feats.append(_feature(
            {"type": "LineString",
             "coordinates": [[round(float(o), 8), round(float(a), 8)]
                             for a, o in zip(lat, lon)]},
            {"id": int(did),
             "mode": str(g["mode"].iloc[0]),
             "median_speed_kmh": round(float(np.median(sp)), 1) if sp.size else None,
             "points": len(keep)}))
    return {"type": "FeatureCollection", "features": feats}


def points_geojson(rows, geo: GeoReference, props_fn) -> dict:
    feats = []
    for r in rows:
        lat, lon = geo.to_wgs84(r["X"], r["Y"])
        feats.append(_feature(
            {"type": "Point", "coordinates": [round(float(lon), 8), round(float(lat), 8)]},
            props_fn(r)))
    return {"type": "FeatureCollection", "features": feats}


def save_geojson(obj, path):
    import pathlib
    pathlib.Path(path).write_text(json.dumps(obj))
