"""Loads the pipeline outputs once and serves them in shapes the UI needs.

The L2 track table is 2 million rows. The browser never wants all of it: at any
instant it needs the ~50 objects visible in the current frame, and for the
charts it needs pre-aggregated summaries. So this module does two things at
startup -- build a compact per-frame overlay index, and compute every aggregate
once -- and then answers requests out of memory.

Overlay data is chunked by time rather than streamed. The client controls
playback, so it always knows which chunk it needs; plain HTTP lets the browser
cache what it has already seen, which makes scrubbing backwards free.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MODE_IDX = {"pedestrian": 0, "motorcycle": 1, "car": 2, "LGV": 3,
            "truck": 4, "HGV": 5, "bus": 6}
IDX_MODE = {v: k for k, v in MODE_IDX.items()}
STATE_IDX = {"moving": 0, "temporarily_stopped": 1, "parked": 2, "unknown": 3}

CHUNK_S = 10.0          # seconds of overlay data per HTTP chunk


@dataclass
class Meta:
    name: str
    video_url: str
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float
    chunk_s: float


class Store:
    def __init__(self, name: str = "intersection_1080p",
                 video: str = "Dataset_Video/Intersection_1080p.MP4",
                 outdir: str = "output"):
        self.name = name
        self.video_path = ROOT / video
        self.out = ROOT / outdir
        self.ready = False
        self.error: str | None = None
        self._chunks: dict[int, dict] = {}
        self.insights: dict = {}
        self.geo = None
        self.net = None

    # ------------------------------------------------------------------ load
    def load(self):
        import cv2
        import sys
        sys.path.insert(0, str(ROOT))
        from vtrack import telemetry as tele, network as netmod, aggregate as agg
        from vtrack.georef import GeoReference

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {self.video_path}")
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        df = pd.read_parquet(self.out / f"{self.name}_l2_tracks.parquet")
        self.moving = df[~df["parked"]].copy()
        self._build_chunks(self.moving)

        srt = self.video_path.with_suffix(".srt")
        entries = tele.parse_srt(srt) if srt.exists() else []
        self.geo = GeoReference.from_telemetry(entries) if entries else None
        self.net = netmod.build(self.moving, self.geo, frame_wh=(self.W, self.H))
        self._build_insights(agg, netmod)
        self.ready = True

    # ------------------------------------------------- per-frame overlay index
    def _build_chunks(self, mv: pd.DataFrame):
        """Group the overlay rows into fixed time chunks.

        Only the fields the canvas actually draws are kept, and coordinates are
        rounded to whole pixels: at 1080p a sub-pixel box edge is invisible, and
        the rounding roughly halves the payload.
        """
        cols = ["frame", "display_id", "mode", "x1", "y1", "x2", "y2",
                "speed_kmh", "motion_state", "longitudinal_acceleration_mps2"]
        d = mv[[c for c in cols if c in mv.columns]].dropna(subset=["display_id"])
        d = d.sort_values("frame")
        d["chunk"] = (d["frame"] / self.fps // CHUNK_S).astype(int)

        for ch, g in d.groupby("chunk"):
            by_frame: dict[int, list] = {}
            for r in g.itertuples(index=False):
                spd = getattr(r, "speed_kmh", float("nan"))
                acc = getattr(r, "longitudinal_acceleration_mps2", float("nan"))
                by_frame.setdefault(int(r.frame), []).append([
                    int(r.display_id),
                    MODE_IDX.get(r.mode, 2),
                    int(round(r.x1)), int(round(r.y1)),
                    int(round(r.x2)), int(round(r.y2)),
                    None if spd != spd else round(float(spd), 1),
                    STATE_IDX.get(getattr(r, "motion_state", "unknown"), 3),
                    None if acc != acc else round(float(acc), 1),
                ])
            self._chunks[int(ch)] = by_frame

    def chunk(self, idx: int) -> dict:
        return self._chunks.get(int(idx), {})

    def n_chunks(self) -> int:
        return int(math.ceil(self.n_frames / self.fps / CHUNK_S))

    # ------------------------------------------------------------- aggregates
    def _build_insights(self, agg, netmod):
        N, fps, mv = self.net, self.fps, self.moving

        def frame(df):
            return json.loads(df.to_json(orient="records")) if len(df) else []

        od = agg.od_summary(N)
        approaches = [g.as_dict() for g in N.gates]
        for a in approaches:
            if self.geo is not None and a["bearing_in_deg"] is not None:
                a["compass_in_deg"] = round(
                    float(self.geo.local_bearing_to_compass(a["bearing_in_deg"])), 0)

        profiles, fds = {}, {}
        for (o, d) in list(N.corridors.keys()):
            sp = agg.speed_profile(mv, N, o, d, bin_m=15.0, speed_limit_kmh=40.0)
            if len(sp):
                profiles[f"{o}->{d}"] = frame(sp)
            fd = agg.fundamental_diagram(mv, N, o, d, fps=fps)
            if len(fd):
                fds[f"{o}->{d}"] = frame(fd)

        queues = {}
        for g in N.gates:
            q = agg.queue_lengths(mv, N, fps, g.name, sample_s=1.0)
            if len(q):
                queues[g.name] = frame(q[["t_s", "queue_vehicles", "queue_length_m"]])

        self.insights = {
            "approaches": approaches,
            "od": frame(od),
            "od_matrix": json.loads(N.od.to_json(orient="split")) if len(N.od) else {},
            "modal_split": frame(agg.modal_split(N)),
            "lane_volumes": frame(agg.lane_volumes(N)),
            "approach_volumes": frame(agg.approach_volumes(N, fps, 60.0)),
            "movement_counts": frame(agg.movement_counts(N, fps, 60.0)),
            "speed_profiles": profiles,
            "fundamental": fds,
            "queues": queues,
            "georeference": self.geo.describe() if self.geo else None,
            "network": netmod.describe(N),
            "totals": {
                "tracks_moving": int(mv["track_id"].nunique()),
                "movements_assigned": int(len(N.movements)),
                "duration_s": round(self.n_frames / self.fps, 1),
            },
        }

    # ------------------------------------------------------------------ misc
    def meta(self) -> Meta:
        return Meta(name=self.name, video_url="/api/video",
                    width=self.W, height=self.H, fps=round(self.fps, 4),
                    frames=self.n_frames,
                    duration_s=round(self.n_frames / self.fps, 2),
                    chunk_s=CHUNK_S)

    def geojson(self, kind: str) -> dict:
        p = self.out / "map" / f"{kind}.geojson"
        if not p.exists():
            return {"type": "FeatureCollection", "features": []}
        return json.loads(p.read_text())

    def track_detail(self, display_id: int) -> dict:
        g = self.moving[self.moving["display_id"] == display_id].sort_values("frame")
        if g.empty:
            return {}
        mvrow = self.net.movements
        tid = int(g["track_id"].iloc[0])
        m = mvrow.loc[tid] if tid in mvrow.index else None
        sp = g["speed_kmh"].to_numpy()
        sp = sp[np.isfinite(sp)]
        pts = []
        if self.geo is not None:
            X = g["world_x_m"].to_numpy()
            Y = g["world_y_m"].to_numpy()
            ok = np.isfinite(X) & np.isfinite(Y)
            lat, lon = self.geo.to_wgs84(X[ok][::10], Y[ok][::10])
            pts = [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lon, lat)]
        return {
            "display_id": int(display_id),
            "mode": str(g["mode"].iloc[0]),
            "mode_confidence": float(g["mode_confidence"].iloc[0]),
            "classification_source": str(g["classification_source"].iloc[0]),
            "estimated_length_m": (None if not np.isfinite(g["estimated_length_m"].iloc[0])
                                   else round(float(g["estimated_length_m"].iloc[0]), 2)),
            "first_frame": int(g["frame"].min()),
            "last_frame": int(g["frame"].max()),
            "median_speed_kmh": round(float(np.median(sp)), 1) if sp.size else None,
            "max_speed_kmh": round(float(sp.max()), 1) if sp.size else None,
            "origin": None if m is None else str(m.get("origin", "")),
            "destination": None if m is None else str(m.get("destination", "")),
            "turn": None if m is None else str(m.get("turn", "")),
            "path": pts,
        }
