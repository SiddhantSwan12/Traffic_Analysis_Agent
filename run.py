#!/usr/bin/env python
"""FlytBase drone traffic pipeline: detect -> track -> post-process -> render.

Typical use:

    # 20-second preview, to judge quality fast
    python run.py --video Dataset_Video/Intersection_1080p.MP4 --preview 20

    # full run
    python run.py --video Dataset_Video/Intersection_1080p.MP4

    # re-tune the tracker without re-running YOLO (the expensive part)
    python run.py --video ... --from-detections

    # re-render only (tweak trails/labels/parked visibility)
    python run.py --video ... --render-only --show-parked
"""
from __future__ import annotations
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vtrack.config import Config
from vtrack import telemetry as tele


# --------------------------------------------------------------------- utils
class FrameReader:
    """Decode in a background thread so the GPU never waits on cv2."""

    def __init__(self, path, stride=1, max_frames=None, qsize=8):
        self.path, self.stride, self.max_frames = str(path), stride, max_frames
        self.q = queue.Queue(maxsize=qsize)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.n = self.total if max_frames is None else min(self.total, max_frames)
        self._t = None

    def __iter__(self):
        self._t = threading.Thread(target=self._work, daemon=True)
        self._t.start()
        while True:
            item = self.q.get()
            if item is None:
                break
            yield item

    def _work(self):
        cap = cv2.VideoCapture(self.path)
        i = 0
        while i < self.n:
            ok, frame = cap.read()
            if not ok:
                break
            if i % self.stride == 0:
                self.q.put((i, frame))
            i += 1
        cap.release()
        self.q.put(None)


class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.t0 = total, label, time.time()
        self.last = 0.0

    def __call__(self, i, extra=""):
        now = time.time()
        if now - self.last < 0.5 and i < self.total:
            return
        self.last = now
        el = now - self.t0
        rate = i / max(el, 1e-6)
        eta = (self.total - i) / max(rate, 1e-6)
        bar = int(28 * i / max(self.total, 1))
        sys.stdout.write(
            f"\r  {self.label} [{'#' * bar}{'.' * (28 - bar)}] "
            f"{i}/{self.total}  {rate:5.1f} f/s  eta {eta/60:5.1f}m  {extra}   ")
        sys.stdout.flush()

    def done(self):
        el = time.time() - self.t0
        sys.stdout.write(f"\r  {self.label}: {self.total} frames in "
                         f"{el/60:.1f} min ({self.total/max(el,1e-6):.1f} f/s)"
                         + " " * 40 + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------- main stages
def stage_detect_track(cfg, reader, det_cache: pd.DataFrame | None):
    """Single decode pass: YOLO (or cached detections) -> tracker."""
    from vtrack.tracker import Tracker

    if det_cache is None:
        from vtrack.detector import TiledDetector
        det = TiledDetector(cfg.detect, reader.W, reader.H)
        names = det.names
        print(f"  model classes: {list(names.values())}")
        print(f"  tiles: {len(det.tiles)} x {cfg.detect.tile}px "
              f"(overlap {cfg.detect.tile_overlap}px)")
    else:
        det = None
        names = {int(k): v for k, v in
                 det_cache[["cls_id", "cls_name"]].drop_duplicates().values}
        cache_by_frame = {int(f): g for f, g in det_cache.groupby("frame")}

    excl = [tuple(map(int, r)) for r in cfg.exclude_rects]
    trk = Tracker(cfg.track, names)

    det_rows, trk_rows = [], []
    prog = Progress(len(range(0, reader.n, cfg.detect.stride)),
                    "detect+track" if det is None or True else "track")

    for k, (fi, frame) in enumerate(reader):
        if det is not None:
            boxes, scores, clss = det(frame)
            if len(boxes):
                det_rows.append(np.column_stack([
                    np.full(len(boxes), fi), boxes, scores, clss]))
        else:
            g = cache_by_frame.get(fi)
            if g is None or len(g) == 0:
                boxes = np.zeros((0, 4), np.float32)
                scores = np.zeros((0,), np.float32)
                clss = np.zeros((0,), np.int32)
            else:
                boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
                scores = g["conf"].to_numpy(np.float32)
                clss = g["cls_id"].to_numpy(np.int32)

        if excl and len(boxes):
            cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
            cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
            keep = np.ones(len(boxes), bool)
            for x1, y1, x2, y2 in excl:
                keep &= ~((cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2))
            boxes, scores, clss = boxes[keep], scores[keep], clss[keep]

        for tid, x1, y1, x2, y2, sc, cid, cname in trk.update(frame, boxes, scores, clss):
            trk_rows.append((fi, tid, x1, y1, x2, y2, sc, cid, cname))

        prog(k + 1, f"tracks {len(trk.tracks):3d} gallery {len(trk.gallery):3d}")
    prog.done()

    dets = None
    if det_rows:
        a = np.concatenate(det_rows)
        dets = pd.DataFrame({
            "frame": a[:, 0].astype(np.int32),
            "x1": a[:, 1], "y1": a[:, 2], "x2": a[:, 3], "y2": a[:, 4],
            "conf": a[:, 5].astype(np.float32),
            "cls_id": a[:, 6].astype(np.int16),
        })
        dets["cls_name"] = dets["cls_id"].map(names)

    tracks = pd.DataFrame(trk_rows, columns=[
        "frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls_id", "cls_name"])
    return dets, tracks


def load_table(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    csv = path.with_suffix(".csv")
    return pd.read_csv(csv) if csv.exists() else None


def save_table(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", default=None, help="output prefix (default: video stem)")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--preview", type=float, default=None,
                    help="only process the first N seconds")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--from-detections", action="store_true",
                    help="reuse cached detections, re-run tracking onwards")
    ap.add_argument("--from-tracks", action="store_true",
                    help="reuse raw tracks, re-run post-processing onwards")
    ap.add_argument("--render-only", action="store_true",
                    help="reuse final tracks, re-render the video only")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--show-parked", action="store_true")
    ap.add_argument("--no-trails", action="store_true")
    ap.add_argument("--labels", choices=["full", "compact", "id"], default=None)
    ap.add_argument("--no-cmc", action="store_true")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--exclude", default=None,
                    help="JSON list of [x1,y1,x2,y2] rectangles to ignore")
    args = ap.parse_args()

    cfg = Config()
    cfg.video = Path(args.video)
    cfg.name = args.name or cfg.video.stem.lower()
    cfg.outdir = Path(args.outdir)
    if args.model:   cfg.detect.model = args.model
    if args.device:  cfg.detect.device = args.device
    if args.conf is not None: cfg.detect.conf = args.conf
    if args.stride:  cfg.detect.stride = args.stride
    if args.encoder: cfg.render.encoder = args.encoder
    if args.show_parked: cfg.render.show_parked = True
    if args.no_trails:   cfg.render.trail_frames = 0
    if args.labels:      cfg.render.label_mode = args.labels
    if args.no_cmc:      cfg.track.use_cmc = False
    if args.exclude:     cfg.exclude_rects = json.loads(args.exclude)
    paths = cfg.paths()

    probe = cv2.VideoCapture(str(cfg.video))
    if not probe.isOpened():
        sys.exit(f"cannot open video: {cfg.video}")
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    VW = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    max_frames = args.max_frames
    if args.preview:
        max_frames = int(args.preview * fps)
    n_proc = min(n_total, max_frames or n_total)

    print(f"\n=== {cfg.video.name} ===")
    print(f"  {n_total} frames @ {fps:.2f} fps  ({n_total/fps/60:.1f} min)")
    if n_proc < n_total:
        print(f"  PREVIEW MODE: first {n_proc} frames ({n_proc/fps:.0f}s)")

    tel0 = {}
    srt = tele.find_srt(cfg.video)
    if srt:
        ents = tele.parse_srt(srt)
        tel0 = ents[0] if ents else {}
        print(f"  telemetry: {srt.name}, {len(ents)} entries "
              f"({'aligned' if len(ents) == n_total else 'will be padded'})")

    # ---- stages
    if args.render_only:
        final = load_table(paths["final"])
        if final is None:
            sys.exit("no final tracks found - run without --render-only first")
        report = json.loads(paths["report"].read_text()) if paths["report"].exists() else {}
    elif args.from_tracks:
        tracks = load_table(paths["tracks"])
        if tracks is None:
            sys.exit("no raw tracks found - run without --from-tracks first")
        print(f"  reusing {len(tracks):,} raw track rows "
              f"({tracks['track_id'].nunique():,} ids)")
        from vtrack.analytics import analyze, save
        final, report, cal = analyze(tracks, fps, VW, VH, tel0, cfg)
        save(final, report, cal, paths)
    else:
        det_cache = None
        if args.from_detections:
            det_cache = load_table(paths["raw"])
            if det_cache is None:
                sys.exit("no cached detections found - run without --from-detections")
            print(f"  reusing {len(det_cache):,} cached detections")

        reader = FrameReader(cfg.video, cfg.detect.stride, max_frames,
                             cfg.detect.reader_queue)
        t0 = time.time()
        dets, tracks = stage_detect_track(cfg, reader, det_cache)
        if dets is not None:
            save_table(dets, paths["raw"])
            print(f"  detections: {len(dets):,} rows -> {paths['raw'].name}")
        save_table(tracks, paths["tracks"])
        print(f"  raw tracks: {len(tracks):,} rows, "
              f"{tracks['track_id'].nunique():,} ids -> {paths['tracks'].name}")

        from vtrack.analytics import analyze, save
        final, report, cal = analyze(tracks, fps, VW, VH, tel0, cfg)
        save(final, report, cal, paths)
        print(f"  detect+track+post took {(time.time()-t0)/60:.1f} min")

    print("\n--- summary " + "-" * 46)
    for k in ("tracks_total", "tracks_moving", "tracks_parked",
              "tracks_dropped_short", "median_track_seconds", "p90_track_seconds"):
        if k in report:
            print(f"  {k:<24} {report[k]}")
    if "counts_by_mode" in report:
        print("  unique road users by mode (moving only):")
        for m, c in report["counts_by_mode"].items():
            print(f"      {m:<12} {c}")
    if "scale" in report:
        s = report["scale"]
        print(f"  scale: {s['m_per_px_top']:.4f} m/px (top) -> "
              f"{s['m_per_px_bottom']:.4f} m/px (bottom), "
              f"{s['n_car_samples']:,} car samples")
    print("-" * 58)

    if args.no_render:
        return

    from vtrack.render import render
    print(f"\n  rendering -> {paths['video'].name}")
    prog = Progress(n_proc, "render")
    info = render(cfg.video, final, paths["video"], fps, cfg,
                  max_frames=n_proc, progress=lambda i, t: prog(i))
    prog.done()
    size_mb = paths["video"].stat().st_size / 1e6
    print(f"  encoder={info['encoder']}  ids shown={info['unique_ids_shown']}  "
          f"{size_mb:.0f} MB")
    print(f"\n  DONE -> {paths['video']}\n")


if __name__ == "__main__":
    main()
