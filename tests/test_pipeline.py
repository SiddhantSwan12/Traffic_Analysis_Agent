"""Synthetic end-to-end checks for the parts of the pipeline that do not need
a GPU or the real video. Run with:  python tests/test_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from vtrack.config import Config
from vtrack.taxonomy import ScaleModel, map_mode, refine_by_size, MODES
from vtrack.tracker import Tracker, iou_matrix, descriptors

W, H, FPS = 1920, 1080, 30.0
VISDRONE = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
            5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"}
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------------ taxonomy
def test_taxonomy():
    print("\n[taxonomy]")
    modes = {map_mode(v) for v in VISDRONE.values()}
    check("all 10 VisDrone classes map into the 7 modes",
          modes <= set(MODES) and None not in modes, sorted(modes))
    check("size refinement: long van -> HGV", refine_by_size("LGV", 10.0) == "HGV")
    check("size refinement: mid van -> truck", refine_by_size("LGV", 8.0) == "truck")
    check("size refinement: short van stays LGV", refine_by_size("LGV", 5.0) == "LGV")
    check("size refinement: long truck -> HGV", refine_by_size("truck", 12.0) == "HGV")
    check("size refinement: big car -> LGV", refine_by_size("car", 6.8) == "LGV")
    check("size refinement: normal car stays car", refine_by_size("car", 4.4) == "car")
    check("size refinement survives NaN", refine_by_size("car", float("nan")) == "car")


def test_scale():
    print("\n[scale calibration]")
    # ground truth: cars appear larger toward the bottom of an oblique frame
    rng = np.random.default_rng(0)
    ys = rng.uniform(0, H, 4000)
    true_px = 30 + 0.025 * ys
    obs = true_px * rng.normal(1.0, 0.08, ys.size)
    sm = ScaleModel.fit(ys, obs, car_length_m=4.3)
    err_top = abs(sm.car_px_at(50) - (30 + 0.025 * 50))
    err_bot = abs(sm.car_px_at(1000) - (30 + 0.025 * 1000))
    check("recovers perspective gradient within 2 px",
          err_top < 2 and err_bot < 2, f"top {err_top:.2f}px bot {err_bot:.2f}px")
    check("m/px is larger at the top (objects smaller there)",
          sm.m_per_px(50) > sm.m_per_px(1000))
    check("degenerate input falls back to a constant",
          np.isfinite(ScaleModel.fit(np.zeros(5), np.zeros(5), 4.3).m_per_px(500)))


# -------------------------------------------------------------------- geometry
def test_tiles():
    print("\n[tile geometry]")
    from vtrack.detector import make_tiles
    tiles = make_tiles(W, H, 640, 200, 160)
    check("1920x1080 tiles into 8", len(tiles) == 8, f"{len(tiles)} tiles")
    check("every tile is exactly 640x640",
          all(t["x1"] - t["x0"] == 640 and t["y1"] - t["y0"] == 640 for t in tiles))

    # the cores must partition the frame with no gap and no overlap
    cover = np.zeros((H, W), np.int8)
    for t in tiles:
        x0, y0, x1, y1 = t["core"]
        cover[y0:y1, x0:x1] += 1
    check("cores tile the frame exactly once", cover.min() == 1 and cover.max() == 1,
          f"min {cover.min()} max {cover.max()}")

    # an object centred anywhere in a core must fit wholly inside that tile
    worst = 0
    for t in tiles:
        cx0, cy0, cx1, cy1 = t["core"]
        for cx, cy in [(cx0, cy0), (cx1 - 1, cy1 - 1), (cx0, cy1 - 1), (cx1 - 1, cy0)]:
            need = [cx - 80, cy - 80, cx + 80, cy + 80]
            inside = (need[0] >= t["x0"] - 1e-6 and need[1] >= t["y0"] - 1e-6 and
                      need[2] <= t["x1"] + 1e-6 and need[3] <= t["y1"] + 1e-6)
            edge = cx <= 80 or cy <= 80 or cx >= W - 80 or cy >= H - 80
            if not inside and not edge:
                worst += 1
    check("160px object in any core is never truncated", worst == 0, f"{worst} violations")

    try:
        make_tiles(W, H, 640, 100, 160)
        check("rejects overlap smaller than the largest object", False)
    except ValueError:
        check("rejects overlap smaller than the largest object", True)


def test_iou():
    print("\n[iou]")
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    m = iou_matrix(a, b)[0]
    check("identical boxes -> 1.0", abs(m[0] - 1.0) < 1e-5)
    check("half overlap -> 1/3", abs(m[1] - 1 / 3) < 1e-4, f"{m[1]:.4f}")
    check("disjoint -> 0", m[2] == 0)
    check("empty input is safe", iou_matrix(np.zeros((0, 4), np.float32), b).shape == (0, 3))


# ------------------------------------------------------- synthetic tracking
N_FRAMES = 280

# Each object may declare a `hide` window of frames during which the detector
# emits nothing for it, standing in for an occluder.
SCENE = [
    # A and B run head-on at nearly the same image row and pass through each
    # other around frame 114 - the classic ID-swap trap.
    dict(name="A", cls=3, x0=200,  y0=520, vx=6.0,  w=46, h=38, col=(40, 40, 220)),
    dict(name="B", cls=3, x0=1400, y0=530, vx=-4.5, w=46, h=38, col=(220, 180, 40)),
    # C vanishes for 21 frames: short enough that post-processing should fill
    # the hole by interpolation.
    dict(name="C", cls=3, x0=100,  y0=800, vx=5.0,  w=48, h=40, col=(60, 200, 60),
         hide=(100, 120)),
    # D is parked, E is a static false positive on a building.
    dict(name="D", cls=3, x0=1600, y0=200, vx=0.0,  w=44, h=36, col=(180, 180, 180)),
    dict(name="E", cls=3, x0=800,  y0=100, vx=0.0,  w=40, h=34, col=(120, 90, 70)),
    # F is hidden for 121 frames - longer than max_age (90), so it must be
    # recovered from the long-term gallery, ~600 px from where it vanished.
    dict(name="F", cls=3, x0=60,   y0=950, vx=5.0,  w=50, h=40, col=(30, 220, 220),
         hide=(80, 200)),
]


def synth_scene(n_frames=N_FRAMES):
    rng = np.random.default_rng(7)
    frames, dets, truth = [], [], []
    for f in range(n_frames):
        img = np.full((H, W, 3), 55, np.uint8)
        # textured background so camera-motion estimation has features to lock on
        img[::40, :] = 90
        img[:, ::40] = 90
        boxes, scores, clss, tags = [], [], [], []
        for o in SCENE:
            hide = o.get("hide")
            if hide and hide[0] <= f <= hide[1]:
                continue
            cx = o["x0"] + o["vx"] * f
            cy = o["y0"]
            x1, y1 = int(cx - o["w"] / 2), int(cy - o["h"] / 2)
            x2, y2 = int(cx + o["w"] / 2), int(cy + o["h"] / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), o["col"], -1)
            cv2.rectangle(img, (x1 + 6, y1 + 6), (x2 - 6, y1 + 14), (250, 250, 250), -1)
            j = rng.normal(0, 0.8, 4)
            boxes.append([x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3]])
            scores.append(0.90)
            clss.append(o["cls"])
            tags.append(o["name"])
        frames.append(img)
        dets.append((np.array(boxes, np.float32).reshape(-1, 4),
                     np.array(scores, np.float32), np.array(clss, np.int32)))
        truth.append(tags)
    return frames, dets, truth


def test_tracking():
    print("\n[tracking: crossing / occlusion / stationary]")
    cfg = Config()
    cfg.track.use_cmc = False          # synthetic camera is perfectly static
    frames, dets, truth = synth_scene()
    trk = Tracker(cfg.track, VISDRONE)

    rows, assoc = [], {}
    for f, (img, (b, s, c), tags) in enumerate(zip(frames, dets, truth)):
        out = trk.update(img, b, s, c)
        for tid, x1, y1, x2, y2, sc, cid, cname in out:
            rows.append((f, tid, x1, y1, x2, y2, sc, cid, cname))
            # attribute the emitted box back to whichever ground-truth object
            # it overlaps most
            if len(b):
                ious = iou_matrix(np.array([[x1, y1, x2, y2]], np.float32), b)[0]
                k = int(np.argmax(ious))
                if ious[k] > 0.3:
                    assoc.setdefault(tags[k], []).append(tid)

    names = [o["name"] for o in SCENE]
    for name in names:
        ids = assoc.get(name, [])
        uniq = len(set(ids))
        check(f"object {name}: one stable id across {len(ids)} frames",
              uniq == 1, f"{uniq} distinct id(s): {sorted(set(ids))[:5]}")

    tracks = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2",
                                         "y2", "conf", "cls_id", "cls_name"])
    check("C survives its 21-frame occlusion",
          len(assoc.get("C", [])) > 230, f"{len(assoc.get('C', []))} frames seen")
    check("F is recovered from the gallery after 121 hidden frames",
          len(assoc.get("F", [])) > 120, f"{len(assoc.get('F', []))} frames seen")

    # ---- post-processing
    from vtrack.postprocess import process
    cfg.post.min_track_len = 10
    cfg.post.min_track_hits = 6
    final, rep = process(tracks, FPS, cfg)
    print(f"    report: {rep['tracks_total']} tracks, "
          f"{rep['tracks_moving']} moving, {rep['tracks_parked']} parked")

    parked_names = set()
    for name in names:
        ids = set(assoc.get(name, []))
        sub = final[final["track_id"].isin(ids)]
        if len(sub) and bool(sub["parked"].iloc[0]):
            parked_names.add(name)
    check("stationary car D flagged parked", "D" in parked_names)
    check("static false positive E flagged parked", "E" in parked_names)
    check("moving cars A/B/C/F not flagged parked",
          not ({"A", "B", "C", "F"} & parked_names), f"parked={sorted(parked_names)}")
    n_interp = int(final["interpolated"].sum())
    check("short gap (21f) was interpolated", 15 <= n_interp <= 40, f"{n_interp} rows")
    check("moving tracks got compact display ids",
          final[~final["parked"]]["display_id"].notna().all())
    check("every row carries one of the 7 modes",
          set(final["mode"].unique()) <= set(MODES), set(final["mode"].unique()))
    check("speed is sane for a 6 px/frame car",
          2.0 < float(final[~final["parked"]]["speed_kmh"].median()) < 200.0,
          f"{float(final[~final['parked']]['speed_kmh'].median()):.1f} km/h")


def test_descriptor():
    print("\n[appearance descriptor]")
    img = np.zeros((200, 200, 3), np.uint8)
    cv2.rectangle(img, (10, 10), (60, 60), (30, 30, 220), -1)   # red
    cv2.rectangle(img, (110, 10), (160, 60), (220, 180, 30), -1)  # blue
    f = descriptors(img, np.array([[10, 10, 60, 60], [110, 10, 160, 60],
                                   [12, 12, 62, 62]], np.float32))
    check("descriptors are L2-normalised",
          np.allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-4))
    same = float(f[0] @ f[2])
    diff = float(f[0] @ f[1])
    check("same object scores far higher than a different-coloured one",
          same > 0.95 and diff < 0.75, f"same {same:.3f} vs diff {diff:.3f}")
    check("zero-area box does not crash",
          descriptors(img, np.array([[5, 5, 5, 5]], np.float32)).shape == (1, 124))


if __name__ == "__main__":
    test_taxonomy()
    test_scale()
    test_tiles()
    test_iou()
    test_descriptor()
    test_tracking()
    print("\n" + "=" * 58)
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("all checks passed")
