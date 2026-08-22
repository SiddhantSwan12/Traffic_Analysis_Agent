"""Annotated video rendering: boxes, persistent IDs, motion trails, per-mode
HUD. Frames are piped straight into ffmpeg, so no multi-gigabyte intermediate
file is ever written to disk.
"""
from __future__ import annotations
import shutil
import subprocess
from collections import defaultdict, deque

import cv2
import numpy as np

from .taxonomy import MODE_COLOR, MODES

FONT = cv2.FONT_HERSHEY_SIMPLEX


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        raise RuntimeError("ffmpeg not found - run: pip install imageio-ffmpeg")


def pick_encoder(requested: str, exe: str) -> str:
    if requested != "auto":
        return requested
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30).stdout
        if "h264_nvenc" in out:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


class VideoPipe:
    """Raw BGR frames in, H.264 mp4 out."""

    def __init__(self, path, w, h, fps, crf=20, encoder="auto"):
        exe = ffmpeg_exe()
        enc = pick_encoder(encoder, exe)
        quality = (["-rc", "vbr", "-cq", str(crf), "-preset", "p5"]
                   if enc == "h264_nvenc" else
                   ["-crf", str(crf), "-preset", "medium"])
        cmd = [exe, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
               "-an", "-c:v", enc, *quality,
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self.encoder = enc
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        try:
            self.proc.stdin.close()
        finally:
            self.proc.wait()


def id_color(i: int):
    """Distinct, stable colour per identity (golden-angle hue rotation)."""
    hue = int((i * 137.508) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# Short forms keep dense scenes readable; the box's coloured cap and the HUD
# legend carry the full class name.
SHORT = {"pedestrian": "ped", "motorcycle": "m/c", "car": "car",
         "LGV": "LGV", "truck": "trk", "HGV": "HGV", "bus": "bus"}


def _text_for(row, disp_id, mode, r, box_px):
    if r.label_mode == "id" or box_px < r.label_min_box_px:
        return str(disp_id)
    name = mode if r.label_mode == "full" else SHORT[mode]
    txt = f"{disp_id} {name}"
    if r.show_speed and row.speed_kmh >= 5.0:
        txt += f" {row.speed_kmh:.0f}"
    return txt


GRID = 8  # occupancy-grid cell size, in pixels


def _try_place(occ, x, y, w, h, W, H):
    """Reserve a w x h label box whose bottom-left corner is (x, y).
    Returns True if the space was free (and is now claimed)."""
    x0, y0 = max(0, x), max(0, y - h)
    x1, y1 = min(W, x + w), min(H, y)
    if x1 <= x0 or y1 <= y0:
        return False
    gx0, gy0, gx1, gy1 = x0 // GRID, y0 // GRID, -(-x1 // GRID), -(-y1 // GRID)
    if occ[gy0:gy1, gx0:gx1].any():
        return False
    occ[gy0:gy1, gx0:gx1] = True
    return True


def _label(img, occ, box, text, short, color, scale, pad=3):
    """Draw a label near `box`, avoiding labels already placed this frame.

    Tries above the box, then below, then a shortened form, then gives up on
    text entirely - the box outline and its mode-coloured cap still identify the
    object, so a dense cluster degrades gracefully instead of turning into an
    unreadable pile of overlapping text.
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    for txt in ([text, short] if short != text else [text]):
        (tw, th), _ = cv2.getTextSize(txt, FONT, scale, 1)
        bw, bh = tw + pad * 2, th + pad * 2
        for ax, ay in ((x1, y1 - 2), (x1, y2 + bh + 2), (x2 + 2, y1 + bh)):
            if _try_place(occ, ax, ay, bw, bh, W, H):
                cv2.rectangle(img, (ax, ay - bh), (ax + bw, ay), color, -1)
                cv2.putText(img, txt, (ax + pad, ay - pad - 1), FONT, scale,
                            (0, 0, 0), 1, cv2.LINE_AA)
                return


def _hud(img, frame_idx, fps, live_counts, seen_counts, n_parked, show_parked):
    h, w = img.shape[:2]
    lines = [m for m in MODES if seen_counts.get(m)]
    # 34 header + one row per mode + 30 for the footer note, which must sit
    # inside the panel rather than spilling onto the frame below it
    panel_h = 34 + 18 * max(len(lines), 1) + 30
    panel_w = 250
    ov = img[8:8 + panel_h, 8:8 + panel_w]
    cv2.rectangle(img, (8, 8), (8 + panel_w, 8 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.25, img[8:8 + panel_h, 8:8 + panel_w], 0.75, 0,
                    img[8:8 + panel_h, 8:8 + panel_w])
    cv2.rectangle(img, (8, 8), (8 + panel_w, 8 + panel_h), (90, 90, 90), 1)

    t = frame_idx / fps
    cv2.putText(img, f"t {int(t)//60:02d}:{t%60:05.2f}   f{frame_idx:06d}",
                (18, 30), FONT, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    y = 52
    cv2.putText(img, "MODE        NOW  TOTAL", (18, y), FONT, 0.38,
                (150, 150, 150), 1, cv2.LINE_AA)
    y += 16
    for m in lines:
        cv2.rectangle(img, (18, y - 8), (28, y + 2), MODE_COLOR[m], -1)
        cv2.putText(img, m, (34, y), FONT, 0.42, (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(img, f"{live_counts.get(m, 0):>4d}", (140, y), FONT, 0.42,
                    (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(img, f"{seen_counts.get(m, 0):>5d}", (180, y), FONT, 0.42,
                    (170, 200, 255), 1, cv2.LINE_AA)
        y += 18
    note = f"parked/static hidden: {n_parked}" if not show_parked else \
           f"parked/static shown: {n_parked}"
    cv2.putText(img, note, (18, y + 6), FONT, 0.36, (140, 140, 140), 1, cv2.LINE_AA)


def render(video_path, df, out_path, fps, cfg, max_frames=None, progress=None):
    r = cfg.render
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    use = df if r.show_parked else df[~df["parked"]]
    use = use[use["display_id"].notna()]
    by_frame = {int(f): g for f, g in use.groupby("frame")}
    n_parked_tracks = int(df[df["parked"]]["track_id"].nunique())

    trails = defaultdict(lambda: deque(maxlen=r.trail_frames))
    last_seen = {}
    seen_counts = defaultdict(int)
    seen_ids = set()

    pipe = VideoPipe(out_path, W, H, fps, r.crf, r.encoder)
    idx = 0
    try:
        while idx < total:
            ok, frame = cap.read()
            if not ok:
                break
            g = by_frame.get(idx)
            live = defaultdict(int)

            if g is not None:
                did = g["display_id"].to_numpy()
                cxs = g["cx"].to_numpy()
                cys = g["cy"].to_numpy()
                for d, cx, cy in zip(did, cxs, cys):
                    trails[int(d)].append((float(cx), float(cy)))
                    last_seen[int(d)] = idx

            # trails first so boxes draw on top
            for d, pts in list(trails.items()):
                if idx - last_seen.get(d, -10**9) > r.trail_frames:
                    del trails[d]
                    continue
                if len(pts) < 3:
                    continue
                col = id_color(d)
                arr = np.asarray(pts, np.int32)
                n = len(arr)
                for i in range(1, n, 2):
                    a = i / n
                    c = (int(col[0] * a), int(col[1] * a), int(col[2] * a))
                    th = max(1, int(round(r.trail_thickness * a)))
                    cv2.line(frame, tuple(arr[i - 1]), tuple(arr[min(i + 1, n - 1)]),
                             c, th, cv2.LINE_AA)

            if g is not None:
                occ = np.zeros((-(-H // GRID), -(-W // GRID)), bool)
                if r.show_hud:   # the HUD is painted last; do not waste labels under it
                    occ[: 230 // GRID, : 250 // GRID] = True
                # bigger objects get first claim on label space
                order = g.assign(_a=(g["x2"] - g["x1"]) * (g["y2"] - g["y1"])) \
                         .sort_values("_a", ascending=False)
                for row in order.itertuples(index=False):
                    d = int(row.display_id)
                    mode = row.mode
                    live[mode] += 1
                    if d not in seen_ids:
                        seen_ids.add(d)
                        seen_counts[mode] += 1
                    x1, y1 = int(row.x1), int(row.y1)
                    x2, y2 = int(row.x2), int(row.y2)
                    col = id_color(d)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, r.box_thickness)
                    # thin mode-coloured cap so class reads at a glance even
                    # when the text label has been shortened away
                    cv2.rectangle(frame, (x1, y1), (x2, y1 + 3), MODE_COLOR[mode], -1)
                    _label(frame, occ, (x1, y1, x2, y2),
                           _text_for(row, d, mode, r, max(x2 - x1, y2 - y1)),
                           str(d), col, r.font_scale)

            if r.show_hud:
                _hud(frame, idx, fps, live, seen_counts, n_parked_tracks,
                     r.show_parked)

            pipe.write(frame)
            idx += 1
            if progress and idx % 200 == 0:
                progress(idx, total)
    finally:
        cap.release()
        pipe.close()
    return {"frames_written": idx, "encoder": pipe.encoder,
            "unique_ids_shown": len(seen_ids)}
