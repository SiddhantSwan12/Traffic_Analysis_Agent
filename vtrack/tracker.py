"""Multi-object tracker: Kalman + camera-motion compensation + two-stage
(ByteTrack-style) association fused with appearance, plus a long-term
re-identification gallery for occlusion recovery.

Design notes for the three hard requirements in the brief:

* through occlusion  - lost tracks keep being Kalman-predicted and stay eligible
                       for matching for max_age frames; beyond that they move to
                       a gallery matched on appearance plus a spatial gate.
* through crossing   - association fuses IoU with a colour/layout descriptor and
                       applies a soft penalty for cross-mode matches, so two
                       vehicles that overlap for a few frames do not swap.
* through long dwell - a stopped vehicle is still detected every frame, so it
                       stays matched by IoU; the Kalman velocity simply decays
                       to zero. No timeout applies while it remains visible.
"""
from __future__ import annotations
from collections import Counter
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanXYAH
from .cmc import CameraMotion
from .taxonomy import map_mode

TRACKED, LOST, NEW = 0, 1, 2
FEAT_DIM = 124


def descriptors(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Cheap 124-D appearance descriptor: 6x6 BGR layout grid plus a 16-bin
    saturation-weighted hue histogram, L2-normalised."""
    H, W = frame.shape[:2]
    out = np.zeros((len(boxes), FEAT_DIM), np.float32)
    for i, bx in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in bx)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(max(x1 + 2, x2), W), min(max(y1 + 2, y2), H)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        grid = cv2.resize(crop, (6, 6), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(cv2.resize(crop, (12, 12), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2HSV)
        hb = (hsv[..., 0].astype(np.int32) * 16) // 180
        sw = hsv[..., 1].astype(np.float32) / 255.0
        hist = np.bincount(hb.ravel(), weights=sw.ravel(), minlength=16).astype(np.float32)
        hist /= hist.sum() + 1e-6
        f = np.concatenate([grid.astype(np.float32).ravel() / 255.0, hist * 2.0])
        out[i] = f / (np.linalg.norm(f) + 1e-6)
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = (a[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i][None, :] for i in range(4))
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return (inter / np.maximum(ua, 1e-6)).astype(np.float32)


def _assign(cost: np.ndarray, max_cost: float):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    r, c = linear_sum_assignment(cost)
    matches = [(i, j) for i, j in zip(r, c) if cost[i, j] <= max_cost]
    mi = {i for i, _ in matches}
    mj = {j for _, j in matches}
    return (matches,
            [i for i in range(cost.shape[0]) if i not in mi],
            [j for j in range(cost.shape[1]) if j not in mj])


def _boxes_of(tracks):
    if not tracks:
        return np.zeros((0, 4), np.float32)
    return np.array([t.box for t in tracks], np.float32)


class Track:
    _next_id = 1

    def __init__(self, kf, box, score, cls_id, cls_name, feat, frame_id):
        self.kf = kf
        self.mean, self.cov = kf.initiate(self._to_xyah(box))
        self.id = 0
        self.state = NEW
        self.score = float(score)
        self.feat = feat.copy()
        self.hits = 1
        self.age = 0
        self.start_frame = frame_id
        self.last_frame = frame_id
        self.votes = Counter()
        self.votes[cls_name] += float(score)
        self.mode = map_mode(cls_name)
        self.cls_id = int(cls_id)

    @staticmethod
    def _to_xyah(b):
        w, h = b[2] - b[0], b[3] - b[1]
        return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], float)

    @property
    def box(self):
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)

    def predict(self):
        # Freeze box *shape* while unobserved but let position keep
        # extrapolating: a vehicle hidden behind a building for four seconds is
        # not where it vanished, and matching it against a stale box would fail
        # the spatial gate on re-emergence.
        if self.state != TRACKED:
            self.mean[6] = 0.0
            self.mean[7] = 0.0
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)

    def apply_cmc(self, M):
        R, t = M[:2, :2], M[:2, 2]
        s = float(np.sqrt(abs(np.linalg.det(R)))) or 1.0
        self.mean[0:2] = R @ self.mean[0:2] + t
        self.mean[4:6] = R @ self.mean[4:6]
        self.mean[3] *= s
        self.cov[0:2, 0:2] = R @ self.cov[0:2, 0:2] @ R.T
        self.cov[4:6, 4:6] = R @ self.cov[4:6, 4:6] @ R.T

    def update(self, box, score, cls_id, cls_name, feat, frame_id, alpha=0.90):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, self._to_xyah(box))
        self.score = float(score)
        self.hits += 1
        self.age = 0
        self.last_frame = frame_id
        self.cls_id = int(cls_id)
        self.votes[cls_name] += float(score)
        self.mode = map_mode(self.votes.most_common(1)[0][0])
        f = alpha * self.feat + (1 - alpha) * feat
        self.feat = f / (np.linalg.norm(f) + 1e-6)
        if self.state in (NEW, LOST):
            self.state = TRACKED

    def confirm(self):
        if self.id == 0:
            self.id = Track._next_id
            Track._next_id += 1


class Tracker:
    def __init__(self, cfg, class_names: dict, m_per_px_hint: float = 0.1):
        self.cfg = cfg
        self.names = class_names
        self.kf = KalmanXYAH()
        self.tracks: list = []
        self.gallery: list = []
        self.cmc = CameraMotion(cfg.cmc_downscale) if cfg.use_cmc else None
        self.m_per_px = m_per_px_hint
        self.frame_id = -1
        Track._next_id = 1

    def _fused_cost(self, tracks, boxes, feats, modes):
        if not len(tracks) or not len(boxes):
            return np.zeros((len(tracks), len(boxes)), np.float32)
        iou = iou_matrix(_boxes_of(tracks), boxes)
        tf = np.array([t.feat for t in tracks], np.float32)
        app = 1.0 - tf @ feats.T
        w = self.cfg.w_appearance
        cost = (1 - w) * (1 - iou) + w * app
        tm = np.array([t.mode or "" for t in tracks])[:, None]
        dm = np.array(modes)[None, :]
        cost = cost + 0.10 * (tm != dm)
        cost[iou < 0.05] = 1e5
        cost[app > self.cfg.app_gate] = 1e5
        return cost

    def update(self, frame, boxes, scores, cls_ids):
        c = self.cfg
        self.frame_id += 1
        fid = self.frame_id

        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        scores = np.asarray(scores, np.float32).reshape(-1)
        cls_ids = np.asarray(cls_ids, np.int32).reshape(-1)

        names = [self.names.get(int(k), str(k)) for k in cls_ids]
        modes = [map_mode(n) or "" for n in names]
        if len(names):
            keep = np.array([m != "" for m in modes], bool)
            boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
            names = [n for n, k in zip(names, keep) if k]
            modes = [m for m, k in zip(modes, keep) if k]
        feats = descriptors(frame, boxes) if len(boxes) else np.zeros((0, FEAT_DIM), np.float32)

        for t in self.tracks:
            t.predict()
        for t in self.gallery:
            t.predict()
        if self.cmc is not None:
            M = self.cmc.estimate(frame, boxes)
            if not np.allclose(M, np.eye(2, 3)):
                for t in self.tracks:
                    t.apply_cmc(M)
                for t in self.gallery:
                    t.apply_cmc(M)

        if len(scores):
            hi = scores >= c.high_thresh
            lo = (scores >= c.low_thresh) & (~hi)
        else:
            hi = lo = np.zeros(0, bool)
        idx_hi, idx_lo = np.flatnonzero(hi), np.flatnonzero(lo)

        active = [t for t in self.tracks if t.state != NEW]
        unconf = [t for t in self.tracks if t.state == NEW]

        # stage 1 - confirmed and recently-lost tracks vs high-confidence dets
        cost = self._fused_cost(active, boxes[idx_hi], feats[idx_hi],
                                [modes[i] for i in idx_hi])
        m1, ut1, ud1 = _assign(cost, c.match_iou)
        for ti, di in m1:
            d = idx_hi[di]
            active[ti].update(boxes[d], scores[d], cls_ids[d], names[d], feats[d], fid)

        # stage 2 - leftovers vs low-confidence dets (IoU only). This is what
        # keeps a partially-occluded vehicle alive: the detector still fires on
        # it, just weakly.
        rem = [active[i] for i in ut1 if active[i].state == TRACKED]
        cost2 = 1.0 - iou_matrix(_boxes_of(rem), boxes[idx_lo])
        m2, _, _ = _assign(cost2, c.low_match_iou)
        for ti, di in m2:
            d = idx_lo[di]
            rem[ti].update(boxes[d], scores[d], cls_ids[d], names[d], feats[d], fid)

        # stage 3 - unconfirmed tracks vs remaining high dets
        left_hi = [int(idx_hi[j]) for j in ud1]
        cost3 = 1.0 - iou_matrix(_boxes_of(unconf), boxes[left_hi])
        m3, _, ud3 = _assign(cost3, c.unconfirmed_iou)
        for ti, di in m3:
            d = left_hi[di]
            unconf[ti].update(boxes[d], scores[d], cls_ids[d], names[d], feats[d], fid)

        # stage 4 - long-term re-identification from the gallery
        strong = [d for d in (left_hi[j] for j in ud3) if scores[d] >= c.init_thresh]
        if self.gallery and strong:
            gb = _boxes_of(self.gallery)
            gf = np.array([t.feat for t in self.gallery], np.float32)
            db = boxes[strong]
            gc = np.stack([(gb[:, 0] + gb[:, 2]) / 2, (gb[:, 1] + gb[:, 3]) / 2], 1)
            dc = np.stack([(db[:, 0] + db[:, 2]) / 2, (db[:, 1] + db[:, 3]) / 2], 1)
            dist_m = np.linalg.norm(gc[:, None, :] - dc[None, :, :], axis=2) * self.m_per_px
            appd = 1.0 - gf @ feats[strong].T
            gm = np.array([t.mode or "" for t in self.gallery])[:, None]
            dmo = np.array([modes[d] for d in strong])[None, :]
            appd = appd + 0.15 * (gm != dmo)
            appd[dist_m > c.reid_radius_m] = 1e5
            m4, _, ud4 = _assign(appd, c.reid_dist)
            revived = set()
            for gi, di in m4:
                d = strong[di]
                t = self.gallery[gi]
                t.update(boxes[d], scores[d], cls_ids[d], names[d], feats[d], fid)
                self.tracks.append(t)
                revived.add(gi)
            self.gallery = [t for i, t in enumerate(self.gallery) if i not in revived]
            strong = [strong[j] for j in ud4]

        for d in strong:
            self.tracks.append(Track(self.kf, boxes[d], scores[d], cls_ids[d],
                                     names[d], feats[d], fid))

        alive = []
        for t in self.tracks:
            if t.last_frame == fid:
                if t.state == NEW and t.hits >= c.min_hits:
                    t.state = TRACKED
                if t.state == TRACKED:
                    t.confirm()
                alive.append(t)
                continue
            t.age += 1
            if t.state == NEW:
                continue
            t.state = LOST
            if t.age <= c.max_age:
                alive.append(t)
            elif t.id and t.hits >= c.min_hits:
                self.gallery.append(t)
        self.tracks = alive
        H, W = frame.shape[:2]
        m = 0.25 * max(W, H)
        self.gallery = [
            t for t in self.gallery
            if fid - t.last_frame <= c.reid_max_age
            and -m <= t.mean[0] <= W + m and -m <= t.mean[1] <= H + m
        ]

        out = []
        for t in self.tracks:
            if t.last_frame == fid and t.id and t.state == TRACKED:
                x1, y1, x2, y2 = t.box
                out.append((t.id, float(x1), float(y1), float(x2), float(y2),
                            float(t.score), int(t.cls_id),
                            self.names.get(int(t.cls_id), "")))
        return out
