"""Global camera-motion compensation.

Even in a stationary hover the gimbal micro-corrects, which drags every box a
few pixels per frame. Left uncompensated this is the single biggest source of
ID switches in dense top-down scenes: the Kalman filter absorbs the drift as
object velocity and predicts every box off-target simultaneously.
"""
from __future__ import annotations
import cv2
import numpy as np


class CameraMotion:
    def __init__(self, downscale: int = 4, max_corners: int = 800):
        self.ds = max(1, int(downscale))
        self.max_corners = max_corners
        self._prev = None
        # Quality of the most recent estimate, in 0..1. A low value means the
        # affine fit could not be trusted and identity was substituted, so any
        # apparent object motion that frame may really be camera motion.
        self.quality = 0.0
        self.inliers = 0

    def estimate(self, frame_bgr: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        """Return a 2x3 affine mapping previous-frame coords to current, in
        full-resolution pixels. Identity when it cannot be estimated."""
        eye = np.eye(2, 3, dtype=np.float64)
        self.quality = 0.0
        self.inliers = 0
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.ds > 1:
            g = cv2.resize(g, (g.shape[1] // self.ds, g.shape[0] // self.ds),
                           interpolation=cv2.INTER_AREA)

        prev = self._prev
        self._prev = (g, det_boxes)
        if prev is None:
            return eye
        pg, pboxes = prev

        # Track only static background: mask out anything the detector found.
        mask = np.full(pg.shape, 255, np.uint8)
        if len(pboxes):
            b = (np.asarray(pboxes, np.float32) / self.ds).astype(np.int32)
            for x1, y1, x2, y2 in b:
                cv2.rectangle(mask, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), 0, -1)

        p0 = cv2.goodFeaturesToTrack(pg, self.max_corners, 0.01, 8, mask=mask,
                                     blockSize=3)
        if p0 is None or len(p0) < 12:
            return eye
        p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, p0, None,
                                             winSize=(15, 15), maxLevel=3)
        if p1 is None:
            return eye
        st = st.ravel().astype(bool)
        a, b = p0[st].reshape(-1, 2), p1[st].reshape(-1, 2)
        if len(a) < 12:
            return eye
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0,
                                             maxIters=800)
        if M is None or inl is None or int(inl.sum()) < 10:
            return eye
        n_in = int(inl.sum())
        ratio = n_in / max(len(a), 1)
        # both a healthy inlier count and a healthy inlier ratio are needed:
        # a handful of agreeing points is not a reliable global estimate
        self.inliers = n_in
        self.quality = float(np.clip(min(n_in / 120.0, 1.0) * min(ratio / 0.6, 1.0),
                                     0.0, 1.0))
        M = M.astype(np.float64)
        M[:, 2] *= self.ds          # translation back to full resolution
        return M
