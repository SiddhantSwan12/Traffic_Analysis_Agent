"""8-state constant-velocity Kalman filter on (cx, cy, aspect, height)."""
from __future__ import annotations
import numpy as np
import scipy.linalg


class KalmanXYAH:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)
        self._sp = 1.0 / 20   # position noise, relative to box height
        self._sv = 1.0 / 160  # velocity noise

    def initiate(self, m):
        mean = np.r_[m, np.zeros(4)]
        h = m[3]
        std = [2 * self._sp * h, 2 * self._sp * h, 1e-2, 2 * self._sp * h,
               10 * self._sv * h, 10 * self._sv * h, 1e-5, 10 * self._sv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self._sp * h, self._sp * h, 1e-2, self._sp * h,
               self._sv * h, self._sv * h, 1e-5, self._sv * h]
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + np.diag(np.square(std))
        return mean, cov

    def project(self, mean, cov):
        h = mean[3]
        std = [self._sp * h, self._sp * h, 1e-1, self._sp * h]
        return self._H @ mean, self._H @ cov @ self._H.T + np.diag(np.square(std))

    def update(self, mean, cov, meas):
        pmean, pcov = self.project(mean, cov)
        chol, lower = scipy.linalg.cho_factor(pcov, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve((chol, lower), (cov @ self._H.T).T,
                                      check_finite=False).T
        innov = meas - pmean
        return mean + innov @ gain.T, cov - gain @ pcov @ gain.T
