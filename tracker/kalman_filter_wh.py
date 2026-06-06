# kalman_filter_wh.py
# -*- coding: utf-8 -*-
"""
Width-Height Kalman Filter for EelTrack-Edge.

This module implements the Kalman filter used in the improved ByteTrack tracker
for rice field eel head tracking.

State vector:
    x = [cx, cy, w, h, vcx, vcy, vw, vh]^T

Measurement vector:
    z = [cx, cy, w, h]^T

Compared with the original ByteTrack / DeepSORT XYAH formulation:
    [cx, cy, a, h], where a = w / h

this version explicitly models width and height:
    [cx, cy, w, h]

This is better suited to rice field eel head tracking because eel-head boxes may
change independently in width and height under deformation, occlusion, and scale
fluctuation.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [x, y, w, h] to [cx, cy, w, h].

    Args:
        tlwh: ndarray with shape (..., 4)

    Returns:
        ndarray with shape (..., 4)
    """
    ret = np.asarray(tlwh, dtype=np.float32).copy()
    ret[..., 0] = ret[..., 0] + ret[..., 2] / 2.0
    ret[..., 1] = ret[..., 1] + ret[..., 3] / 2.0
    return ret


def xywh_to_tlwh(xywh: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [cx, cy, w, h] to [x, y, w, h].

    Args:
        xywh: ndarray with shape (..., 4)

    Returns:
        ndarray with shape (..., 4)
    """
    ret = np.asarray(xywh, dtype=np.float32).copy()
    ret[..., 0] = ret[..., 0] - ret[..., 2] / 2.0
    ret[..., 1] = ret[..., 1] - ret[..., 3] / 2.0
    return ret


def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h].

    Args:
        tlbr: ndarray with shape (..., 4)

    Returns:
        ndarray with shape (..., 4)
    """
    ret = np.asarray(tlbr, dtype=np.float32).copy()
    ret[..., 2] = ret[..., 2] - ret[..., 0]
    ret[..., 3] = ret[..., 3] - ret[..., 1]
    return ret


def tlbr_to_xywh(tlbr: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [x1, y1, x2, y2] to [cx, cy, w, h].

    Args:
        tlbr: ndarray with shape (..., 4)

    Returns:
        ndarray with shape (..., 4)
    """
    return tlwh_to_xywh(tlbr_to_tlwh(tlbr))


def xywh_to_tlbr(xywh: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [cx, cy, w, h] to [x1, y1, x2, y2].

    Args:
        xywh: ndarray with shape (..., 4)

    Returns:
        ndarray with shape (..., 4)
    """
    tlwh = xywh_to_tlwh(xywh)
    ret = tlwh.copy()
    ret[..., 2] = ret[..., 0] + ret[..., 2]
    ret[..., 3] = ret[..., 1] + ret[..., 3]
    return ret


class KalmanFilterWH:
    """
    Kalman filter with explicit width-height state representation.

    The model assumes constant velocity:

        cx_t = cx_{t-1} + vcx_{t-1}
        cy_t = cy_{t-1} + vcy_{t-1}
        w_t  = w_{t-1}  + vw_{t-1}
        h_t  = h_{t-1}  + vh_{t-1}

    State:
        mean = [cx, cy, w, h, vcx, vcy, vw, vh]

    Measurement:
        measurement = [cx, cy, w, h]
    """

    ndim: int = 4

    def __init__(
        self,
        dt: float = 1.0,
        std_weight_position: float = 1.0 / 20.0,
        std_weight_velocity: float = 1.0 / 160.0,
        min_size: float = 1.0,
    ) -> None:
        """
        Args:
            dt:
                Time step between adjacent frames.
                For frame-by-frame tracking, use 1.0.
            std_weight_position:
                Relative standard deviation for position-related process noise.
            std_weight_velocity:
                Relative standard deviation for velocity-related process noise.
            min_size:
                Minimum allowed width and height to avoid numerical instability.
        """
        self.dt = float(dt)
        self.std_weight_position = float(std_weight_position)
        self.std_weight_velocity = float(std_weight_velocity)
        self.min_size = float(min_size)

        # State transition matrix F.
        # x_t = F x_{t-1}
        self.motion_mat = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self.motion_mat[i, self.ndim + i] = self.dt

        # Observation matrix H.
        # z_t = H x_t
        self.update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)

    def _scale(self, xywh: np.ndarray) -> float:
        """
        Estimate scale for adaptive process and observation noise.

        The original ByteTrack commonly uses h as the scale reference.
        Here, max(w, h) is used because width and height are independently modeled.
        """
        w = float(xywh[2])
        h = float(xywh[3])
        return max(self.min_size, w, h)

    def _clip_size(self, mean: np.ndarray) -> np.ndarray:
        """
        Ensure predicted width and height are valid.
        """
        mean = mean.copy()
        mean[2] = max(self.min_size, float(mean[2]))
        mean[3] = max(self.min_size, float(mean[3]))
        return mean

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a new Kalman track from the first detection.

        Args:
            measurement:
                ndarray with shape (4,), format [cx, cy, w, h]

        Returns:
            mean:
                ndarray with shape (8,)
            covariance:
                ndarray with shape (8, 8)
        """
        measurement = np.asarray(measurement, dtype=np.float32).reshape(4)
        measurement[2] = max(self.min_size, float(measurement[2]))
        measurement[3] = max(self.min_size, float(measurement[3]))

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos, dtype=np.float32)
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)

        scale = self._scale(measurement)

        # Larger initial uncertainty for velocity because the first detection
        # contains no motion information.
        std = np.array(
            [
                2.0 * self.std_weight_position * scale,
                2.0 * self.std_weight_position * scale,
                2.0 * self.std_weight_position * scale,
                2.0 * self.std_weight_position * scale,
                10.0 * self.std_weight_velocity * scale,
                10.0 * self.std_weight_velocity * scale,
                10.0 * self.std_weight_velocity * scale,
                10.0 * self.std_weight_velocity * scale,
            ],
            dtype=np.float32,
        )

        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state.

        Args:
            mean:
                ndarray with shape (8,)
            covariance:
                ndarray with shape (8, 8)

        Returns:
            predicted_mean, predicted_covariance
        """
        mean = np.asarray(mean, dtype=np.float32).reshape(8)
        covariance = np.asarray(covariance, dtype=np.float32).reshape(8, 8)

        scale = self._scale(mean[:4])

        std_pos = self.std_weight_position * scale
        std_vel = self.std_weight_velocity * scale

        motion_std = np.array(
            [
                std_pos,
                std_pos,
                std_pos,
                std_pos,
                std_vel,
                std_vel,
                std_vel,
                std_vel,
            ],
            dtype=np.float32,
        )

        motion_cov = np.diag(np.square(motion_std)).astype(np.float32)

        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov

        mean = self._clip_size(mean)

        return mean.astype(np.float32), covariance.astype(np.float32)

    def multi_predict(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict multiple tracks.

        Args:
            means:
                ndarray with shape (N, 8)
            covariances:
                ndarray with shape (N, 8, 8)

        Returns:
            predicted_means:
                ndarray with shape (N, 8)
            predicted_covariances:
                ndarray with shape (N, 8, 8)
        """
        means = np.asarray(means, dtype=np.float32)
        covariances = np.asarray(covariances, dtype=np.float32)

        if len(means) == 0:
            return means, covariances

        predicted_means = []
        predicted_covariances = []

        for mean, covariance in zip(means, covariances):
            mean, covariance = self.predict(mean, covariance)
            predicted_means.append(mean)
            predicted_covariances.append(covariance)

        return (
            np.asarray(predicted_means, dtype=np.float32),
            np.asarray(predicted_covariances, dtype=np.float32),
        )

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project the state distribution into measurement space.

        Args:
            mean:
                ndarray with shape (8,)
            covariance:
                ndarray with shape (8, 8)

        Returns:
            projected_mean:
                ndarray with shape (4,)
            projected_covariance:
                ndarray with shape (4, 4)
        """
        mean = np.asarray(mean, dtype=np.float32).reshape(8)
        covariance = np.asarray(covariance, dtype=np.float32).reshape(8, 8)

        scale = self._scale(mean[:4])

        observation_std = np.array(
            [
                self.std_weight_position * scale,
                self.std_weight_position * scale,
                self.std_weight_position * scale,
                self.std_weight_position * scale,
            ],
            dtype=np.float32,
        )

        innovation_cov = np.diag(np.square(observation_std)).astype(np.float32)

        projected_mean = self.update_mat @ mean
        projected_covariance = (
            self.update_mat @ covariance @ self.update_mat.T + innovation_cov
        )

        return projected_mean.astype(np.float32), projected_covariance.astype(np.float32)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct the predicted state using a matched detection.

        Args:
            mean:
                predicted state mean, shape (8,)
            covariance:
                predicted state covariance, shape (8, 8)
            measurement:
                detection measurement, shape (4,), format [cx, cy, w, h]

        Returns:
            updated_mean, updated_covariance
        """
        mean = np.asarray(mean, dtype=np.float32).reshape(8)
        covariance = np.asarray(covariance, dtype=np.float32).reshape(8, 8)
        measurement = np.asarray(measurement, dtype=np.float32).reshape(4)

        measurement[2] = max(self.min_size, float(measurement[2]))
        measurement[3] = max(self.min_size, float(measurement[3]))

        projected_mean, projected_covariance = self.project(mean, covariance)

        # Kalman gain:
        # K = P H^T (H P H^T + R)^-1
        kalman_gain = (
            covariance
            @ self.update_mat.T
            @ np.linalg.inv(projected_covariance)
        )

        innovation = measurement - projected_mean

        updated_mean = mean + kalman_gain @ innovation
        updated_covariance = covariance - (
            kalman_gain @ projected_covariance @ kalman_gain.T
        )

        updated_mean = self._clip_size(updated_mean)

        # Symmetrize covariance to reduce floating-point asymmetry.
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)

        return updated_mean.astype(np.float32), updated_covariance.astype(np.float32)

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """
        Compute gating distance between predicted state and candidate measurements.

        This function is useful if you want to add Mahalanobis gating before IoU
        matching. In your article, the main task-specific gating is the downstream
        motion-corridor constraint, so this function is optional.

        Args:
            mean:
                ndarray with shape (8,)
            covariance:
                ndarray with shape (8, 8)
            measurements:
                ndarray with shape (N, 4), format [cx, cy, w, h]
            only_position:
                If True, use only [cx, cy].
            metric:
                "maha" for squared Mahalanobis distance.
                "gaussian" for squared Euclidean distance.

        Returns:
            distances:
                ndarray with shape (N,)
        """
        measurements = np.asarray(measurements, dtype=np.float32)

        if measurements.ndim == 1:
            measurements = measurements.reshape(1, 4)

        projected_mean, projected_covariance = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_covariance = projected_covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - projected_mean

        if metric == "gaussian":
            return np.sum(d * d, axis=1).astype(np.float32)

        if metric != "maha":
            raise ValueError("metric must be either 'maha' or 'gaussian'.")

        # Squared Mahalanobis distance:
        # d^T S^-1 d
        inv_cov = np.linalg.inv(projected_covariance)
        distances = np.einsum("ij,jk,ik->i", d, inv_cov, d)

        return distances.astype(np.float32)


# Alias used by byte_tracker_eel.py
EelKalmanFilterWH = KalmanFilterWH


if __name__ == "__main__":
    # Minimal sanity check.
    kf = KalmanFilterWH()

    # First detection: [cx, cy, w, h]
    measurement_1 = np.array([120.0, 80.0, 28.0, 22.0], dtype=np.float32)

    mean, covariance = kf.initiate(measurement_1)
    print("Initial mean:")
    print(mean)

    mean, covariance = kf.predict(mean, covariance)
    print("\nPredicted mean:")
    print(mean)

    # Matched detection in next frame.
    measurement_2 = np.array([122.0, 91.0, 29.0, 21.0], dtype=np.float32)

    mean, covariance = kf.update(mean, covariance, measurement_2)
    print("\nUpdated mean:")
    print(mean)

    print("\nPredicted tlbr box:")
    print(xywh_to_tlbr(mean[:4]))