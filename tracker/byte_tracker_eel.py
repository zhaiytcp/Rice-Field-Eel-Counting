# byte_tracker_eel.py
# -*- coding: utf-8 -*-
"""
Improved ByteTrack for rice field eel head tracking.

This implementation is designed for the EelTrack-Edge pipeline:
1. YOLO11s-Eel detects eel-head bounding boxes frame by frame.
2. BYTETrackerEel associates detections across frames.
3. A downstream counting module can use the returned TrackIDs.

Main modifications compared with the original ByteTrack:
- Kalman state uses explicit width and height:
    x = [cx, cy, w, h, vcx, vcy, vw, vh]^T
- Data association uses a flow-aligned motion-corridor constraint:
    0 <= s_ij <= L_i, |r_ij| <= W_i
- Two-stage ByteTrack association is retained:
    high-confidence detections first, low-confidence detections second.

Expected detection format:
    np.ndarray with shape [N, 5] or [N, 6]

    [x1, y1, x2, y2, score]
    or
    [x1, y1, x2, y2, score, cls]

All coordinates should be absolute pixel coordinates after resizing, e.g. 640 x 640.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


# ----------------------------------------------------------------------
# Basic geometry utilities
# ----------------------------------------------------------------------

def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    ret = np.asarray(tlbr, dtype=np.float32).copy()
    ret[2] = ret[2] - ret[0]
    ret[3] = ret[3] - ret[1]
    return ret


def tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    ret = np.asarray(tlwh, dtype=np.float32).copy()
    ret[2] = ret[0] + ret[2]
    ret[3] = ret[1] + ret[3]
    return ret


def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] to [cx, cy, w, h]."""
    ret = np.asarray(tlwh, dtype=np.float32).copy()
    ret[0] = ret[0] + ret[2] / 2.0
    ret[1] = ret[1] + ret[3] / 2.0
    return ret


def xywh_to_tlwh(xywh: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x, y, w, h]."""
    ret = np.asarray(xywh, dtype=np.float32).copy()
    ret[0] = ret[0] - ret[2] / 2.0
    ret[1] = ret[1] - ret[3] / 2.0
    return ret


def bbox_ious(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU between two sets of boxes.

    Args:
        boxes_a: [N, 4], tlbr format.
        boxes_b: [M, 4], tlbr format.

    Returns:
        IoU matrix with shape [N, M].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = boxes_a.astype(np.float32)
    boxes_b = boxes_b.astype(np.float32)

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(
        0.0, boxes_a[:, 3] - boxes_a[:, 1]
    )
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(
        0.0, boxes_b[:, 3] - boxes_b[:, 1]
    )

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-12)


def iou_distance(tracks: Sequence["STrack"], detections: Sequence["STrack"]) -> np.ndarray:
    """Cost matrix = 1 - IoU."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    track_boxes = np.asarray([t.tlbr for t in tracks], dtype=np.float32)
    det_boxes = np.asarray([d.tlbr for d in detections], dtype=np.float32)
    return 1.0 - bbox_ious(track_boxes, det_boxes)


def fuse_score(cost_matrix: np.ndarray, detections: Sequence["STrack"]) -> np.ndarray:
    """
    Optional ByteTrack-style score fusion.

    Higher detection confidence decreases the matching cost.
    """
    if cost_matrix.size == 0:
        return cost_matrix

    scores = np.asarray([d.score for d in detections], dtype=np.float32)
    iou_sim = 1.0 - cost_matrix
    fuse_sim = iou_sim * scores[None, :]
    return 1.0 - fuse_sim


def linear_assignment_with_threshold(
    cost_matrix: np.ndarray,
    thresh: float,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Hungarian matching with a maximum allowed cost.

    Returns:
        matches: ndarray of shape [K, 2], each row is [track_index, detection_index]
        unmatched_rows: unmatched track indices
        unmatched_cols: unmatched detection indices
    """
    num_rows, num_cols = cost_matrix.shape

    if num_rows == 0 or num_cols == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            list(range(num_rows)),
            list(range(num_cols)),
        )

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_rows = set(range(num_rows))
        unmatched_cols = set(range(num_cols))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append([r, c])
                unmatched_rows.discard(r)
                unmatched_cols.discard(c)

        return (
            np.asarray(matches, dtype=np.int32),
            sorted(unmatched_rows),
            sorted(unmatched_cols),
        )

    # Fallback greedy matching if scipy is not installed.
    pairs = []
    for r in range(num_rows):
        for c in range(num_cols):
            pairs.append((cost_matrix[r, c], r, c))
    pairs.sort(key=lambda x: x[0])

    matched_rows = set()
    matched_cols = set()
    matches = []

    for cost, r, c in pairs:
        if cost > thresh:
            break
        if r in matched_rows or c in matched_cols:
            continue
        matches.append([r, c])
        matched_rows.add(r)
        matched_cols.add(c)

    unmatched_rows = [r for r in range(num_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(num_cols) if c not in matched_cols]

    return (
        np.asarray(matches, dtype=np.int32),
        unmatched_rows,
        unmatched_cols,
    )


# ----------------------------------------------------------------------
# Kalman filter with explicit width-height state
# ----------------------------------------------------------------------

class EelKalmanFilterWH:
    """
    Kalman filter using the state:
        [cx, cy, w, h, vcx, vcy, vw, vh]

    Measurement:
        [cx, cy, w, h]
    """

    ndim = 4
    dt = 1.0

    def __init__(
        self,
        std_weight_position: float = 1.0 / 20,
        std_weight_velocity: float = 1.0 / 160,
    ) -> None:
        self.std_weight_position = std_weight_position
        self.std_weight_velocity = std_weight_velocity

        self.motion_mat = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self.motion_mat[i, self.ndim + i] = self.dt

        self.update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)

    @staticmethod
    def _scale_from_measurement(measurement: np.ndarray) -> float:
        return float(max(1.0, measurement[2], measurement[3]))

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from an initial measurement.

        Args:
            measurement: [cx, cy, w, h]
        """
        measurement = np.asarray(measurement, dtype=np.float32)
        mean = np.r_[measurement, np.zeros_like(measurement)]

        scale = self._scale_from_measurement(measurement)
        std = np.array(
            [
                2 * self.std_weight_position * scale,
                2 * self.std_weight_position * scale,
                2 * self.std_weight_position * scale,
                2 * self.std_weight_position * scale,
                10 * self.std_weight_velocity * scale,
                10 * self.std_weight_velocity * scale,
                10 * self.std_weight_velocity * scale,
                10 * self.std_weight_velocity * scale,
            ],
            dtype=np.float32,
        )
        covariance = np.diag(np.square(std))
        return mean.astype(np.float32), covariance.astype(np.float32)

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman prediction."""
        scale = float(max(1.0, mean[2], mean[3]))

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
        motion_cov = np.diag(np.square(motion_std))

        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov

        mean[2] = max(1.0, mean[2])
        mean[3] = max(1.0, mean[3])

        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space."""
        scale = float(max(1.0, mean[2], mean[3]))
        std = np.array(
            [
                self.std_weight_position * scale,
                self.std_weight_position * scale,
                self.std_weight_position * scale,
                self.std_weight_position * scale,
            ],
            dtype=np.float32,
        )
        innovation_cov = np.diag(np.square(std))

        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T
        projected_cov = projected_cov + innovation_cov

        return projected_mean.astype(np.float32), projected_cov.astype(np.float32)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman correction."""
        measurement = np.asarray(measurement, dtype=np.float32)

        projected_mean, projected_cov = self.project(mean, covariance)
        kalman_gain = covariance @ self.update_mat.T @ np.linalg.inv(projected_cov)

        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        new_mean[2] = max(1.0, new_mean[2])
        new_mean[3] = max(1.0, new_mean[3])

        return new_mean.astype(np.float32), new_covariance.astype(np.float32)


# ----------------------------------------------------------------------
# Track object
# ----------------------------------------------------------------------

class TrackState(Enum):
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    track_id: int = 0
    state: TrackState = TrackState.Tracked
    is_activated: bool = False
    frame_id: int = 0
    start_frame: int = 0

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_id(cls) -> None:
        cls._count = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


class STrack(BaseTrack):
    """
    Single eel-head track.
    """

    shared_kalman = EelKalmanFilterWH()

    def __init__(
        self,
        tlwh: Union[np.ndarray, Sequence[float]],
        score: float,
        cls: float = -1,
    ) -> None:
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.cls = int(cls) if cls is not None else -1

        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.tracklet_len = 0
        self.is_activated = False
        self.state = TrackState.Tracked

        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0

        self.lateral_offsets: List[float] = []
        self.center_history: List[np.ndarray] = []

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()

        xywh = self.mean[:4].copy()
        return xywh_to_tlwh(xywh)

    @property
    def tlbr(self) -> np.ndarray:
        return tlwh_to_tlbr(self.tlwh)

    @property
    def xywh(self) -> np.ndarray:
        if self.mean is None:
            return tlwh_to_xywh(self._tlwh)
        return self.mean[:4].copy()

    @property
    def center(self) -> np.ndarray:
        return self.xywh[:2].copy()

    @property
    def lateral_sigma(self) -> float:
        if len(self.lateral_offsets) < 2:
            return 0.0
        return float(np.std(self.lateral_offsets))

    @staticmethod
    def from_tlbr(
        tlbr: Union[np.ndarray, Sequence[float]],
        score: float,
        cls: float = -1,
    ) -> "STrack":
        return STrack(tlbr_to_tlwh(np.asarray(tlbr, dtype=np.float32)), score, cls)

    def _record_lateral_offset(
        self,
        delta_center: np.ndarray,
        flow_perp: Optional[np.ndarray],
        max_history: int = 30,
    ) -> None:
        if flow_perp is None:
            return

        lateral_offset = float(np.dot(delta_center, flow_perp))
        self.lateral_offsets.append(lateral_offset)

        if len(self.lateral_offsets) > max_history:
            self.lateral_offsets = self.lateral_offsets[-max_history:]

    def predict(self, kalman_filter: Optional[EelKalmanFilterWH] = None) -> None:
        if self.mean is None or self.covariance is None:
            return

        kf = kalman_filter if kalman_filter is not None else self.shared_kalman
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

    @staticmethod
    def multi_predict(
        tracks: Sequence["STrack"],
        kalman_filter: Optional[EelKalmanFilterWH] = None,
    ) -> None:
        if len(tracks) == 0:
            return

        kf = kalman_filter if kalman_filter is not None else STrack.shared_kalman
        for track in tracks:
            track.predict(kf)

    def activate(self, kalman_filter: EelKalmanFilterWH, frame_id: int) -> None:
        self.track_id = self.next_id()

        measurement = tlwh_to_xywh(self._tlwh)
        self.mean, self.covariance = kalman_filter.initiate(measurement)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        self.center_history.append(self.center)

    def re_activate(
        self,
        new_track: "STrack",
        frame_id: int,
        kalman_filter: EelKalmanFilterWH,
        flow_perp: Optional[np.ndarray] = None,
        new_id: bool = False,
    ) -> None:
        if self.mean is None or self.covariance is None:
            return

        pred_center = self.mean[:2].copy()
        measurement = tlwh_to_xywh(new_track.tlwh)

        self.mean, self.covariance = kalman_filter.update(
            self.mean,
            self.covariance,
            measurement,
        )

        self._record_lateral_offset(measurement[:2] - pred_center, flow_perp)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.cls = new_track.cls

        if new_id:
            self.track_id = self.next_id()

        self.center_history.append(self.center)

    def update(
        self,
        new_track: "STrack",
        frame_id: int,
        kalman_filter: EelKalmanFilterWH,
        flow_perp: Optional[np.ndarray] = None,
    ) -> None:
        if self.mean is None or self.covariance is None:
            return

        pred_center = self.mean[:2].copy()
        measurement = tlwh_to_xywh(new_track.tlwh)

        self.mean, self.covariance = kalman_filter.update(
            self.mean,
            self.covariance,
            measurement,
        )

        self._record_lateral_offset(measurement[:2] - pred_center, flow_perp)

        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls

        self.center_history.append(self.center)


# ----------------------------------------------------------------------
# Track-list utilities
# ----------------------------------------------------------------------

def joint_stracks(a: Sequence[STrack], b: Sequence[STrack]) -> List[STrack]:
    exists = {}
    res = []

    for track in a:
        exists[track.track_id] = 1
        res.append(track)

    for track in b:
        if track.track_id not in exists:
            exists[track.track_id] = 1
            res.append(track)

    return res


def sub_stracks(a: Sequence[STrack], b: Sequence[STrack]) -> List[STrack]:
    track_ids_b = {t.track_id for t in b}
    return [t for t in a if t.track_id not in track_ids_b]


def remove_duplicate_stracks(
    stracksa: Sequence[STrack],
    stracksb: Sequence[STrack],
    duplicate_iou_thresh: float = 0.85,
) -> Tuple[List[STrack], List[STrack]]:
    """
    Remove duplicate tracks between tracked and lost lists.
    If two tracks have high IoU, keep the longer-lived one.
    """
    if len(stracksa) == 0 or len(stracksb) == 0:
        return list(stracksa), list(stracksb)

    ious = bbox_ious(
        np.asarray([t.tlbr for t in stracksa], dtype=np.float32),
        np.asarray([t.tlbr for t in stracksb], dtype=np.float32),
    )

    duplicates = np.where(ious > duplicate_iou_thresh)

    remove_a = set()
    remove_b = set()

    for i, j in zip(*duplicates):
        time_a = stracksa[i].frame_id - stracksa[i].start_frame
        time_b = stracksb[j].frame_id - stracksb[j].start_frame

        if time_a > time_b:
            remove_b.add(j)
        else:
            remove_a.add(i)

    res_a = [t for i, t in enumerate(stracksa) if i not in remove_a]
    res_b = [t for i, t in enumerate(stracksb) if i not in remove_b]

    return res_a, res_b


# ----------------------------------------------------------------------
# Tracker configuration
# ----------------------------------------------------------------------

@dataclass
class EelTrackerArgs:
    # Detection thresholds
    track_thresh: float = 0.30
    low_thresh: float = 0.10
    new_track_thresh: float = 0.30

    # Matching thresholds
    # In ByteTrack this is usually a maximum IoU-distance threshold.
    # cost = 1 - IoU, so match_thresh=0.9 means IoU >= 0.1.
    match_thresh: float = 0.90
    second_match_thresh: float = 0.90

    # Track buffer
    max_time_lost: int = 40

    # Misc
    min_box_area: float = 0.0
    fuse_score: bool = False
    duplicate_iou_thresh: float = 0.85

    # Flow-aligned motion corridor
    enable_motion_corridor: bool = True
    flow_direction: Tuple[float, float] = (0.0, 1.0)

    # Paper parameters for 640 x 640 input resolution
    corridor_alpha: float = 2.5
    corridor_beta: float = 10.0
    corridor_gamma: float = 2.0
    corridor_eta: float = 8.0

    # Practical warm-up: disable corridor for very new tracks.
    # This avoids fragmentation before velocity is estimated.
    # Set to 0 if you want the strictest version.
    corridor_warmup: int = 2

    # Set to 0.0 for the strict equation 0 <= s_ij.
    backward_tolerance: float = 0.0

    # Invalid association cost after corridor gating
    invalid_cost: float = 1e5


# ----------------------------------------------------------------------
# BYTETrackerEel
# ----------------------------------------------------------------------

class BYTETrackerEel:
    """
    Improved ByteTrack tracker for rice field eel head tracking.
    """

    def __init__(
        self,
        args: Optional[EelTrackerArgs] = None,
        frame_rate: int = 120,
        **kwargs,
    ) -> None:
        if args is None:
            args = EelTrackerArgs()

        for key, value in kwargs.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown tracker argument: {key}")
            setattr(args, key, value)

        self.args = args
        self.frame_rate = frame_rate

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.kalman_filter = EelKalmanFilterWH()

        self.flow_dir = np.asarray(args.flow_direction, dtype=np.float32)
        norm = np.linalg.norm(self.flow_dir)
        if norm < 1e-12:
            raise ValueError("flow_direction must be a non-zero vector.")
        self.flow_dir = self.flow_dir / norm

        # A 90-degree rotation. For u=(0,1), u_perp=(-1,0).
        self.flow_perp = np.asarray(
            [-self.flow_dir[1], self.flow_dir[0]],
            dtype=np.float32,
        )

    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        BaseTrack.reset_id()

    def _make_detections(self, dets: np.ndarray) -> Tuple[List[STrack], List[STrack]]:
        """
        Convert raw detector outputs to high-confidence and low-confidence tracks.
        """
        if dets is None:
            dets = np.empty((0, 5), dtype=np.float32)

        dets = np.asarray(dets, dtype=np.float32)

        if dets.size == 0:
            dets = np.empty((0, 6), dtype=np.float32)

        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        if dets.shape[1] < 5:
            raise ValueError(
                "detections must have shape [N, 5] or [N, 6]: "
                "[x1, y1, x2, y2, score, optional_cls]"
            )

        boxes = dets[:, :4]
        scores = dets[:, 4]
        classes = dets[:, 5] if dets.shape[1] >= 6 else np.full_like(scores, -1)

        high_mask = scores >= self.args.track_thresh
        low_mask = (scores >= self.args.low_thresh) & (scores < self.args.track_thresh)

        high_dets: List[STrack] = []
        low_dets: List[STrack] = []

        for box, score, cls in zip(boxes[high_mask], scores[high_mask], classes[high_mask]):
            tlwh = tlbr_to_tlwh(box)
            if tlwh[2] * tlwh[3] < self.args.min_box_area:
                continue
            high_dets.append(STrack(tlwh, float(score), float(cls)))

        for box, score, cls in zip(boxes[low_mask], scores[low_mask], classes[low_mask]):
            tlwh = tlbr_to_tlwh(box)
            if tlwh[2] * tlwh[3] < self.args.min_box_area:
                continue
            low_dets.append(STrack(tlwh, float(score), float(cls)))

        return high_dets, low_dets

    def _motion_corridor_mask(
        self,
        tracks: Sequence[STrack],
        detections: Sequence[STrack],
    ) -> np.ndarray:
        """
        Build a boolean matrix indicating whether each detection is inside
        the flow-aligned motion corridor of each track.
        """
        num_tracks = len(tracks)
        num_dets = len(detections)

        mask = np.ones((num_tracks, num_dets), dtype=bool)

        if not self.args.enable_motion_corridor:
            return mask

        if num_tracks == 0 or num_dets == 0:
            return mask

        for i, track in enumerate(tracks):
            if track.mean is None:
                continue

            # Disable corridor during the first few associations to estimate velocity.
            if track.tracklet_len < self.args.corridor_warmup:
                continue

            pred_center = track.mean[:2].astype(np.float32)
            pred_velocity = track.mean[4:6].astype(np.float32)

            search_length = (
                self.args.corridor_alpha
                + self.args.corridor_beta * float(np.linalg.norm(pred_velocity))
            )
            lateral_half_width = (
                self.args.corridor_gamma
                + self.args.corridor_eta * track.lateral_sigma
            )

            for j, det in enumerate(detections):
                det_center = det.center.astype(np.float32)
                delta = det_center - pred_center

                along = float(np.dot(delta, self.flow_dir))
                lateral = float(np.dot(delta, self.flow_perp))

                valid = (
                    -self.args.backward_tolerance <= along <= search_length
                    and abs(lateral) <= lateral_half_width
                )
                mask[i, j] = valid

        return mask

    def _matching_distance(
        self,
        tracks: Sequence[STrack],
        detections: Sequence[STrack],
    ) -> np.ndarray:
        """
        IoU distance with optional score fusion and motion-corridor gating.
        """
        dists = iou_distance(tracks, detections)

        if self.args.fuse_score:
            dists = fuse_score(dists, detections)

        if self.args.enable_motion_corridor and len(tracks) > 0 and len(detections) > 0:
            valid_mask = self._motion_corridor_mask(tracks, detections)
            dists = dists.copy()
            dists[~valid_mask] = self.args.invalid_cost

        return dists.astype(np.float32)

    def update(
        self,
        detections: Optional[np.ndarray],
        return_numpy: bool = False,
    ) -> Union[List[STrack], np.ndarray]:
        """
        Update tracker with detections of the current frame.

        Args:
            detections:
                np.ndarray with shape [N, 5] or [N, 6].
                Format: [x1, y1, x2, y2, score, optional_cls].
            return_numpy:
                If True, return an ndarray with rows:
                [x1, y1, x2, y2, track_id, score, cls]

        Returns:
            Active tracks or ndarray results.
        """
        self.frame_id += 1

        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        detections_high, detections_low = self._make_detections(detections)

        # Step 1: predict existing tracks.
        tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool, self.kalman_filter)

        # Step 2: first association with high-confidence detections.
        dists = self._matching_distance(strack_pool, detections_high)
        matches, u_track, u_detection = linear_assignment_with_threshold(
            dists,
            self.args.match_thresh,
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]

            if track.state == TrackState.Tracked:
                track.update(
                    det,
                    self.frame_id,
                    self.kalman_filter,
                    flow_perp=self.flow_perp,
                )
                activated_stracks.append(track)
            else:
                track.re_activate(
                    det,
                    self.frame_id,
                    self.kalman_filter,
                    flow_perp=self.flow_perp,
                    new_id=False,
                )
                refind_stracks.append(track)

        # Step 3: second association with low-confidence detections.
        # Only unmatched tracks that are still in Tracked state participate.
        remaining_tracked = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]

        dists_low = self._matching_distance(remaining_tracked, detections_low)
        matches_low, u_track_low, _ = linear_assignment_with_threshold(
            dists_low,
            self.args.second_match_thresh,
        )

        for itracked, idet in matches_low:
            track = remaining_tracked[itracked]
            det = detections_low[idet]

            track.update(
                det,
                self.frame_id,
                self.kalman_filter,
                flow_perp=self.flow_perp,
            )
            activated_stracks.append(track)

        # Unmatched remaining tracked tracks become lost.
        for it in u_track_low:
            track = remaining_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 4: initialize new tracks from unmatched high-confidence detections.
        for inew in u_detection:
            track = detections_high[inew]

            if track.score < self.args.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 5: remove lost tracks after max_time_lost frames.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.args.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Step 6: update track lists.
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

        self.removed_stracks.extend(removed_stracks)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks,
            self.lost_stracks,
            duplicate_iou_thresh=self.args.duplicate_iou_thresh,
        )

        output_tracks = [
            track
            for track in self.tracked_stracks
            if track.is_activated and track.state == TrackState.Tracked
        ]

        if return_numpy:
            return self.tracks_to_numpy(output_tracks)

        return output_tracks

    @staticmethod
    def tracks_to_numpy(tracks: Sequence[STrack]) -> np.ndarray:
        """
        Convert active tracks to ndarray:
            [x1, y1, x2, y2, track_id, score, cls]
        """
        if len(tracks) == 0:
            return np.empty((0, 7), dtype=np.float32)

        rows = []
        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            rows.append([x1, y1, x2, y2, t.track_id, t.score, t.cls])

        return np.asarray(rows, dtype=np.float32)

    def update_from_ultralytics(
        self,
        result,
        return_numpy: bool = False,
    ) -> Union[List[STrack], np.ndarray]:
        """
        Convenience wrapper for an Ultralytics YOLO result object.

        Example:
            results = model(frame)
            tracks = tracker.update_from_ultralytics(results[0])
        """
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return self.update(np.empty((0, 6), dtype=np.float32), return_numpy)

        boxes = result.boxes

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy().reshape(-1, 1)

        if getattr(boxes, "cls", None) is not None:
            cls = boxes.cls.detach().cpu().numpy().reshape(-1, 1)
        else:
            cls = -np.ones((xyxy.shape[0], 1), dtype=np.float32)

        dets = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
        return self.update(dets, return_numpy=return_numpy)


# ----------------------------------------------------------------------
# Minimal usage example
# ----------------------------------------------------------------------

if __name__ == "__main__":
    tracker = BYTETrackerEel()

    # Example detections from one frame:
    # [x1, y1, x2, y2, score, cls]
    dets = np.array(
        [
            [100, 120, 130, 150, 0.92, 0],
            [220, 80, 250, 112, 0.75, 0],
        ],
        dtype=np.float32,
    )

    tracks = tracker.update(dets, return_numpy=True)
    print(tracks)