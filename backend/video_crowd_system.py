# -*- coding: utf-8 -*-
"""
VIDEO CROWD FLOW ANALYSIS SYSTEM
High-precision crowd flow & movement classification:
- Distinguishes sitting/stationary crowds (talking, head/hand movement) -> STATIONARY
- Detects true spatial displacement across frames -> MOVING (RIGHT / LEFT / FORWARD / BACKWARD)
- Background-isolated camera motion compensation with RANSAC
- Foreground person optical flow + multi-frame centroid trajectory tracking
- Adaptive thresholding scaled to resolution and person scale
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.model.csrnet import build_csrnet

# ─────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────

# Target sampling rate: ~6 processed frames per second
TARGET_PROCESS_FPS = 6.0

# Detection confidence
DETECTION_CONF = 0.35

# CSRNet density settings
DENSITY_SAMPLE_COUNT = 8
VIDEO_DENSITY_MAX_DIM = 1280
DENSE_SCENE_COUNT_THRESHOLD = 25

# ── Multi-frame Centroid Tracking ─────────────
MIN_CONFIRMED_OBSERVATIONS = 5
IOU_MATCH_THRESHOLD = 0.30
CENTROID_MATCH_PX = 180
MAX_TRACK_GAP = 8
TEMPORAL_WINDOW = 8
TRACK_MOVING_FRACTION = 0.50

# ── Adaptive Motion Thresholds ────────────────
# Minimum persistent velocity (pixels per frame in normalized space)
# to distinguish genuine walking displacement from micro-movements (sitting/talking)
BASE_MOTION_THRESHOLD_PX = 1.8     # base pixel threshold at 720p equivalent
PERSISTENCE_MIN_RATIO = 0.28       # at least 28% of video frames must exhibit coherent directional displacement
CONSISTENCY_MIN_RATIO = 0.55       # directional consensus among active frames
DIRECTION_DOMINANCE_RATIO = 1.20   # ratio of top direction over 2nd direction


# ─────────────────────────────────────────────
# VideoAnalyzer
# ─────────────────────────────────────────────

class VideoAnalyzer:
    """
    Analyzes crowd video to determine crowd count, movement status,
    and directional flow (RIGHT, LEFT, FORWARD, BACKWARD, or STATIONARY).
    """

    def __init__(self):
        self.detector = YOLO("yolov8n.pt")
        self.density_model = build_csrnet(device=torch.device("cpu"))
        self.density_model.eval()

    def analyze_video(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

        sample_every = max(1, int(round(fps / TARGET_PROCESS_FPS)))

        print(
            f"[VideoAnalyzer] {w}x{h} | {total_frames} frames "
            f"@ {fps:.1f} FPS | sampling every {sample_every} frames"
        )

        # State tracking
        tracks: dict = {}
        next_id: list = [0]
        frame_detection_counts = []
        density_counts = []
        density_sample_every = max(1, total_frames // DENSITY_SAMPLE_COUNT)

        prev_gray = None
        prev_boxes = []
        sampled_idx = 0

        # Frame-level optical flow displacement records: (med_dx, med_dy, num_points, person_scale)
        frame_flow_records: list = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                # 1. CSRNet density estimation (sampled sparsely)
                if frame_idx % density_sample_every == 0:
                    density_counts.append(self._estimate_density_count(frame))

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 2. YOLO person detection
                detections = self._detect(frame)
                frame_detection_counts.append(len(detections))

                if prev_gray is not None:
                    # 3. Background-isolated Camera Motion Estimation (RANSAC)
                    cam_dx, cam_dy = self._estimate_camera_motion(
                        prev_gray, gray, prev_boxes, detections
                    )

                    # 4. Foreground Person Optical Flow (pure human motion)
                    flow_entry = self._compute_person_optical_flow(
                        prev_gray, gray, prev_boxes, detections, cam_dx, cam_dy
                    )
                    frame_flow_records.append(flow_entry)

                    # 5. Multi-frame Centroid Tracking (camera-compensated)
                    self._update_tracks(
                        tracks, next_id, detections, sampled_idx, cam_dx, cam_dy
                    )
                else:
                    self._update_tracks(
                        tracks, next_id, detections, sampled_idx, 0.0, 0.0
                    )

                prev_gray = gray
                prev_boxes = detections
                sampled_idx += 1

            frame_idx += 1

        cap.release()

        # ── Compute People Count ───────────────────────────────────────
        tracking_summary = self._summarize_tracks(tracks, sampled_idx)
        tracked_count = tracking_summary["people_count"]
        tracked_moving_count = tracking_summary["moving_count"]

        density_count = self._aggregate_density_counts(density_counts)
        frame_count = self._aggregate_frame_detection_counts(frame_detection_counts)

        if density_count is not None and density_count >= DENSE_SCENE_COUNT_THRESHOLD:
            people_count = density_count
        elif frame_count is not None and frame_count > 0:
            people_count = frame_count
        else:
            people_count = max(1, tracked_count)

        # ── Dual-Engine Flow Analysis ──────────────────────────────────
        flow_result = self._evaluate_crowd_flow(
            frame_flow_records, tracking_summary, people_count, w, h
        )

        final_result = {
            "people_count": people_count,
            "movement_status": flow_result["movement_status"],
            "confidence": flow_result["confidence"],
            "stationary_count": flow_result["stationary_count"],
            "moving_count": flow_result["moving_count"],
            "stationary_pct": flow_result["stationary_pct"],
            "moving_pct": flow_result["moving_pct"],
            "flow": flow_result["flow"],
            "dominant_direction": flow_result["dominant_direction"],
        }

        print(
            f"[VideoAnalyzer] Results -> people={people_count} | "
            f"status={final_result['movement_status']} | "
            f"direction={final_result['dominant_direction']} | "
            f"flow={final_result['flow']} | "
            f"stationary={final_result['stationary_pct']}% moving={final_result['moving_pct']}%"
        )
        return final_result

    # ─── 1. Background Camera Motion ──────────────────────────────────

    def _estimate_camera_motion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_boxes: list,
        curr_boxes: list,
    ) -> tuple:
        """
        Estimate camera motion by tracking features ONLY in background regions
        (outside detected person bounding boxes). Uses RANSAC affine estimation.
        """
        h, w = prev_gray.shape

        # Create background mask (255 where no person is present)
        bg_mask = np.full((h, w), 255, dtype=np.uint8)
        for b in prev_boxes + curr_boxes:
            x1 = max(0, int(b[0]) - 5)
            y1 = max(0, int(b[1]) - 5)
            x2 = min(w, int(b[2]) + 5)
            y2 = min(h, int(b[3]) + 5)
            bg_mask[y1:y2, x1:x2] = 0

        # If background is too small (very dense crowd filling whole screen), fallback to full frame
        if int(bg_mask.sum() // 255) < (w * h * 0.15):
            bg_mask = None

        try:
            pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=300,
                qualityLevel=0.01,
                minDistance=8,
                blockSize=5,
                mask=bg_mask,
            )
            if pts is None or len(pts) < 8:
                return 0.0, 0.0

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts, None,
                winSize=(15, 15), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            if next_pts is None:
                return 0.0, 0.0

            good_prev = pts[status.ravel() == 1]
            good_next = next_pts[status.ravel() == 1]

            if len(good_prev) < 4:
                return 0.0, 0.0

            transform, inliers = cv2.estimateAffinePartial2D(
                good_prev, good_next, method=cv2.RANSAC,
                ransacReprojThreshold=3.0, maxIters=400,
                confidence=0.99, refineIters=10,
            )
            if transform is None or inliers is None or int(inliers.sum()) < 6:
                return 0.0, 0.0

            return float(transform[0, 2]), float(transform[1, 2])
        except Exception:
            return 0.0, 0.0

    # ─── 2. Foreground Person Optical Flow ────────────────────────────

    def _compute_person_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_boxes: list,
        curr_boxes: list,
        cam_dx: float,
        cam_dy: float,
    ) -> tuple:
        """
        Track keypoints inside person bounding boxes or across full frame (with RANSAC
        camera compensation) when bounding boxes are sparse or missing in dense/portrait scenes.
        Returns (med_dx, med_dy, valid_pts_count, median_person_scale).
        """
        boxes = curr_boxes if len(curr_boxes) > 0 else prev_boxes
        h, w = prev_gray.shape
        person_mask = None
        median_scale = 50.0

        if boxes and len(boxes) >= 2:
            person_mask = np.zeros((h, w), dtype=np.uint8)
            person_scales = []
            for b in boxes:
                x1 = max(0, int(b[0]))
                y1 = max(0, int(b[1]))
                x2 = min(w, int(b[2]))
                y2 = min(h, int(b[3]))
                bw = x2 - x1
                bh = y2 - y1
                if bw >= 6 and bh >= 10:
                    person_mask[y1:y2, x1:x2] = 255
                    person_scales.append(min(bw, bh))
            if person_scales:
                median_scale = float(np.median(person_scales))

        # Shi-Tomasi feature detection (inside person mask if available, else full frame)
        pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=600,
            qualityLevel=0.005,
            minDistance=5,
            blockSize=5,
            mask=person_mask,
        )
        if pts is None or len(pts) < 6:
            return 0.0, 0.0, 0, median_scale

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None,
            winSize=(21, 21), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        if next_pts is None:
            return 0.0, 0.0, 0, median_scale

        good_prev = pts[status.ravel() == 1]
        good_next = next_pts[status.ravel() == 1]

        if len(good_prev) < 6:
            return 0.0, 0.0, 0, median_scale

        # Compute camera-corrected motion vectors
        dx_all = (good_next[:, 0, 0] - good_prev[:, 0, 0]) - cam_dx
        dy_all = (good_next[:, 0, 1] - good_prev[:, 0, 1]) - cam_dy
        mags = np.hypot(dx_all, dy_all)

        # Filter points above noise floor (ignore static background points)
        active_mask = mags > 1.2
        if active_mask.sum() >= 4:
            med_dx = float(np.median(dx_all[active_mask]))
            med_dy = float(np.median(dy_all[active_mask]))
            valid_count = int(active_mask.sum())
        else:
            med_dx = float(np.median(dx_all))
            med_dy = float(np.median(dy_all))
            valid_count = len(dx_all)

        return med_dx, med_dy, valid_count, median_scale

    # ─── 3. Dual-Engine Flow Evaluation ───────────────────────────────

    def _evaluate_crowd_flow(
        self,
        flow_records: list,
        tracking_summary: dict,
        people_count: int,
        frame_width: int,
        frame_height: int,
    ) -> dict:
        """
        Synthesizes foreground optical flow and multi-frame track trajectories.
        Strictly distinguishes sitting/standing still (STATIONARY) vs walking (MOVING).
        """
        # Default STATIONARY response
        stationary_response = {
            "movement_status": "STATIONARY",
            "dominant_direction": "Mostly Stationary",
            "confidence": "HIGH",
            "stationary_count": people_count,
            "moving_count": 0,
            "stationary_pct": 100,
            "moving_pct": 0,
            "flow": {"Right": 0, "Left": 0, "Forward": 0, "Backward": 0},
        }

        if not flow_records:
            return stationary_response

        # Calculate resolution normalization factor (standardized to 720p base)
        res_diag = np.hypot(frame_width, frame_height)
        scale_factor = max(0.5, res_diag / 1468.0)  # 1468 is diagonal of 1280x720

        # Extract active displacement records
        valid_dx = []
        valid_dy = []
        valid_mags = []
        dir_magnitudes = {"Right": 0.0, "Left": 0.0, "Forward": 0.0, "Backward": 0.0}

        for dx, dy, count, person_scale in flow_records:
            if count < 6:
                continue

            mag = np.hypot(dx, dy)
            # Adaptive threshold: scaled with person size and resolution
            # A person walking moves >= 2.5% of their bbox side per frame
            dyn_threshold = max(
                BASE_MOTION_THRESHOLD_PX * scale_factor,
                0.025 * person_scale
            )

            if mag >= dyn_threshold:
                valid_dx.append(dx)
                valid_dy.append(dy)
                valid_mags.append(mag)
                d = _classify_cardinal_direction(dx, dy)
                dir_magnitudes[d] += mag

        total_frames = len(flow_records)
        active_frames = len(valid_dx)
        persistence_ratio = active_frames / total_frames if total_frames > 0 else 0.0

        # Check trajectory tracking evidence
        tracked_people = tracking_summary["people_count"]
        tracked_moving = tracking_summary["moving_count"]
        tracked_moving_ratio = tracked_moving / tracked_people if tracked_people > 0 else 0.0

        # Case A: If tracking confirms all people are stationary (e.g. 4 sitting people)
        # and optical flow persistence is low -> STATIONARY
        if tracked_people >= 2 and tracked_moving == 0 and persistence_ratio < 0.40:
            return stationary_response

        # Case B: If persistence ratio is below required minimum -> STATIONARY (sitting/talking)
        if persistence_ratio < PERSISTENCE_MIN_RATIO and tracked_moving_ratio < 0.35:
            return stationary_response

        if not valid_dx:
            return stationary_response

        # Compute robust directional statistics
        med_dx = float(np.median(valid_dx))
        med_dy = float(np.median(valid_dy))
        net_mag = np.hypot(med_dx, med_dy)

        # Cross-frame directional consistency check
        if abs(med_dx) >= abs(med_dy):
            same_dir = sum(1 for x in valid_dx if x * med_dx > 0)
            consistency = same_dir / active_frames if active_frames > 0 else 0.0
        else:
            same_dir = sum(1 for y in valid_dy if y * med_dy > 0)
            consistency = same_dir / active_frames if active_frames > 0 else 0.0

        if consistency < CONSISTENCY_MIN_RATIO:
            # Micro-movements in conflicting directions (e.g. sitting room) -> STATIONARY
            return stationary_response

        # Determine dominant direction
        total_mag = sum(dir_magnitudes.values())
        if total_mag == 0:
            return stationary_response

        sorted_dirs = sorted(dir_magnitudes.items(), key=lambda x: x[1], reverse=True)
        top_dir, top_mag = sorted_dirs[0]
        second_mag = sorted_dirs[1][1] if len(sorted_dirs) > 1 else 0.0

        top_fraction = top_mag / total_mag
        dominance = top_mag / second_mag if second_mag > 0 else float("inf")

        if top_fraction < 0.35 or (dominance < DIRECTION_DOMINANCE_RATIO and second_mag > 0):
            # Equal movement in opposing directions without dominant flow
            dominant_label = "Mixed Flow"
        else:
            dir_labels = {
                "Right": "RIGHT \u2192",
                "Left": "LEFT \u2190",
                "Forward": "FORWARD \u2191",
                "Backward": "BACKWARD \u2193",
            }
            dominant_label = dir_labels.get(top_dir, top_dir)

        # Flow percentage breakdown (normalized to 100%)
        flow_pcts = {
            d: int(round(100.0 * m / total_mag))
            for d, m in dir_magnitudes.items()
        }
        diff = 100 - sum(flow_pcts.values())
        if diff != 0:
            flow_pcts[top_dir] += diff

        # Estimate moving percentage based on persistence and tracked moving ratio
        est_moving_pct = int(round(np.clip(
            max(persistence_ratio, tracked_moving_ratio) * 100 * 1.1,
            20, 100
        )))
        est_stationary_pct = 100 - est_moving_pct
        moving_count = max(1, int(round(people_count * est_moving_pct / 100.0)))
        stationary_count = people_count - moving_count

        confidence = "HIGH" if (persistence_ratio >= 0.50 and consistency >= 0.70) else "MEDIUM"

        return {
            "movement_status": "MOVING",
            "dominant_direction": dominant_label,
            "confidence": confidence,
            "stationary_count": stationary_count,
            "moving_count": moving_count,
            "stationary_pct": est_stationary_pct,
            "moving_pct": est_moving_pct,
            "flow": flow_pcts,
        }

    # ─── 4. Multi-frame Centroid Tracker ──────────────────────────────

    @staticmethod
    def _new_track(box: list, cx: float, cy: float, sampled_idx: int) -> dict:
        return {
            "observations": 1,
            "last_seen": sampled_idx,
            "first_seen": sampled_idx,
            "centroids": [(sampled_idx, cx, cy)],
            "bbox_sizes": [min(box[2] - box[0], box[3] - box[1])],
            "last_box": box,
        }

    def _update_tracks(
        self,
        tracks: dict,
        next_id: list,
        detections: list,
        sampled_idx: int,
        cam_dx: float,
        cam_dy: float,
    ):
        """IoU + Centroid track matching with camera motion compensation."""
        predicted = {}
        for tid, tk in tracks.items():
            if sampled_idx - tk["last_seen"] > MAX_TRACK_GAP:
                continue
            _, pcx, pcy = tk["centroids"][-1]
            predicted[tid] = (pcx + cam_dx, pcy + cam_dy)

        assigned_tids = set()
        matched_dets = set()

        det_centers = [
            ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            for b in detections
        ]

        # 1. IoU Match
        for di, box in enumerate(detections):
            best_iou = IOU_MATCH_THRESHOLD
            best_tid = None
            for tid in predicted:
                if tid in assigned_tids:
                    continue
                last_box = tracks[tid].get("last_box")
                if last_box is None:
                    continue
                shifted = [
                    last_box[0] + cam_dx, last_box[1] + cam_dy,
                    last_box[2] + cam_dx, last_box[3] + cam_dy,
                ]
                iou = _compute_iou(box, shifted)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                cx, cy = det_centers[di]
                _update_track_entry(tracks[best_tid], box, cx - cam_dx, cy - cam_dy, sampled_idx)
                assigned_tids.add(best_tid)
                matched_dets.add(di)

        # 2. Centroid Distance Match
        for di, box in enumerate(detections):
            if di in matched_dets:
                continue
            cx, cy = det_centers[di]
            best_dist = CENTROID_MATCH_PX
            best_tid = None
            for tid in predicted:
                if tid in assigned_tids:
                    continue
                pcx, pcy = predicted[tid]
                dist = float(np.hypot(cx - pcx, cy - pcy))
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                _update_track_entry(tracks[best_tid], box, cx - cam_dx, cy - cam_dy, sampled_idx)
                assigned_tids.add(best_tid)
                matched_dets.add(di)

        # 3. New Tracks
        for di, box in enumerate(detections):
            if di in matched_dets:
                continue
            cx, cy = det_centers[di]
            tid = next_id[0]
            next_id[0] += 1
            tracks[tid] = VideoAnalyzer._new_track(box, cx - cam_dx, cy - cam_dy, sampled_idx)

    def _summarize_tracks(self, tracks: dict, total_sampled: int) -> dict:
        """Classify individual tracks as stationary or moving based on cumulative trajectory."""
        stationary_count = 0
        moving_count = 0

        for tid, tk in tracks.items():
            min_obs = min(
                MIN_CONFIRMED_OBSERVATIONS,
                max(2, int(round(total_sampled * 0.05)))
            )
            if tk["observations"] < min_obs:
                continue

            median_bbox = float(np.median(tk["bbox_sizes"]))
            disp_thresh = max(8.0, 0.08 * median_bbox)

            centroids = tk["centroids"]
            n = len(centroids)
            if n < 2:
                stationary_count += 1
                continue

            windows_total = 0
            windows_moving = 0
            step = max(1, TEMPORAL_WINDOW // 3)
            i = 0
            while i + TEMPORAL_WINDOW <= n:
                _, ax, ay = centroids[i]
                _, bx, by = centroids[i + TEMPORAL_WINDOW - 1]
                disp = float(np.hypot(bx - ax, by - ay))
                windows_total += 1
                if disp > disp_thresh:
                    windows_moving += 1
                i += step

            if windows_total == 0:
                _, ax, ay = centroids[0]
                _, bx, by = centroids[-1]
                disp = float(np.hypot(bx - ax, by - ay))
                if disp > disp_thresh:
                    moving_count += 1
                else:
                    stationary_count += 1
                continue

            if windows_moving / windows_total >= TRACK_MOVING_FRACTION:
                moving_count += 1
            else:
                stationary_count += 1

        total = stationary_count + moving_count
        return {
            "people_count": total,
            "moving_count": moving_count,
            "stationary_count": stationary_count,
        }

    # ─── 5. CSRNet Density Estimation ─────────────────────────────────

    def _estimate_density_count(self, frame: np.ndarray) -> int:
        height, width = frame.shape[:2]
        scale = min(1.0, VIDEO_DENSITY_MAX_DIM / max(height, width))
        if scale < 1.0:
            frame = cv2.resize(
                frame, (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            height, width = frame.shape[:2]
        pad_height = (8 - height % 8) % 8
        pad_width = (8 - width % 8) % 8
        if pad_height or pad_width:
            frame = cv2.copyMakeBorder(
                frame, 0, pad_height, 0, pad_width, cv2.BORDER_REFLECT
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0)
        with torch.no_grad():
            return max(0, int(round(float(self.density_model(tensor).sum().item()))))

    @staticmethod
    def _aggregate_density_counts(counts: list) -> int | None:
        valid = sorted(c for c in counts if c >= 0)
        if not valid:
            return None
        if len(valid) >= 5:
            valid = valid[1:-1]
        return int(round(float(np.median(valid))))

    @staticmethod
    def _aggregate_frame_detection_counts(counts: list) -> int | None:
        valid = [c for c in counts if c > 0]
        if not valid:
            return None
        return max(1, int(round(float(np.percentile(valid, 90)))))

    def _detect(self, frame: np.ndarray) -> list:
        out = self.detector(
            frame, classes=[0], conf=DETECTION_CONF, verbose=False
        )[0]
        boxes = []
        if out.boxes is not None and len(out.boxes) > 0:
            for b in out.boxes.xyxy.cpu().numpy():
                boxes.append(b.tolist())
        return boxes


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def _classify_cardinal_direction(dx: float, dy: float) -> str:
    """Classifies (dx, dy) into Right, Left, Forward (downwards), Backward (upwards)."""
    if abs(dx) >= abs(dy):
        return "Right" if dx > 0 else "Left"
    else:
        return "Forward" if dy > 0 else "Backward"


def _compute_iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _update_track_entry(tk: dict, box: list, cx: float, cy: float, sampled_idx: int):
    tk["observations"] += 1
    tk["last_seen"] = sampled_idx
    tk["centroids"].append((sampled_idx, cx, cy))
    tk["bbox_sizes"].append(min(box[2] - box[0], box[3] - box[1]))
    tk["last_box"] = box


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python video_crowd_system.py <video_path>")
        sys.exit(1)

    analyzer = VideoAnalyzer()
    res = analyzer.analyze_video(sys.argv[1])
    print("\n" + "=" * 60)
    print(f"People Count      : {res['people_count']}")
    print(f"Movement Status   : {res['movement_status']}")
    print(f"Dominant Direction: {res['dominant_direction']}")
    print(f"Flow Distribution : {res['flow']}")
    print(f"Confidence        : {res['confidence']}")
    print("=" * 60)
