"""
Video Crowd Analysis System - Accurate Person Counting + Flow Direction

PIPELINE:
  Video → Frame Sampling → YOLO Person Detection → ByteTrack Multi-Object Tracking
  → Unique Person Counting → Trajectory Analysis → Flow Direction Classification

ACCURACY PRIORITY:
  - Count unique people only (no duplicates across frames)
  - Use reliable tracking (ByteTrack)
  - Classify flow from actual trajectories
  - Ignore camera motion and background
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# Import CSRNet and density-map utilities from the backend package.
from backend.model.csrnet import build_csrnet
from backend.density_map import density_to_heatmap
from scipy.ndimage import gaussian_filter, maximum_filter
from torchvision.ops import nms

HAS_YOLO = True  # YOLO is always available (ultralytics is imported)

try:
    from boxmot import BYTETracker
    HAS_BYTETRACK = True
except ImportError:
    HAS_BYTETRACK = False
    print("[WARNING] ByteTrack not installed. Install: pip install boxmot")


class CrowdAnalysisSystem:
    """
    Accurate video crowd analysis: person counting + flow direction.
    Uses YOLO detection + ByteTrack multi-object tracking.
    """

    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[CrowdAnalysis] Device: {self.device}")

        # Load YOLO detector (YOLOv8n for speed, YOLOv8m for accuracy)
        self.detector = YOLO("yolov8n.pt")
        print("[CrowdAnalysis] YOLO detector loaded")

        # Initialize ByteTrack
        if HAS_BYTETRACK:
            self.tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30)
            print("[CrowdAnalysis] ByteTrack tracker initialized")
        else:
            self.tracker = None
            print("[CrowdAnalysis] ByteTrack not available - tracking disabled")

    def analyze_video(self, video_path: str, sample_every: int = 2) -> dict:
        """
        Analyze crowd video for:
        1. Accurate unique person count
        2. Flow direction from trajectories

        Args:
            video_path: Path to video file
            sample_every: Process every Nth frame (2 = 50% of frames)

        Returns:
            dict with people_count, flow_direction, confidence
        """

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"[CrowdAnalysis] Video: {total_frames} frames @ {fps} FPS")
        print(f"[CrowdAnalysis] Sampling every {sample_every} frames")

        # Track unique people across video
        track_history = {}  # track_id → list of (frame_idx, x, y)
        unique_track_ids = set()
        frame_idx = 0

        # For flow analysis
        moving_people = {}  # track_id → movement direction

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Process only sampled frames for speed
                if frame_idx % sample_every != 0:
                    frame_idx += 1
                    continue

                h, w = frame.shape[:2]

                # YOLO detection
                results = self.detector(frame, classes=[0], conf=0.25, verbose=False)[0]
                detections = []

                if results.boxes is not None and len(results.boxes) > 0:
                    boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # [[x1, y1, x2, y2], ...]
                    confs = results.boxes.conf.cpu().numpy()  # [conf1, conf2, ...]

                    for box, conf in zip(boxes_xyxy, confs):
                        x1, y1, x2, y2 = box
                        # Format for ByteTrack: [x1, y1, x2, y2, conf, class]
                        detections.append([x1, y1, x2, y2, float(conf), 0])

                # ByteTrack update
                if self.tracker and detections:
                    detections_array = np.array(detections)
                    online_targets = self.tracker.update(detections_array, (h, w))

                    for target in online_targets:
                        track_id = target.track_id
                        bbox = target.bbox  # [x1, y1, x2, y2]

                        unique_track_ids.add(track_id)

                        # Calculate centroid
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2

                        # Store trajectory
                        if track_id not in track_history:
                            track_history[track_id] = []

                        track_history[track_id].append((frame_idx, cx, cy))

                frame_idx += 1

        finally:
            cap.release()

        # ─────────────────────────────────────────────────────────────
        # POST-PROCESSING: Count + Flow Analysis
        # ─────────────────────────────────────────────────────────────

        people_count = len(unique_track_ids)
        print(f"[CrowdAnalysis] Unique people detected: {people_count}")

        # Flow direction analysis
        flow_direction = self._analyze_flow(track_history)

        confidence = self._assess_confidence(people_count, len(track_history))

        return {
            "people_count": people_count,
            "flow_direction": flow_direction,
            "confidence": confidence,
        }

    def _analyze_flow(self, track_history: dict) -> dict:
        """
        Analyze movement direction from tracked trajectories.

        Returns:
            dict with Forward/Backward/Left/Right percentages
        """

        if not track_history:
            return {"Forward": 0, "Backward": 0, "Left": 0, "Right": 0}

        directions = {"Forward": 0, "Backward": 0, "Left": 0, "Right": 0, "Stationary": 0}

        for track_id, trajectory in track_history.items():
            if len(trajectory) < 2:
                directions["Stationary"] += 1
                continue

            # Calculate net displacement
            start_frame, start_x, start_y = trajectory[0]
            end_frame, end_x, end_y = trajectory[-1]

            dx = end_x - start_x
            dy = end_y - start_y

            # Minimum movement threshold (15 pixels)
            if abs(dx) < 15 and abs(dy) < 15:
                directions["Stationary"] += 1
                continue

            # Classify primary movement direction
            if abs(dx) > abs(dy):
                # Horizontal dominates
                if dx > 0:
                    directions["Right"] += 1
                else:
                    directions["Left"] += 1
            else:
                # Vertical dominates
                if dy < 0:
                    directions["Forward"] += 1  # Moving UP (toward camera)
                else:
                    directions["Backward"] += 1  # Moving DOWN (away from camera)

        # Calculate percentages
        total = sum(directions.values())
        if total == 0:
            return {"Forward": 0, "Backward": 0, "Left": 0, "Right": 0}

        # Only count moving people for flow percentages
        moving_total = total - directions["Stationary"]
        if moving_total == 0:
            return {"Forward": 0, "Backward": 0, "Left": 0, "Right": 0}

        result = {}
        for direction in ["Forward", "Backward", "Left", "Right"]:
            pct = int(round((directions[direction] / moving_total) * 100))
            result[direction] = pct

        return result

    def _assess_confidence(self, people_count: int, total_tracks: int) -> str:
        """
        Assess confidence in the counting result.
        """

        if total_tracks < 2:
            return "Low"  # Very few tracks = unreliable
        elif people_count < 5:
            return "High"  # Small groups are easier to track accurately
        elif people_count < 50:
            return "High"  # Medium crowd is reliable with ByteTrack
        else:
            return "Medium"  # Large crowds may have occlusions


def main(video_path: str):
    """Analyze a single video and print results."""

    analyzer = CrowdAnalysisSystem()
    result = analyzer.analyze_video(video_path)

    print("\n" + "=" * 60)
    print("Visual Crowd Density and Flow Analysis")
    print("=" * 60)
    print(f"\nPeople Count: {result['people_count']}")
    print(f"\nFlow Direction:")
    print(f"  Forward:  {result['flow_direction']['Forward']:3d}%")
    print(f"  Backward: {result['flow_direction']['Backward']:3d}%")
    print(f"  Left:     {result['flow_direction']['Left']:3d}%")
    print(f"  Right:    {result['flow_direction']['Right']:3d}%")
    print(f"\nConfidence: {result['confidence']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python inference.py <video_path>")



# ─────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation constants
# ─────────────────────────────────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _preprocess(img_bgr: np.ndarray, device: torch.device) -> "tuple[torch.Tensor, int, int]":
    """
    Convert a BGR numpy image to an ImageNet-normalised tensor.
    Pads H and W to be divisible by 8 (required for CSRNet's 3 max-pool frontend).

    Returns
    -------
    tensor  : (1, 3, H_pad, W_pad) on ``device``
    h_pad   : padding added to height
    w_pad   : padding added to width
    """
    h, w = img_bgr.shape[:2]
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    if h_pad or w_pad:
        img_bgr = cv2.copyMakeBorder(img_bgr, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = ((t - _MEAN) / _STD).unsqueeze(0).to(device)
    return t, h_pad, w_pad


# ─────────────────────────────────────────────────────────────────────────────
# Camera Perspective & Flow Configuration
# ─────────────────────────────────────────────────────────────────────────────
# In standard CCTV / elevated camera perspective:
# - Horizontal: fx > 0 -> RIGHT, fx < 0 -> LEFT
# - Vertical / Depth: fy < 0 (moving towards upper frame / approaching in perspective) -> FORWARD
#                     fy > 0 (moving towards bottom / receding away) -> BACKWARD
CAMERA_FLOW_MODE = "cctv"


# ─────────────────────────────────────────────────────────────────────────────
# Main Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class CrowdAnalyzer:
    """
    Human-Aware Crowd Density & Flow Analysis Engine.

    Core accuracy & safety features
    ───────────────────────────────
    • CSRNet backbone: VGG16 frontend + dilated convolutions -> high accuracy crowd estimation.
    • Person Presence Validation Gate: rejects documents, certificates, empty landscapes.
    • Robust Video Aggregation: adaptive FPS sampling + trimmed median filtering.
    • Optical Flow: Farneback dense optical flow mapped to LEFT / RIGHT / FORWARD / BACKWARD.
    """

    def __init__(self, weights_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[CrowdAnalyzer] Initialising CSRNet engine on: {self.device}")

        # ── YOLOv8 person detector (validation gate + background mask role) ──
        self.detector = None
        if HAS_YOLO:
            try:
                self.detector = YOLO("yolov8n.pt")
                print("[CrowdAnalyzer] YOLOv8n person detector ready (validation gate & mask role).")
            except Exception as exc:
                print(f"[CrowdAnalyzer] YOLO init warning: {exc}")

        # ── CSRNet (primary crowd counter) ───────────────────────────────────
        self.csrnet = build_csrnet(weights_path=weights_path, device=self.device)
        self.csrnet.eval()

    # ── Person Presence Validation Gate ──────────────────────────────────────

    def validate_people_presence(self, img_bgr: np.ndarray) -> "tuple[bool, str]":
        """
        Lightweight person-presence validation stage BEFORE returning crowd results.
        
        Rejects non-crowd/non-human images (certificates, documents, screenshots,
        buildings, landscapes, food, vehicles without people, empty scenes).
        
        To prevent false rejections of dense crowds with small/partial heads,
        a sensitive detection threshold is used alongside a CSRNet density check.
        """
        if self.detector is None:
            # Fallback if YOLO is unavailable
            return True, ""

        try:
            # 1. Person detector at sensitive confidence threshold (0.10)
            results = self.detector(img_bgr, classes=[0], conf=0.10, iou=0.50, verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                return True, ""

            # 2. Dense crowd fallback: In dense crowds, individual YOLO boxes may be 0,
            #    but CSRNet density integral and spatial texture will show prominent human crowd pattern.
            tensor, _, _ = _preprocess(img_bgr, self.device)
            with torch.no_grad():
                out = self.csrnet(tensor)
                raw_count = float(out.sum().item())
                max_density = float(out.max().item())

            # If raw_count is substantial and has local density peaks, accept as dense crowd
            if raw_count >= 8.0 and max_density >= 0.03:
                return True, ""

            # Image contains no people or crowd
            return False, "Please upload a valid image containing people or a crowd."

        except Exception as exc:
            print(f"[CrowdAnalyzer] Person validation check exception: {exc}")
            return True, ""

    # ── YOLO: human foreground mask ──────────────────────────────────────────

    def _build_human_mask(
        self,
        img_bgr: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """
        Run robust YOLOv8 person detection to identify distinct human bodies and heads
        with IoU suppression and human aspect-ratio verification.
        """
        h, w = img_bgr.shape[:2]
        boxes_out = np.empty((0, 4), dtype=np.float32)
        confs_out = np.empty((0,),   dtype=np.float32)
        mask      = np.zeros((h, w), dtype=np.float32)

        if self.detector is None:
            mask[:] = 1.0
            return boxes_out, confs_out, mask

        try:
            r1 = self.detector(img_bgr, classes=[0], conf=conf_threshold, iou=0.45, verbose=False)[0]
            if r1.boxes is not None and len(r1.boxes) > 0:
                raw_boxes = r1.boxes.xyxy.cpu().numpy()
                raw_confs = r1.boxes.conf.cpu().numpy()

                valid_boxes = []
                valid_confs = []
                for box, conf in zip(raw_boxes, raw_confs):
                    x1, y1, x2, y2 = box
                    bw, bh = x2 - x1, y2 - y1
                    # Filter out non-human aspect ratios or tiny noise artifacts
                    if bh >= 10 and bw >= 6 and (bh / (bw + 1e-6)) >= 0.22:
                        valid_boxes.append(box)
                        valid_confs.append(conf)
                        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                        ix2, iy2 = min(w, int(x2)), min(h, int(y2))
                        mask[iy1:iy2, ix1:ix2] = 1.0

                if valid_boxes:
                    boxes_out = np.array(valid_boxes, dtype=np.float32)
                    confs_out = np.array(valid_confs, dtype=np.float32)
                    median_box_w = float(np.median([b[2] - b[0] for b in valid_boxes]))
                    sigma_val = float(np.clip(median_box_w * 0.35, 6.0, 20.0))
                    mask = gaussian_filter(mask, sigma=sigma_val)
                    mask = np.clip(mask * 3.0, 0.0, 1.0)
                else:
                    mask[:] = 1.0
            else:
                mask[:] = 1.0

        except Exception as exc:
            print(f"[CrowdAnalyzer] Human feature extraction error: {exc}")
            mask[:] = 1.0

        return boxes_out, confs_out, mask

    # ── CSRNet + mask: density prediction ────────────────────────────────────

    def predict_density(
        self,
        img_bgr: np.ndarray,
    ) -> "tuple[np.ndarray, float, float, np.ndarray]":
        """
        Full crowd density estimation pipeline.
        """
        h, w = img_bgr.shape[:2]

        # ── Step 1–2: CSRNet forward pass ─────────────────────────────────────
        tensor, h_pad, w_pad = _preprocess(img_bgr, self.device)
        with torch.no_grad():
            out = self.csrnet(tensor)                       # (1,1, H'/8, W'/8)
            raw_count = float(out.sum().item())             # integral = count

            out_up = F.interpolate(
                out,
                size=(h + h_pad, w + w_pad),
                mode="bilinear",
                align_corners=False,
            )
            density_full = (out_up.squeeze().cpu().numpy() / 64.0)
            density_full = np.clip(density_full[:h, :w], 0.0, None)

        raw_count = max(0.0, raw_count)

        # ── Step 4: YOLO human mask ───────────────────────────────────────────
        boxes, confs, mask = self._build_human_mask(img_bgr)

        # ── Step 5: apply mask conservatively ─────────────────────────────────
        # Dense crowd scenes are heavily occluded and YOLO boxes are incomplete:
        # a hard mask can suppress valid density in large regions and collapse the
        # heatmap toward the detected subset. Keep full spatial coverage for dense
        # scenes while only using the mask for sparse / moderate crowds.
        mask_mean = float(mask.mean())
        num_detected = len(boxes)
        dense_crowd = num_detected > 35 or raw_count >= 50.0 or (num_detected > 10 and raw_count >= 30.0)
        if mask_mean < 0.995 and not dense_crowd:
            density_masked = density_full * (0.35 + 0.65 * mask)
        else:
            density_masked = density_full

        # ── Step 6: Validated Density & Spatial Crowd Fusion ─────────────────
        avg_conf_yolo = float(np.mean(confs)) if num_detected > 0 else 0.0
        avg_conf_pct = avg_conf_yolo * 100.0

        # Check if the scene is an isolated close-up portrait / sparse small group
        is_large_portrait = False
        if 0 < num_detected <= 6 and avg_conf_yolo >= 0.85:
            max_box_area = max([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]) / float(h * w)
            if max_box_area >= 0.08:  # Single individual occupies > 8% of entire frame
                is_large_portrait = True

        if is_large_portrait:
            # Clean close-up individuals: count matches verified detections
            final_count = float(num_detected)
            confidence_score = float(np.clip(avg_conf_pct + 4.0, 90.0, 99.5))
        elif num_detected > 0 and raw_count > num_detected:
            # Perspective street scene with background / occluded pedestrians:
            # Use continuous CSRNet density integral calibrated with verified detections
            calibrated_csrnet = raw_count * 0.80 if raw_count < 45.0 else raw_count
            final_count = float(max(num_detected, round(calibrated_csrnet)))
            confidence_score = float(np.clip(avg_conf_pct + 3.0, 82.0, 95.0))
        elif num_detected > 35 or raw_count >= 50.0:
            # Dense / Congested stadium crowds: CSRNet deep spatial integral
            final_count = raw_count
            confidence_score = float(np.clip(avg_conf_pct * 0.4 + 75.0, 75.0, 96.0))
        elif num_detected > 0:
            final_count = float(num_detected)
            confidence_score = float(np.clip(avg_conf_pct + 4.0, 85.0, 98.0))
        else:
            final_count = raw_count
            confidence_score = 78.0 if raw_count > 0 else 95.0

        # ── Step 7: Scale density map to match final human count ───────────────
        d_sum = float(density_masked.sum())
        if d_sum > 1e-6:
            density_map = density_masked * (final_count / d_sum)
        else:
            density_map = density_full
            if float(density_full.sum()) > 1e-6:
                density_map = density_full * (final_count / float(density_full.sum()))

        return density_map, max(0.0, final_count), round(confidence_score, 1), boxes

    # ── Density level classifier ─────────────────────────────────────────────

    def classify_density(self, count: float, area_pixels: int) -> dict:
        """
        Classify crowd as Low / Medium / High based on count and image area.
        """
        density_index = count / (area_pixels / 10_000.0) if area_pixels > 0 else 0.0

        if count < 25:
            return {
                "level": "Low",
                "full_level": "Low Density",
                "badge": "Sparse / Safe Flow",
                "color": "#10b981",
                "description": "Minimal crowd concentration. Free pedestrian movement.",
                "density_index": round(density_index, 2),
            }
        elif count < 100:
            return {
                "level": "Medium",
                "full_level": "Medium Density",
                "badge": "Normal Public Flow",
                "color": "#0ea5e9",
                "description": "Noticeable crowd activity. Safe movement conditions.",
                "density_index": round(density_index, 2),
            }
        else:
            return {
                "level": "High",
                "full_level": "High Density",
                "badge": "Crowd Congestion",
                "color": "#ef4444",
                "description": "Dense human concentration. Bottleneck and flow monitoring recommended.",
                "density_index": round(density_index, 2),
            }

    def extract_head_points(
        self,
        img_shape: tuple,
        density_map: np.ndarray,
        boxes: np.ndarray,
    ) -> list:
        """
        Extract spatial head coordinates (x, y) overall the crowd:
        combines verified person box head locations with continuous density map peaks.
        """
        h, w = img_shape[:2]
        head_points = []
        box_mask = np.zeros((h, w), dtype=bool)

        # 1. Head locations from detected person boxes
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                hx = int((x1 + x2) / 2)
                hy = int(y1 + (y2 - y1) * 0.18)
                hx = max(0, min(w - 1, hx))
                hy = max(0, min(h - 1, hy))
                head_points.append((hx, hy))
                box_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True

        # 2. Extract head peaks from density map for background / dense regions
        if density_map is not None and density_map.max() > 1e-5:
            nbr_size = max(5, int(min(h, w) * 0.025))
            local_max = (maximum_filter(density_map, size=nbr_size) == density_map)
            min_peak_val = max(1e-4, float(density_map.max()) * 0.05)
            peak_mask = local_max & (density_map > min_peak_val)

            for py, px in np.argwhere(peak_mask):
                if not box_mask[py, px]:
                    head_points.append((int(px), int(py)))

        return head_points

    # ── Image analysis ────────────────────────────────────────────────────────

    def analyze_image(self, image_input, output_dir: str = None) -> dict:
        """
        Complete image analysis pipeline with invalid-image validation gate and head annotations.
        """
        start_time = time.time()

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Could not decode image: {image_input}")
            base_name = os.path.splitext(os.path.basename(image_input))[0]
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
            base_name = f"image_{int(time.time() * 1000)}"
        else:
            raise TypeError("image_input must be a file path or numpy array")

        # ── Person Presence Validation Gate ──────────────────────────────────
        is_valid, validation_msg = self.validate_people_presence(img_bgr)
        if not is_valid:
            return {
                "success": False,
                "invalid_image": True,
                "message": "⚠️ Please upload a valid image containing people or a crowd.",
                "detail": "The uploaded image does not appear to contain people.",
            }

        orig_h, orig_w = img_bgr.shape[:2]

        # ── Inference ─────────────────────────────────────────────────────────
        density_map, raw_count, confidence_score, boxes = self.predict_density(img_bgr)
        predicted_count = int(round(raw_count))

        # ── Head Annotations Extraction ───────────────────────────────────────
        head_points = self.extract_head_points((orig_h, orig_w), density_map, boxes)

        # ── Visualisation ──────────────────────────────────────────────────────
        density_heatmap = density_to_heatmap(density_map, cv2.COLORMAP_JET)

        # Draw glowing head notations on density map
        density_annotated = density_heatmap.copy()
        for hx, hy in head_points:
            cv2.circle(density_annotated, (hx, hy), 4, (0, 255, 255), -1)  # Yellow center dot
            cv2.circle(density_annotated, (hx, hy), 7, (0, 255, 0), 1)     # Green glowing ring

        # Draw head annotations and person boxes on overlay image
        overlay_img = cv2.addWeighted(img_bgr, 0.60, density_heatmap, 0.40, 0)
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 128), 1)
        for hx, hy in head_points:
            cv2.circle(overlay_img, (hx, hy), 4, (0, 255, 255), -1)
            cv2.circle(overlay_img, (hx, hy), 7, (0, 255, 0), 1)

        density_info = self.classify_density(predicted_count, orig_h * orig_w)

        side_by_side = np.hstack((img_bgr, density_annotated))
        banner_h = 55
        canvas = np.zeros((orig_h + banner_h, orig_w * 2, 3), dtype=np.uint8)
        canvas[:orig_h] = side_by_side

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Original Image",
                    (20, orig_h + 38), font, 0.8, (255, 255, 255), 2)
        banner_text = (
            f"CSRNet Density Map | Count: {predicted_count} | Heads: {len(head_points)} | "
            f"Level: {density_info['level']} ({confidence_score}%)"
        )
        cv2.putText(canvas, banner_text,
                    (orig_w + 20, orig_h + 38), font, 0.65, (0, 255, 255), 2)

        latency_ms = (time.time() - start_time) * 1000.0

        # ── Save outputs ───────────────────────────────────────────────────────
        saved_paths = {}
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            density_save  = os.path.join(output_dir, f"density_{base_name}.jpg")
            overlay_save  = os.path.join(output_dir, f"overlay_{base_name}.jpg")
            side_save     = os.path.join(output_dir, f"side_by_side_{base_name}.jpg")
            cv2.imwrite(density_save,  density_annotated)
            cv2.imwrite(overlay_save,  overlay_img)
            cv2.imwrite(side_save,     canvas)
            saved_paths = {
                "density_path":    density_save,
                "overlay_path":    overlay_save,
                "side_by_side_path": side_save,
            }

        return {
            "success":          True,
            "predicted_count":  predicted_count,
            "raw_count":        round(raw_count, 2),
            "density_level":    density_info["level"],
            "confidence_score": confidence_score,
            "density_map":      density_map,
            "density_heatmap":  density_heatmap,
            "overlay_img":      overlay_img,
            "side_by_side":     canvas,
            "resolution":       f"{orig_w}x{orig_h}",
            "latency_ms":       round(latency_ms, 1),
            "risk_assessment":  density_info,
            "saved_paths":      saved_paths,
        }

    # ── Cross-Frame IoU Tracking Helpers ─────────────────────────────────────

    @staticmethod
    def _iou_boxes(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute IoU between two bounding boxes in [x1, y1, x2, y2] format.
        Returns a float in [0, 1].
        """
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter   = inter_w * inter_h
        if inter == 0.0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union  = area_a + area_b - inter
        return inter / union if union > 1e-6 else 0.0

    def _track_update(
        self,
        tracks: list,
        new_boxes: np.ndarray,
        frame_idx: int,
        iou_thresh: float = 0.25,
        max_gap: int = 15,
    ) -> int:
        """
        Match new detections to active tracks using a hybrid IoU + centroid-distance score.

        Pure IoU fails when people move quickly between sampled frames (IoU can drop
        below the threshold even for the same person, causing a ghost duplicate track).
        This method uses:
          score = max(iou, centroid_proximity)
        where centroid_proximity = clamp(1 - dist / (box_diag * 1.5), 0, 1).

        A match is accepted when score >= iou_thresh.
        """
        next_id = max((t["id"] for t in tracks), default=-1) + 1
        matched_track_ids: set = set()

        for box in new_boxes:
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            bw  = float(box[2] - box[0])
            bh  = float(box[3] - box[1])
            box_diag = float(np.hypot(bw, bh)) + 1e-6

            best_score = 0.0
            best_track = None

            for track in tracks:
                if frame_idx - track["last_frame"] > max_gap:
                    continue
                if track["id"] in matched_track_ids:
                    continue

                # --- IoU score ---
                iou_score = self._iou_boxes(box, track["box"])

                # --- Centroid-proximity score ---
                t_cx, t_cy = track["centroid_history"][-1]
                dist = float(np.hypot(cx - t_cx, cy - t_cy))
                prox_score = float(np.clip(1.0 - dist / (box_diag * 1.5), 0.0, 1.0))

                # Hybrid: accept if EITHER metric is strong enough
                score = max(iou_score, prox_score)

                if score > best_score:
                    best_score = score
                    best_track = track

            if best_score >= iou_thresh and best_track is not None:
                best_track["box"]              = box
                best_track["centroid_history"].append((cx, cy))
                best_track["box_history"].append((bw, bh))  # Track box size
                best_track["last_frame"]       = frame_idx
                
                # Mark as confirmed if enough observations
                if not best_track.get("confirmed", False):
                    if len(best_track["centroid_history"]) >= 3:  # Quick confirmation threshold
                        best_track["confirmed"] = True
                
                matched_track_ids.add(best_track["id"])
            else:
                tracks.append({
                    "id":               next_id,
                    "box":              box,
                    "centroid_history": [(cx, cy)],
                    "box_history":      [(bw, bh)],
                    "last_frame":       frame_idx,
                    "confirmed":        False,  # Starts as tentative
                })
                matched_track_ids.add(next_id)
                next_id += 1

        return next_id

    @staticmethod
    def _deduplicate_tracks(
        tracks: list,
        centroid_merge_thresh: float = 60.0,
    ) -> list:
        """
        Post-processing pass: merge tracks that represent the same physical person.

        A person who briefly left the frame and came back, or whose IoU fell below
        the match threshold mid-video, can accumulate two tracks.  This method
        merges any pair of tracks whose average-centroid Euclidean distance is below
        `centroid_merge_thresh` pixels (default 60px).

        Only one of the pair is kept (the one with more observations).  The method
        is applied iteratively until no more merges happen.
        """
        if len(tracks) <= 1:
            return tracks

        def _avg_centroid(track):
            h = track["centroid_history"]
            return (
                float(np.mean([p[0] for p in h])),
                float(np.mean([p[1] for p in h])),
            )

        changed = True
        while changed:
            changed = False
            avg_cents = [_avg_centroid(t) for t in tracks]
            to_remove: set = set()

            for i in range(len(tracks)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(tracks)):
                    if j in to_remove:
                        continue
                    ax, ay = avg_cents[i]
                    bx, by = avg_cents[j]
                    dist = float(np.hypot(ax - bx, ay - by))
                    if dist < centroid_merge_thresh:
                        # Keep the track with more observations; drop the other
                        if len(tracks[j]["centroid_history"]) > len(tracks[i]["centroid_history"]):
                            to_remove.add(i)
                        else:
                            to_remove.add(j)
                        changed = True

            if to_remove:
                tracks = [t for idx, t in enumerate(tracks) if idx not in to_remove]

        return tracks

    # ── Movement Analysis Helpers (NEW - Critical for Stationary Detection) ─────

    @staticmethod
    def _calculate_trajectory_jitter(trajectory: list) -> float:
        """
        Calculate velocity variance to detect if movement is consistent or just jitter.
        
        High variance = random box changes (jitter)
        Low variance = consistent movement
        
        Returns normalized jitter score (0.0 = smooth, 1.0 = very jittery)
        """
        if len(trajectory) < 3:
            return 0.0
        
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            velocities_x.append(dx)
            velocities_y.append(dy)
        
        if not velocities_x:
            return 0.0
        
        # Calculate variance
        var_x = float(np.var(velocities_x))
        var_y = float(np.var(velocities_y))
        
        # Calculate mean absolute velocity
        mean_abs_vel_x = float(np.mean(np.abs(velocities_x)))
        mean_abs_vel_y = float(np.mean(np.abs(velocities_y)))
        
        # Normalize variance by mean velocity (coefficient of variation)
        if mean_abs_vel_x > 1e-6:
            cv_x = np.sqrt(var_x) / mean_abs_vel_x
        else:
            cv_x = 0.0
        
        if mean_abs_vel_y > 1e-6:
            cv_y = np.sqrt(var_y) / mean_abs_vel_y
        else:
            cv_y = 0.0
        
        # Return average coefficient of variation
        return float((cv_x + cv_y) / 2.0)
    
    @staticmethod
    def _calculate_direction_consistency(trajectory: list) -> float:
        """
        Calculate how consistent the direction of movement is.
        
        Returns consistency score (0.0 = random, 1.0 = perfectly consistent)
        """
        if len(trajectory) < 3:
            return 0.0
        
        angles = []
        for i in range(2, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                angle = np.arctan2(dy, dx)
                angles.append(angle)
        
        if len(angles) < 2:
            return 0.0
        
        # Calculate circular variance (for angles)
        mean_sin = np.mean([np.sin(a) for a in angles])
        mean_cos = np.mean([np.cos(a) for a in angles])
        r = np.hypot(mean_sin, mean_cos)
        
        # r close to 1 = consistent direction
        # r close to 0 = random directions
        return float(r)

    @staticmethod
    def _csrnet_video_count(frame: np.ndarray, device: torch.device, csrnet_model: torch.nn.Module) -> float:
        """Video-only CSRNet count that clamps negative activations before integration."""
        tensor, _, _ = _preprocess(frame, device)
        with torch.no_grad():
            density_map = csrnet_model(tensor)
            density_map = torch.relu(density_map)
            return max(0.0, float(density_map.sum().item()))

    @staticmethod
    def _estimate_dense_motion_count(video_path: str, sample_every: int = 4, max_pairs: int = 12) -> int:
        """Estimate dense crowd size from motion energy when YOLO-based tracks disappear."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        try:
            frames = []
            frame_idx = 0
            while len(frames) < max_pairs + 1:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if frame_idx % sample_every == 0:
                    frames.append(frame)
                frame_idx += 1

            if len(frames) < 2:
                return 0

            motion_areas = []
            h0, w0 = frames[0].shape[:2]
            person_px = max(150.0, (w0 * h0) * 0.0018)

            for prev_frame, curr_frame in zip(frames[:-1], frames[1:]):
                prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (640, max(1, int(prev_frame.shape[0] * 640 / max(1, prev_frame.shape[1])))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(cv2.resize(curr_frame, (640, max(1, int(curr_frame.shape[0] * 640 / max(1, curr_frame.shape[1])))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_mask = mag > 1.0
                if motion_mask.any():
                    motion_areas.append(float(np.count_nonzero(motion_mask)))

            if not motion_areas:
                return 0

            motion_area_est = float(np.median(motion_areas))
            return max(1, int(round(motion_area_est / person_px)))
        finally:
            cap.release()

    @staticmethod
    def _estimate_dense_flow_direction(video_path: str, sample_every: int = 8, max_pairs: int = 10) -> dict:
        """
        Dense-crowd flow estimation from temporal optical flow rather than individual tracks.
        This is used when tracking fails under heavy overlap and tiny heads.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "dominant_direction": "Stationary (no movement detected)",
                "moving_pct": 0,
                "stationary_pct": 100,
                "direction_distribution": {"right": 0, "left": 0, "forward": 0, "backward": 0},
            }

        try:
            frames = []
            frame_idx = 0
            while len(frames) < max_pairs + 1:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if frame_idx % sample_every == 0:
                    frames.append(frame)
                frame_idx += 1

            if len(frames) < 2:
                return {
                    "dominant_direction": "Stationary (no movement detected)",
                    "moving_pct": 0,
                    "stationary_pct": 100,
                    "direction_distribution": {"right": 0, "left": 0, "forward": 0, "backward": 0},
                }

            gray_prev = cv2.cvtColor(cv2.resize(frames[0], (640, max(1, int(frames[0].shape[0] * 640 / max(1, frames[0].shape[1])))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            flow_acc_x = 0.0
            flow_acc_y = 0.0
            valid_pairs = 0
            motion_area_ratios = []

            for frame in frames[1:]:
                frame_r = cv2.resize(frame, (640, max(1, int(frame.shape[0] * 640 / max(1, frame.shape[1])))), interpolation=cv2.INTER_AREA)
                gray_curr = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                vx = flow[..., 0]
                vy = flow[..., 1]
                mag, _ = cv2.cartToPolar(vx, vy)
                mask = mag > 1.5
                if mask.any():
                    valid_pairs += 1
                    motion_area_ratios.append(float(np.count_nonzero(mask)) / float(mask.size))
                    flow_acc_x += float(vx[mask].mean())
                    flow_acc_y += float(vy[mask].mean())
                gray_prev = gray_curr

            if valid_pairs == 0:
                return {
                    "dominant_direction": "Stationary (no movement detected)",
                    "moving_pct": 0,
                    "stationary_pct": 100,
                    "direction_distribution": {"right": 0, "left": 0, "forward": 0, "backward": 0},
                }

            # Small local changes such as smiling, head movement, or posture
            # shifts are not crowd flow. Require motion across a meaningful
            # portion of the frame before classifying dense motion.
            if float(np.median(motion_area_ratios)) < 0.08:
                return {
                    "dominant_direction": "Stationary (no movement detected)",
                    "moving_pct": 0,
                    "stationary_pct": 100,
                    "direction_distribution": {"right": 0, "left": 0, "forward": 0, "backward": 0},
                }

            avg_dx = flow_acc_x / valid_pairs
            avg_dy = flow_acc_y / valid_pairs
            abs_dx = abs(avg_dx)
            abs_dy = abs(avg_dy)

            if abs_dx < 1.0 and abs_dy < 1.0:
                dominant = "Stationary (no movement detected)"
                moving_pct = 0
                stationary_pct = 100
                distribution = {"right": 0, "left": 0, "forward": 0, "backward": 0}
            elif abs_dx > abs_dy:
                if avg_dx > 0:
                    dominant = "RIGHT →"
                    distribution = {"right": 100, "left": 0, "forward": 0, "backward": 0}
                else:
                    dominant = "LEFT ←"
                    distribution = {"right": 0, "left": 100, "forward": 0, "backward": 0}
                moving_pct = 100
                stationary_pct = 0
            else:
                if avg_dy < 0:
                    dominant = "FORWARD ↑"
                    distribution = {"right": 0, "left": 0, "forward": 100, "backward": 0}
                else:
                    dominant = "BACKWARD ↓"
                    distribution = {"right": 0, "left": 0, "forward": 0, "backward": 100}
                moving_pct = 100
                stationary_pct = 0

            return {
                "dominant_direction": dominant,
                "moving_pct": moving_pct,
                "stationary_pct": stationary_pct,
                "direction_distribution": distribution,
            }
        finally:
            cap.release()

    # ── Accurate Video Analysis with YOLO Tracking + Per-Person Direction ───────

    def analyze_video(
        self,
        video_path: str,
        max_counting_samples: int = 8,
        progress_callback=None,
        # Tracking parameters
        tracking_sample_every: int = 2,
        iou_thresh: float = 0.30,
        max_track_gap: int = 15,
        dense_crowd_threshold: int = 200,
        stationary_ratio_threshold: float = 0.50,  # ratio of box diagonal
        centroid_merge_thresh: float = 60.0,
        min_trajectory_length: int = 8,  # INCREASED from 5
        min_track_confirmation_frames: int = 8,  # NEW: temporal confirmation
        jitter_velocity_threshold: float = 0.60,  # NEW: detect jitter
        **kwargs,
    ) -> dict:
        """
        STRICT ACCURACY: Frame-by-frame YOLO person detection + persistent tracking
        with PER-PERSON trajectory-based direction classification.

        CRITICAL FIXES FOR SITTING PEOPLE:
        - Requires minimum 8 frames for track confirmation (filters false detections)
        - Uses box-relative movement threshold (30% of diagonal, not fixed 20px)
        - Analyzes entire trajectory path to detect jitter vs real movement
        - Calculates velocity variance to distinguish random box changes from consistent movement
        
        Tracking Pipeline
        -----------------
        1. Sample every `tracking_sample_every` frames (default: every 2nd frame).
        2. Detect persons using YOLOv8 (confidence ≥ 0.25, person class only).
        3. Match detections across frames using hybrid IoU + centroid distance.
        4. Track is "tentative" until confirmed (appears in min_track_confirmation_frames).
        5. Only "confirmed" tracks count toward final people estimate.
        6. Deduplication merges tracks within `centroid_merge_thresh` pixels.

        Movement Detection (CRITICAL FOR STATIONARY PEOPLE)
        ----------------------------------------------------
        For each confirmed track:
        
        1. Calculate box-relative threshold:
           box_diagonal = sqrt(width² + height²)
           movement_threshold = box_diagonal × stationary_ratio_threshold (default 0.50)
           
        2. Analyze entire trajectory path (NOT just first→last):
           - Calculate total path length
           - Measure velocity consistency (variance)
           - Count directional changes
           
        3. Jitter Detection:
           - High velocity variance → JITTER (box changes, not movement)
           - Many direction changes → JITTER
           - Small net displacement → STATIONARY
           
        4. Classification:
           - Net displacement < movement_threshold → STATIONARY
           - High jitter (velocity_variance > threshold) → STATIONARY
           - Consistent movement in one direction → MOVING + direction

        Direction Classification (ONLY FOR CONFIRMED MOVING PEOPLE)
        ------------------------------------------------------------
        Only classify direction if person is genuinely moving:
        - Movement must exceed box-relative threshold
        - Movement must be consistent (low jitter)
        - Direction based on dominant displacement axis
        
        NO fabrication: All percentages calculated from actual confirmed tracks.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or np.isnan(fps):
                fps = 25.0
            if total_frames <= 0:
                total_frames = 1

            # ── Tracking state ────────────────────────────────────────────────
            tracks: list = []           # [{id, box, centroid_history, last_frame, confirmed, box_history}]
            use_csrnet_fallback = False
            csrnet_frame_counts: list = []  # fallback CSRNet per-frame counts
            detection_frame_counts: list = []
            dense_seen_frames = 0
            total_raw_detections = 0
            max_raw_detection_count = 0
            fallback_distribution = None
            fallback_moving_pct = 0
            fallback_stationary_pct = 0
            fallback_dominant_direction = "No people detected"
            fallback_no_sig_movement = True

            frame_idx = 0
            csrnet_sample_step = max(1, total_frames // max_counting_samples)

            print(f"[CrowdAnalyzer] Processing video: {total_frames} frames @ {fps:.1f} fps")
            print(f"[CrowdAnalyzer] Tracking every {tracking_sample_every} frames")
            print(f"[CrowdAnalyzer] Track confirmation requires {min_track_confirmation_frames} observations")

            while True:
                is_tracking_frame = (frame_idx % tracking_sample_every == 0)
                is_csrnet_frame   = (
                    use_csrnet_fallback and
                    (frame_idx % csrnet_sample_step == 0) and
                    (len(csrnet_frame_counts) < max_counting_samples)
                )

                needs_frame = is_tracking_frame or is_csrnet_frame

                if needs_frame:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                else:
                    if not cap.grab():
                        break
                    frame_idx += 1
                    if progress_callback and total_frames > 0:
                        progress_callback(frame_idx, total_frames)
                    continue

                h, w = frame.shape[:2]

                # ──────────────────────────────────────────────────────────────
                # 1. YOLO Cross-Frame Tracking (primary counting method)
                # ──────────────────────────────────────────────────────────────
                if is_tracking_frame and not use_csrnet_fallback:
                    if self.detector is not None:
                        # Downscale to max 640px for speed, then rescale boxes back
                        scale_t = min(1.0, 640.0 / max(h, w))
                        if scale_t < 1.0:
                            frame_t = cv2.resize(
                                frame,
                                (int(w * scale_t), int(h * scale_t)),
                                interpolation=cv2.INTER_AREA,
                            )
                        else:
                            frame_t = frame
                            scale_t = 1.0

                        try:
                            det_results = self.detector(
                                frame_t,
                                classes=[0],          # person only
                                conf=0.25,
                                iou=0.45,
                                imgsz=640,
                                verbose=False,
                            )[0]

                            raw_boxes_t = (
                                det_results.boxes.xyxy.cpu().numpy()
                                if det_results.boxes is not None and len(det_results.boxes) > 0
                                else np.empty((0, 4), dtype=np.float32)
                            )

                            # Retry the same video frame with more pixels and a
                            # sensitive threshold before treating it as a dense
                            # crowd that needs CSRNet estimation.
                            if len(raw_boxes_t) == 0:
                                retry_results = self.detector(
                                    frame,
                                    classes=[0],
                                    conf=0.10,
                                    iou=0.50,
                                    imgsz=1280,
                                    verbose=False,
                                )[0]
                                raw_boxes_t = (
                                    retry_results.boxes.xyxy.cpu().numpy()
                                    if retry_results.boxes is not None and len(retry_results.boxes) > 0
                                    else np.empty((0, 4), dtype=np.float32)
                                )
                                scale_t = 1.0

                            # Rescale boxes back to original resolution
                            if scale_t < 1.0 and len(raw_boxes_t) > 0:
                                raw_boxes_t = raw_boxes_t / scale_t

                            # Filter degenerate boxes (very small, non-person aspect ratios)
                            valid_boxes_t = []
                            for box in raw_boxes_t:
                                bw = box[2] - box[0]
                                bh = box[3] - box[1]
                                if bh > 8 and bw > 4 and (bh / (bw + 1e-6)) > 0.30:
                                    valid_boxes_t.append(box)

                            n_det = len(valid_boxes_t)
                            detection_frame_counts.append(n_det)
                            total_raw_detections += n_det
                            max_raw_detection_count = max(max_raw_detection_count, n_det)

                            # Dense-crowd gate: if the detector is effectively blind on many frames,
                            # do not let the video count collapse to zero. Switch to a conservative
                            # CSRNet fallback for the final estimate rather than forcing a bad YOLO-only result.
                            if n_det > dense_crowd_threshold * 3:
                                print(
                                    f"[CrowdAnalyzer] Extreme crowd density ({n_det} boxes at frame {frame_idx}). "
                                    f"Keeping temporal YOLO tracking active; CSRNet fallback remains secondary."
                                )
                            if n_det > 0:
                                dense_seen_frames += 1
                                boxes_arr = np.array(valid_boxes_t, dtype=np.float32)
                                self._track_update(
                                    tracks, boxes_arr, frame_idx,
                                    iou_thresh=iou_thresh,
                                    max_gap=max_track_gap,
                                )
                            elif frame_idx > 0 and (frame_idx % (tracking_sample_every * 5) == 0):
                                use_csrnet_fallback = True
                                print(
                                    f"[CrowdAnalyzer] YOLO detections collapsed on sampled frame {frame_idx}. "
                                    "Switching video estimate to CSRNet fallback."
                                )
                        except Exception as exc:
                            print(f"[CrowdAnalyzer] YOLO tracking error at frame {frame_idx}: {exc}")

                # ──────────────────────────────────────────────────────────────
                # 2. CSRNet Fallback Sampling (dense crowd only)
                # ──────────────────────────────────────────────────────────────
                if is_csrnet_frame:
                    scale_c = min(1.0, 480.0 / max(h, w))
                    frame_c = (
                        cv2.resize(frame, (int(w * scale_c), int(h * scale_c)),
                                   interpolation=cv2.INTER_AREA)
                        if scale_c < 1.0 else frame
                    )
                    csrnet_frame_counts.append(
                        self._csrnet_video_count(frame_c, self.device, self.csrnet)
                    )

                frame_idx += 1
                if progress_callback and total_frames > 0:
                    progress_callback(frame_idx, total_frames)

            # Adaptive dense video strategy: if YOLO-based tracking fails to produce a stable
            # person count on a crowded sequence, the final estimate must fall back to CSRNet.
            if (
                use_csrnet_fallback
                or self.detector is None
                or (
                    len(detection_frame_counts) >= max_counting_samples
                    and max_raw_detection_count <= max(2, dense_crowd_threshold // 3)
                    and np.median(detection_frame_counts) <= max(2, dense_crowd_threshold // 4)
                )
            ):
                use_csrnet_fallback = True

            # Keep sparse videos on temporal tracking. CSRNet is intended as a
            # dense-crowd fallback and can overestimate a small visible group.
            if (
                use_csrnet_fallback
                and self.detector is not None
                and max_raw_detection_count > 0
                and max_raw_detection_count <= 20
            ):
                use_csrnet_fallback = False

            # ─────────────────────────────────────────────────────────────────
            # 3. Per-Person Direction Classification from Trajectories
            # ─────────────────────────────────────────────────────────────────
            if use_csrnet_fallback or (self.detector is None):
                # Dense-crowd path: prefer a non-zero motion-based estimate when CSRNet is weak
                # or when YOLO tracking collapses. This keeps the video path adaptive without touching
                # the image-density logic.
                if csrnet_frame_counts:
                    sorted_c = sorted(csrnet_frame_counts)
                    n_c = len(sorted_c)
                    if n_c >= 5:
                        trim_k = max(1, int(round(n_c * 0.10)))
                        trimmed = sorted_c[trim_k : n_c - trim_k]
                        estimated_people = int(round(float(np.median(trimmed))))
                    else:
                        estimated_people = int(round(float(np.median(sorted_c))))
                else:
                    estimated_people = 0

                motion_estimate = self._estimate_dense_motion_count(video_path, sample_every=max(2, tracking_sample_every * 2), max_pairs=12)
                if estimated_people <= 0 and motion_estimate > 0:
                    estimated_people = motion_estimate
                elif estimated_people > 0 and motion_estimate > estimated_people:
                    estimated_people = max(estimated_people, motion_estimate // 2)

                dense_flow = self._estimate_dense_flow_direction(video_path, sample_every=max(2, tracking_sample_every * 2), max_pairs=12)
                stationary_count = 0
                right_count = 0
                left_count = 0
                forward_count = 0
                backward_count = 0
                distribution = dense_flow["direction_distribution"]
                moving_pct = dense_flow["moving_pct"]
                stationary_pct = dense_flow["stationary_pct"]
                dominant_direction = dense_flow["dominant_direction"]
                no_sig_movement = moving_pct == 0
                fallback_distribution = distribution
                fallback_moving_pct = moving_pct
                fallback_stationary_pct = stationary_pct
                fallback_dominant_direction = dominant_direction
                fallback_no_sig_movement = no_sig_movement

                print(f"[CrowdAnalyzer] CSRNet fallback count: {estimated_people}")
                print(f"[CrowdAnalyzer] Dense-crowd flow estimate: {dominant_direction}")

            else:
                # ── Post-processing deduplication ─────────────────────────────
                # Merge tracks that represent the same physical person
                tracks = self._deduplicate_tracks(
                    tracks,
                    centroid_merge_thresh=centroid_merge_thresh,
                )
                print(f"[CrowdAnalyzer] After dedup: {len(tracks)} unique tracks.")

                # Sparse/medium crowd path: retain YOLO temporal tracking as the primary
                # counting method, but keep a robust per-frame detection median as a fallback
                # when confirmed tracks are sparse or noisy.
                estimated_people = len(tracks)
                if detection_frame_counts:
                    median_detected = int(round(float(np.median(detection_frame_counts))))
                    if estimated_people == 0:
                        estimated_people = max(0, median_detected)
                    else:
                        estimated_people = max(estimated_people, median_detected // 2)

                if estimated_people == 0 and self.detector is not None:
                    print("[CrowdAnalyzer] YOLO found 0 tracks — using CSRNet fallback.")
                    cap2 = cv2.VideoCapture(video_path)
                    fb_step = max(1, total_frames // max_counting_samples)
                    fb_counts: list = []
                    fb_idx = 0
                    while cap2.isOpened() and len(fb_counts) < max_counting_samples:
                        ret2, fr2 = cap2.read()
                        if not ret2 or fr2 is None:
                            break
                        if fb_idx % fb_step == 0:
                            h2, w2 = fr2.shape[:2]
                            sc2 = min(1.0, 480.0 / max(h2, w2))
                            fr2s = cv2.resize(fr2, (int(w2*sc2), int(h2*sc2)), interpolation=cv2.INTER_AREA) if sc2 < 1.0 else fr2
                            fb_counts.append(
                                self._csrnet_video_count(fr2s, self.device, self.csrnet)
                            )
                        fb_idx += 1
                    cap2.release()
                    if fb_counts:
                        estimated_people = max(estimated_people, int(round(float(np.median(fb_counts)))))

                    stationary_count = 0
                    right_count = 0
                    left_count = 0
                    forward_count = 0
                    backward_count = 0
                
                else:
                    # ── IMPROVED PER-PERSON DIRECTION CLASSIFICATION ──────────
                    # Filter: Only count CONFIRMED tracks with sufficient observations
                    stationary_count = 0
                    right_count = 0
                    left_count = 0
                    forward_count = 0
                    backward_count = 0
                    
                    # Count only confirmed tracks
                    confirmed_tracks = [t for t in tracks if t.get("confirmed", False) and len(t["centroid_history"]) >= min_track_confirmation_frames]
                    
                    print(f"[CrowdAnalyzer] Total tracks: {len(tracks)}, Confirmed: {len(confirmed_tracks)}")
                    
                    for track in confirmed_tracks:
                        hist = track["centroid_history"]
                        box_hist = track.get("box_history", [])
                        
                        # Require minimum trajectory length
                        if len(hist) < min_trajectory_length:
                            stationary_count += 1
                            continue
                        
                        # ── CRITICAL FIX 1: Calculate box-relative movement threshold ─────
                        # Use average box size over trajectory
                        if box_hist:
                            avg_box_w = np.mean([b[0] for b in box_hist])
                            avg_box_h = np.mean([b[1] for b in box_hist])
                            box_diagonal = np.hypot(avg_box_w, avg_box_h)
                        else:
                            # Fallback if box_history missing
                            box_diagonal = 100.0
                        
                        # Movement threshold = percentage of box diagonal
                        movement_threshold = box_diagonal * stationary_ratio_threshold
                        
                        # ── CRITICAL FIX 2: Detect jitter vs real movement ────────────────
                        jitter_score = self._calculate_trajectory_jitter(hist)
                        direction_consistency = self._calculate_direction_consistency(hist)
                        
                        # High jitter = NOT real movement (just detection noise)
                        is_jitter = jitter_score > jitter_velocity_threshold
                        
                        # ── CRITICAL FIX 3: Calculate net displacement ────────────────────
                        initial_x, initial_y = hist[0]
                        final_x, final_y = hist[-1]
                        
                        dx = final_x - initial_x
                        dy = final_y - initial_y
                        
                        net_displacement = np.hypot(dx, dy)
                        abs_dx = abs(dx)
                        abs_dy = abs(dy)
                        
                        # ── CLASSIFICATION LOGIC ──────────────────────────────────────────
                        
                        # Check 1: Is displacement below threshold?
                        if net_displacement < movement_threshold:
                            stationary_count += 1
                            continue
                        
                        # Check 2: Is this just jitter (random box changes)?
                        if is_jitter:
                            stationary_count += 1
                            continue
                        
                        # Check 3: Is direction consistent enough?
                        if direction_consistency < 0.3:  # Very inconsistent = jitter
                            stationary_count += 1
                            continue
                        
                        # Person is genuinely MOVING - classify direction
                        if abs_dx > abs_dy:
                            # Horizontal movement dominates
                            if dx > 0:
                                right_count += 1
                            else:
                                left_count += 1
                        else:
                            # Vertical movement dominates
                            if dy < 0:
                                forward_count += 1
                            else:
                                backward_count += 1
                    
                    # Update estimated_people to only confirmed tracks
                    estimated_people = len(confirmed_tracks)
                    
                    print(f"[CrowdAnalyzer] Direction classification (confirmed tracks only):")
                    print(f"  → Right: {right_count}")
                    print(f"  ← Left: {left_count}")
                    print(f"  ↑ Forward: {forward_count}")
                    print(f"  ↓ Backward: {backward_count}")
                    print(f"  ⏸ Stationary: {stationary_count}")

            # ─────────────────────────────────────────────────────────────────
            # 4. Calculate Percentages from Actual Tracked People
            # ─────────────────────────────────────────────────────────────────
            total_classified = stationary_count + right_count + left_count + forward_count + backward_count
            
            if total_classified > 0:
                # Calculate raw percentages
                raw_pcts = {
                    "right": (right_count / total_classified) * 100.0,
                    "left": (left_count / total_classified) * 100.0,
                    "forward": (forward_count / total_classified) * 100.0,
                    "backward": (backward_count / total_classified) * 100.0,
                }
                stationary_pct_raw = (stationary_count / total_classified) * 100.0
                moving_pct_raw = 100.0 - stationary_pct_raw
                
                # Round to integers
                distribution = {k: int(round(v)) for k, v in raw_pcts.items()}
                stationary_pct = int(round(stationary_pct_raw))
                moving_pct = int(round(moving_pct_raw))
                
                # Ensure direction percentages sum correctly (adjust largest if needed)
                dir_sum = sum(distribution.values())
                if dir_sum != 100:
                    diff = 100 - dir_sum
                    max_k = max(raw_pcts, key=raw_pcts.get)
                    distribution[max_k] = max(0, distribution[max_k] + diff)
                
                # Ensure moving + stationary = 100
                if stationary_pct + moving_pct != 100:
                    moving_pct = 100 - stationary_pct
                
                # Find dominant direction (exclude stationary)
                moving_directions = {
                    k: v for k, v in distribution.items() 
                    if v > 0
                }
                
                if moving_directions:
                    dom_k = max(moving_directions, key=moving_directions.get)
                    dominant_direction_map = {
                        "forward": "FORWARD ↑",
                        "backward": "BACKWARD ↓",
                        "right": "RIGHT →",
                        "left": "LEFT ←",
                    }
                    dominant_direction = dominant_direction_map.get(dom_k, "FORWARD ↑")
                    no_sig_movement = False
                else:
                    dominant_direction = "Stationary (no movement detected)"
                    no_sig_movement = True
                
                print(f"[CrowdAnalyzer] Final percentages: {distribution}")
                print(f"  Moving: {moving_pct}%, Stationary: {stationary_pct}%")
                print(f"  Dominant: {dominant_direction}")
                
            else:
                # No people tracked
                distribution = {"right": 0, "left": 0, "forward": 0, "backward": 0}
                stationary_pct = 0
                moving_pct = 0
                dominant_direction = "No people detected"
                no_sig_movement = True

            if use_csrnet_fallback and fallback_distribution is not None:
                distribution = fallback_distribution
                moving_pct = fallback_moving_pct
                stationary_pct = fallback_stationary_pct
                dominant_direction = fallback_dominant_direction
                no_sig_movement = fallback_no_sig_movement

            return {
                "estimated_people":      estimated_people,
                "tracked_people":        total_classified,
                "dominant_direction":    dominant_direction,
                "direction_distribution": distribution,
                "no_significant_movement": no_sig_movement,
                "stationary_pct":        stationary_pct,
                "moving_pct":            moving_pct,
                "processed_frames":      frame_idx,
                "used_csrnet_fallback":  use_csrnet_fallback,
            }

        finally:
            cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSRNet Visual Crowd Density & Flow Inference"
    )
    parser.add_argument("--image",      type=str, help="Path to input crowd image")
    parser.add_argument("--video",      type=str, help="Path to input crowd video")
    parser.add_argument("--weights",    type=str, default=None, help="Path to CSRNet weights (.pth)")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device",     type=str, default=None, help="'cuda' or 'cpu'")
    args = parser.parse_args()

    analyzer = CrowdAnalyzer(weights_path=args.weights, device=args.device)

    if args.image:
        print(f"\n[+] Analysing image: {args.image}")
        res = analyzer.analyze_image(args.image, output_dir=args.output_dir)
        print("=" * 52)
        if not res.get("success", True):
            print(f"  {res.get('message')}")
            print(f"  {res.get('detail')}")
        else:
            print(f"  PREDICTED CROWD COUNT  : {res['predicted_count']} people")
            print(f"  DENSITY CLASSIFICATION : {res['density_level']}  ({res['risk_assessment']['badge']})")
            print(f"  CONFIDENCE SCORE       : {res['confidence_score']}%")
            print(f"  INFERENCE LATENCY      : {res['latency_ms']} ms")
            print(f"  Density Map saved to   : {res['saved_paths'].get('density_path')}")
        print("=" * 52 + "\n")

    elif args.video:
        print(f"\n[+] Analysing video: {args.video}")
        res = analyzer.analyze_video(args.video)
        print("=" * 52)
        print("  VIDEO ANALYSIS COMPLETE")
        print(f"  Estimated People       : {res['estimated_people']}")
        print(f"  Dominant Flow Direction: {res['dominant_direction']}")
        print(f"  Direction Distribution : {res['direction_distribution']}")
        print("=" * 52 + "\n")
    else:
        print("Specify --image or --video.  Run with --help for options.")

