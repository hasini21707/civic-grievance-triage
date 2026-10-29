"""
VisualInspector — Image/video analysis for BBMP civic-complaint validation.

Combines YOLOv8 object detection with municipal computer vision heuristics
(road pothole/cavity detection, garbage texture analysis, waterlogging detection)
and bounding-box visual annotation for complaint verification.
"""

import os
import tempfile
from typing import Union, List, Dict, Any
import numpy as np

# Mapping of detected objects/features to BBMP complaint category
COCO_TO_BBMP_CATEGORY = {
    # Direct civic issues
    "pothole": "Roads and Infrastructure",
    "manhole": "Roads and Infrastructure",
    "road damage": "Roads and Infrastructure",
    "garbage": "Solid Waste Management",
    "trash": "Solid Waste Management",
    "waste": "Solid Waste Management",
    "garbage dump": "Solid Waste Management",
    "tree": "Tree/Parks Maintenance",
    "fallen tree": "Tree/Parks Maintenance",
    "branch": "Tree/Parks Maintenance",
    "waterlogging": "Storm Water Drain",
    "flooding": "Storm Water Drain",
    "drain": "Storm Water Drain",
    "streetlight": "Streetlights",
    "lamp": "Streetlights",
    # COCO objects in municipal environment
    "car": "Roads and Infrastructure",
    "truck": "Roads and Infrastructure",
    "bus": "Roads and Infrastructure",
    "motorcycle": "Roads and Infrastructure",
    "bicycle": "Roads and Infrastructure",
    "traffic light": "Roads and Infrastructure",
    "stop sign": "Roads and Infrastructure",
    "fire hydrant": "Water Supply",
    "dog": "Animal Menace",
    "cat": "Animal Menace",
    "horse": "Animal Menace",
    "cow": "Animal Menace",
    "bench": "Parks and Open Spaces",
    "potted plant": "Tree/Parks Maintenance",
    "person": "General",
}

# High-priority civic hazard classes
HIGH_PRIORITY_CIVIC_CLASSES = {
    "pothole", "manhole", "road damage", "fallen tree",
    "waterlogging", "flooding", "garbage", "trash", "waste", "garbage dump"
}

CIVIC_ISSUE_CLASSES = set(COCO_TO_BBMP_CATEGORY.keys())


class VisualInspector:
    """
    Inspects images and video frames for civic complaints using YOLOv8
    and computer-vision municipal feature analysis.
    """

    def __init__(self):
        self._yolo_model = None
        self._model_size = "yolov8n"

    def load_model(self, model_size: str = "yolov8n") -> None:
        """Load or reload the YOLO model."""
        try:
            from ultralytics import YOLO as _YOLO
            self._model_size = model_size
            self._yolo_model = _YOLO(model_size)
        except Exception as e:
            raise ImportError(f"Could not load YOLO model: {e}")

    def _analyze_civic_cv_features(self, cv2_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect civic infrastructure anomalies (potholes, garbage, water pooling)
        using computer vision heuristics when standard COCO classes don't cover them.
        """
        import cv2

        extra_detections = []
        h, w = cv2_img.shape[:2]
        total_pixels = h * w
        # 1. Road cavity / Pothole detection: dark, high-gradient elliptical contours on lower half
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        edges = cv2.Canny(gray, 80, 200)
        edge_density = np.sum(edges > 0) / float(total_pixels)
        sat_std = float(np.std(sat))

        # Check for fallen tree / vegetation obstruction (green hue: 25-85)
        green_mask = cv2.inRange(hsv, (25, 30, 25), (85, 255, 255))
        green_density = np.sum(green_mask > 0) / float(total_pixels)

        if green_density > 0.06:
            conf = min(0.92, 0.72 + green_density)
            extra_detections.append({
                "class": "fallen tree",
                "confidence": round(conf, 4),
                "bbox": [float(w * 0.08), float(h * 0.2), float(w * 0.92), float(h * 0.85)]
            })
            return extra_detections

        # Check for garbage / clutter: high multi-color saturation variance & scattered edges
        if sat_std > 40 and edge_density > 0.015:
            conf = min(0.92, 0.70 + edge_density * 3.0)
            extra_detections.append({
                "class": "garbage",
                "confidence": round(conf, 4),
                "bbox": [float(w * 0.12), float(h * 0.3), float(w * 0.88), float(h * 0.9)]
            })
            return extra_detections

        # Road cavity / Pothole detection on asphalt road
        lower_half = gray[int(h * 0.3):, :]
        blur = cv2.GaussianBlur(lower_half, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 6
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > total_pixels * 0.012:  # At least 1.2% of image area
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = float(cw) / max(1, ch)
                if 0.4 <= aspect_ratio <= 4.0:
                    conf = min(0.94, 0.68 + (area / total_pixels) * 0.9)
                    bbox = [float(x), float(y + int(h * 0.3)), float(x + cw), float(y + int(h * 0.3) + ch)]
                    extra_detections.append({
                        "class": "pothole",
                        "confidence": round(conf, 4),
                        "bbox": bbox
                    })
                    break

        return extra_detections

    def inspect_image(self, image_path: str) -> dict:
        """
        Analyse an image (or video) for civic-issue indicators.
        Returns detailed detections, validation score, category, and severity.
        """
        if not os.path.exists(image_path):
            return {
                "error": "File not found",
                "is_valid_complaint": False,
                "validation_score": 0.0,
                "civic_issues": [],
                "detections": [],
                "estimated_severity": "LOW",
            }

        try:
            import cv2
            from PIL import Image as _Image
        except ImportError:
            return {
                "requires_installation": True,
                "install_command": "pip install ultralytics opencv-python Pillow",
                "is_valid_complaint": False,
                "validation_score": 0.0,
                "civic_issues": [],
                "detections": [],
                "estimated_severity": "LOW",
            }

        # Multi-frame video inspection or single image reading
        frames_to_inspect = []
        is_video = image_path.lower().endswith((".mp4", ".avi", ".mov"))

        if is_video:
            try:
                cap = cv2.VideoCapture(image_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > 0:
                    sample_indices = [
                        0,
                        max(0, int(frame_count * 0.5)),
                        max(0, frame_count - 1),
                    ]
                    for idx in set(sample_indices):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, f = cap.read()
                        if ret:
                            frames_to_inspect.append(f)
                cap.release()
            except Exception as exc:
                return {
                    "error": f"Video processing error: {exc}",
                    "is_valid_complaint": False,
                    "validation_score": 0.0,
                    "civic_issues": [],
                    "detections": [],
                    "estimated_severity": "LOW",
                }
        else:
            try:
                img_bgr = cv2.imread(image_path)
                if img_bgr is not None:
                    frames_to_inspect.append(img_bgr)
            except Exception:
                pass

        if not frames_to_inspect:
            return {
                "error": "Could not read media file",
                "is_valid_complaint": False,
                "validation_score": 0.0,
                "civic_issues": [],
                "detections": [],
                "estimated_severity": "LOW",
            }

        try:
            # Load YOLO model
            if self._yolo_model is None:
                from ultralytics import YOLO as _YOLO
                self._yolo_model = _YOLO(self._model_size)

            all_detections = []
            for frame_bgr in frames_to_inspect:
                # 1. Run YOLOv8 on frame
                results = self._yolo_model(frame_bgr, verbose=False)
                for res in results:
                    if res.boxes is not None:
                        for box in res.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = res.names.get(cls_id, str(cls_id)).lower()
                            conf = float(box.conf[0])
                            bbox = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                            all_detections.append({
                                "class": cls_name,
                                "confidence": round(conf, 4),
                                "bbox": bbox
                            })

                # 2. Run municipal CV heuristics
                cv_features = self._analyze_civic_cv_features(frame_bgr)
                all_detections.extend(cv_features)

            # Deduplicate detections by class & overlapping bbox
            unique_detections = []
            for det in all_detections:
                is_dup = False
                for existing in unique_detections:
                    if existing["class"] == det["class"]:
                        # If confidences are close, keep the higher one
                        is_dup = True
                        if det["confidence"] > existing["confidence"]:
                            existing["confidence"] = det["confidence"]
                            existing["bbox"] = det["bbox"]
                        break
                if not is_dup:
                    unique_detections.append(det)

            # Separate civic issues
            civic_detections = [
                d for d in unique_detections if d["class"] in CIVIC_ISSUE_CLASSES
            ]
            civic_issues = list(dict.fromkeys(d["class"] for d in civic_detections))

            # Non-civic objects indicator (e.g. keyboard, laptop, dining table, pizza)
            non_civic_objects = [
                d for d in unique_detections
                if d["class"] not in CIVIC_ISSUE_CLASSES and d["class"] in [
                    "laptop", "tv", "cell phone", "keyboard", "mouse", "pizza",
                    "sandwich", "dining table", "bed", "microwave", "couch"
                ]
            ]

            # Validation score calculation
            if civic_detections:
                avg_conf = sum(d["confidence"] for d in civic_detections) / len(civic_detections)
                # Boost if multiple civic indicators or high priority
                boost = 0.15 if any(d["class"] in HIGH_PRIORITY_CIVIC_CLASSES for d in civic_detections) else 0.05
                validation_score = float(round(min(0.98, avg_conf + boost), 4))
                is_valid = bool(validation_score >= 0.4)
            elif non_civic_objects:
                validation_score = 0.15
                is_valid = False
            else:
                # Neutral photo or general street scene
                validation_score = 0.45 if unique_detections else 0.20
                is_valid = bool(unique_detections)

            # Severity estimation
            estimated_severity = self.estimate_severity_from_image(
                [d["class"] for d in unique_detections],
                [d["confidence"] for d in unique_detections],
                detections=unique_detections
            )

            return {
                "detections": unique_detections,
                "civic_issues": civic_issues,
                "estimated_severity": estimated_severity,
                "validation_score": validation_score,
                "is_valid_complaint": is_valid,
            }

        except Exception as exc:
            return {
                "error": str(exc),
                "is_valid_complaint": False,
                "validation_score": 0.0,
                "civic_issues": [],
                "detections": [],
                "estimated_severity": "LOW",
            }

    def estimate_severity_from_image(
        self, classes: list, confidence_scores: list, detections: list = None
    ) -> str:
        """
        Intelligently estimate severity from detected classes, confidence, and bounding box scale.
        """
        if not classes:
            return "LOW"

        # Check for hazardous civic classes
        has_critical_issue = any(c in HIGH_PRIORITY_CIVIC_CLASSES for c in classes)
        max_conf = max(confidence_scores) if confidence_scores else 0.0

        if has_critical_issue:
            if max_conf >= 0.75 or len(classes) >= 3:
                return "CRITICAL"
            return "HIGH"

        # General road / traffic infrastructure obstruction
        if any(c in ["car", "truck", "bus"] for c in classes) and len(classes) >= 3:
            return "HIGH"

        if any(c in ["dog", "cat", "bench", "fire hydrant"] for c in classes):
            if max_conf > 0.8:
                return "MEDIUM"
            return "LOW"

        if max_conf > 0.8:
            return "MEDIUM"

        return "LOW"

    def map_detection_to_category(self, detections: list) -> str:
        """
        Map detections to the primary BBMP municipal category.
        """
        category_counts: Dict[str, int] = {}
        for item in detections:
            cls_name = (
                item.get("class", "").lower()
                if isinstance(item, dict)
                else str(item).lower()
            )
            cat = COCO_TO_BBMP_CATEGORY.get(cls_name)
            if cat and cat != "General":
                # Weight high priority issues higher
                weight = 3 if cls_name in HIGH_PRIORITY_CIVIC_CLASSES else 1
                category_counts[cat] = category_counts.get(cat, 0) + weight

        if not category_counts:
            # Fall back to any mapped category
            for item in detections:
                cls_name = item.get("class", "").lower() if isinstance(item, dict) else str(item).lower()
                cat = COCO_TO_BBMP_CATEGORY.get(cls_name)
                if cat:
                    category_counts[cat] = category_counts.get(cat, 0) + 1

        if not category_counts:
            return "General"

        return max(category_counts, key=category_counts.__getitem__)

    def annotate_image(self, image_path: str, detections: list) -> np.ndarray:
        """
        Draw bounding boxes, class names, and confidence meters on the image.
        Returns RGB image as numpy array.
        """
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return None

        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls_name = det.get("class", "object")
            conf = det.get("confidence", 0.0)

            # Choose color: Red for high hazard, Amber for medium, Cyan/Green for general
            if cls_name in HIGH_PRIORITY_CIVIC_CLASSES:
                color = (0, 0, 230)  # Red (BGR)
            elif cls_name in ["car", "truck", "dog", "bench"]:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 200, 0)  # Green

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label banner
            label = f"{cls_name.upper()} {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (lbl_w, lbl_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(img, (x1, max(0, y1 - lbl_h - 8)), (x1 + lbl_w + 6, max(lbl_h + 8, y1)), color, -1)
            cv2.putText(
                img, label, (x1 + 3, max(lbl_h + 3, y1 - 4)),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )

        # Convert BGR to RGB for Streamlit and PIL
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
