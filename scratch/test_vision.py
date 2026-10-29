import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.visual_inspector import VisualInspector

inspector = VisualInspector()
images = [
    "data/sample_multimodal/images/BBMP-SAMPLE-007.jpg",
    "data/sample_multimodal/images/BBMP-SAMPLE-008.jpg",
    "data/sample_multimodal/images/BBMP-SAMPLE-010.jpg",
]

for img_path in images:
    res = inspector.inspect_image(img_path)
    annotated = inspector.annotate_image(img_path, res["detections"])
    cat = inspector.map_detection_to_category(res["detections"])
    print("=" * 60)
    print("Image:", img_path)
    print("Valid:", res["is_valid_complaint"], "| Score:", res["validation_score"])
    print("Estimated Severity:", res["estimated_severity"])
    print("Category:", cat)
    print("Civic Issues:", res["civic_issues"])
    print("Detections count:", len(res["detections"]))
    print("Annotated image shape:", annotated.shape if annotated is not None else None)
