# Demonstrates different video detection and segmentation options
# using RFDETRBase model and Supervision library
# Author: Gary Stafford
# Date: December 2025

import os
import cv2
import supervision as sv
from rfdetr import RFDETRBase  # or RFDETRNano/Small/Medium/Large
from rfdetr.util.coco_classes import COCO_CLASSES

# ========= CONFIG =========
SOURCE_VIDEO_PATH = "local_visualizations/sample_video.mp4"  # <-- set your input video
TARGET_VIDEO_PATH = (
    "local_visualizations/sample_video_annotated.mp4"  # <-- set your output video
)
THRESHOLD = 0.25  # confidence threshold for displaying boxes
# ==========================

assert os.path.exists(SOURCE_VIDEO_PATH), f"Source video not found: {SOURCE_VIDEO_PATH}"

# Optional: sanity-check that OpenCV can read the first frame
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
ret, test_frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError(f"Could not read first frame from {SOURCE_VIDEO_PATH}")
print("First frame shape:", test_frame.shape)

print("Loading RF-DETR model...")
model = RFDETRBase()  # change to RFDETRNano/Small/Medium/Large if desired
print("Optimizing model for inference...")
model.optimize_for_inference(compile=False)
print("Model loaded and optimized.")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


def callback(frame, index: int):
    # supervision gives BGR; RF-DETR expects RGB
    # .copy() is needed because [:, :, ::-1] creates negative strides which PyTorch doesn't support
    rgb_frame = frame[:, :, ::-1].copy()

    detections = model.predict(rgb_frame, threshold=THRESHOLD)

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

    if index % 50 == 0:
        print(f"Processed frame {index}")

    return annotated_frame


print("Starting video processing...")
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback,
)
print("Processing complete.")
print("Final file size (bytes):", os.path.getsize(TARGET_VIDEO_PATH))
