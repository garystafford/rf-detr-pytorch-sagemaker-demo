# Demonstrates different image detection and segmentation options
# using RFDETRSegPreview model and Supervision library
# Author: Gary Stafford
# Date: December 2025

from PIL import Image
import supervision as sv
from rfdetr import RFDETRSegPreview
from rfdetr.util.coco_classes import COCO_CLASSES
import numpy as np

# 1. Load model (public Seg Preview weights)
model = RFDETRSegPreview(pretrain_weights="rf-detr-seg-preview.pt")

# Optionally optimize for latency
# model.optimize_for_inference()

# 2. Load image (RGB)
image = Image.open("local_visualizations/sample_image.png").convert("RGB")

default_palette = sv.ColorPalette.DEFAULT  # default supervision colors
roboflow_palette = sv.ColorPalette.ROBOFLOW  # Roboflow brand-ish colors

# For a single image, RF-DETR Seg Preview typically returns a Detections object
detections = model.predict(image, threshold=0.5)

# If your version returns a list, uncomment this:
# detections = model.predict(image, threshold=0.5)[0]

# If class names are already included, you can use them directly
# labels = detections["class_name"]

# Build human-readable labels from COCO classes
labels = [
    f"{COCO_CLASSES[class_id]} {confidence * 100:.1f}%"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

# Annotators
mask_annotator = sv.MaskAnnotator(
    # color=sv.Color.from_hex("#01AF01"),  # mask color / palette
    opacity=0.4,  # mask transparency
)
box_annotator = sv.BoxAnnotator(
    # color=sv.Color.from_hex("#01AF01"),  # box color / palette
    thickness=2,
)
label_annotator = sv.LabelAnnotator(
    # text_color=sv.Color.WHITE,  # label text color
    # color=sv.Color.from_hex("#01AF01"),  # label background box color / palette
    text_scale=0.9,  # or label_scale / font_size depending on version
    text_padding=10,
    text_thickness=2,
    smart_position=True,  # <– try to avoid overlapping labels
)

# just image
annotated_00 = image.copy()
annotated_00.save("sample_image_annotated_00.jpg")

# just masks
annotated_01 = image.copy()
annotated_01 = mask_annotator.annotate(annotated_01, detections)
annotated_01.save("sample_image_annotated_01.jpg")

# just boxes and masks
annotated_02 = image.copy()
annotated_02 = mask_annotator.annotate(annotated_02, detections)
annotated_02 = box_annotator.annotate(annotated_02, detections=detections)
annotated_02.save("sample_image_annotated_02.jpg")

# boxes, labels
annotated_03 = image.copy()
annotated_03 = label_annotator.annotate(
    scene=annotated_03,
    detections=detections,
    labels=labels,  # overrides default labeling
)
annotated_03 = box_annotator.annotate(annotated_03, detections=detections)
annotated_03.save("sample_image_annotated_03.jpg")

# masks, labels
annotated_04 = image.copy()
annotated_04 = mask_annotator.annotate(annotated_04, detections)
annotated_04 = label_annotator.annotate(
    scene=annotated_04,
    detections=detections,
    labels=labels,  # overrides default labeling
)
annotated_04.save("sample_image_annotated_04.jpg")


# detections.mask: (N, H, W) boolean / 0‑1 masks for each instance
masks = detections.mask

# 3. Prepare a numpy copy of the original image to draw on
outlined = np.array(image.copy())  # RGB, same as original

# 4. For each instance mask, convert to polygons and draw outlines
for i, mask in enumerate(masks):
    # mask_to_polygons returns a list of polygons for this instance
    # each polygon is an (K, 2) ndarray of (x, y) vertices
    polygons = sv.mask_to_polygons(mask)  # supervision util[web:112][web:128]
    color = default_palette.by_idx(i % len(default_palette.colors))

    for polygon in polygons:
        outlined = sv.draw_polygon(
            scene=outlined,
            polygon=polygon,
            color=color,  # sv.Color.from_hex("#01AF01"),  # outline color
            thickness=2,  # outline thickness
        )

# 5. Save or display the overlaid result
outlined_image = Image.fromarray(outlined)
annotated_05 = label_annotator.annotate(
    scene=outlined_image,
    detections=detections,
    labels=labels,  # overrides default labeling
)
annotated_05.save("sample_image_annotated_05.jpg")
