import os
import json
import time
import logging
from io import BytesIO
from typing import Any, List

import numpy as np
from PIL import Image
import torch
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRBase, RFDETRSmall, RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONF_THRESHOLD = 0.25
SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/jpg")
MAX_IMAGE_DIMENSION = 10000  # Prevent extremely large images

# Model type mapping
MODEL_CLASSES = {
    "rfdetr-nano": RFDETRNano,
    "rfdetr-small": RFDETRSmall,
    "rfdetr-medium": RFDETRMedium,
    "rfdetr-base": RFDETRBase,
    "rfdetr-large": RFDETRLarge,
}


def model_fn(model_dir: str):
    """Load and prepare RF-DETR model for inference.

    Args:
        model_dir: Directory containing the model checkpoint

    Returns:
        Configured RF-DETR model ready for inference
    """
    logger.info("Loading RF-DETR model from %s", model_dir)

    # Get configuration from environment variables
    model_name = os.getenv("RFDETR_MODEL", "rf-detr-large.pth")
    checkpoint_path = os.path.join(model_dir, model_name)
    model_type = os.getenv("RFDETR_MODEL_TYPE", "rfdetr-large")
    resolution = os.getenv("RFDETR_RESOLUTION", "560")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Validate model_type
    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be one of {list(MODEL_CLASSES.keys())}"
        )

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    try:
        # Load RF-DETR model
        model_class = MODEL_CLASSES[model_type]
        model_kwargs = {
            "pretrain_weights": checkpoint_path,
            "device": device,
            "resolution": int(resolution),
        }

        model = model_class(**model_kwargs)

    except Exception as e:
        logger.error("Failed to load RF-DETR model: %s", e)
        raise RuntimeError(f"Model loading failed: {e}") from e

    # Apply optimization (enabled by default for production)
    if os.getenv("RFDETR_OPTIMIZE", "true").lower() == "true":
        compile_flag = os.getenv("RFDETR_COMPILE", "true").lower() == "true"
        logger.info("Optimizing model for inference (compile=%s)", compile_flag)
        try:
            model.optimize_for_inference(
                compile=compile_flag, batch_size=1, dtype=torch.float32
            )
            logger.info("Model optimization completed successfully")
        except Exception as e:
            logger.warning("Could not optimize model: %s", e)

    # Store configuration on model object
    try:
        model.conf_threshold = float(
            os.getenv("RFDETR_CONF", str(DEFAULT_CONF_THRESHOLD))
        )
    except ValueError:
        logger.warning(
            "Invalid RFDETR_CONF value, using default: %s", DEFAULT_CONF_THRESHOLD
        )
        model.conf_threshold = DEFAULT_CONF_THRESHOLD

    logger.info(
        "Model loaded successfully with confidence threshold: %.2f",
        model.conf_threshold,
    )
    logger.info("Model type: %s", model_type)
    logger.info("Model resolution: %s", resolution)

    return model


def input_fn(request_body: bytes, request_content_type: str) -> np.ndarray:
    """Decode image from request body.

    Args:
        request_body: Raw bytes of the image
        request_content_type: MIME type of the image

    Returns:
        Numpy array in RGB format ready for RF-DETR inference
    """
    if request_content_type not in SUPPORTED_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported content type: {request_content_type}. "
            f"Supported types: {SUPPORTED_CONTENT_TYPES}"
        )

    # Decode image bytes using Pillow
    try:
        img = Image.open(BytesIO(request_body))

        # Validate image dimensions
        width, height = img.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ValueError(
                f"Image dimensions ({width}x{height}) exceed maximum allowed "
                f"({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
            )

        logger.info("Decoded image: %dx%d, mode=%s", width, height, img.mode)

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            logger.debug("Converting image from %s to RGB", img.mode)
            img = img.convert("RGB")

        # Convert PIL Image to numpy array (RGB format)
        img_array = np.array(img)

        return img_array

    except Image.UnidentifiedImageError as e:
        raise ValueError(f"Invalid image format or corrupted image data: {e}")
    except Exception as e:
        logger.error("Image decoding failed: %s", e)
        raise ValueError(f"Failed to decode image: {e}")


def predict_fn(input_data: np.ndarray, model) -> List[Any]:
    """Run inference on input image using RF-DETR.

    Args:
        input_data: Image as numpy array in RGB format
        model: Loaded RF-DETR model

    Returns:
        List of supervision Detections objects from RF-DETR
    """
    conf_threshold = getattr(model, "conf_threshold", DEFAULT_CONF_THRESHOLD)
    logger.info(
        "Running RF-DETR inference on image shape=%s, conf_threshold=%.2f",
        input_data.shape,
        conf_threshold,
    )

    start = time.perf_counter()
    try:
        # RF-DETR predict method handles preprocessing internally
        # Input should be RGB numpy array in range [0, 255]
        detections = model.predict(input_data, threshold=conf_threshold)

    except Exception as e:
        logger.error("RF-DETR inference failed: %s", e)
        raise RuntimeError(f"Model inference failed: {e}") from e

    elapsed = (time.perf_counter() - start) * 1000

    # Ensure detections is a list
    if not isinstance(detections, list):
        detections = [detections]

    # Count detections
    total_detections = sum(len(d.xyxy) if hasattr(d, "xyxy") else 0 for d in detections)
    logger.info(
        "Inference completed in %.2f ms, detected %d objects", elapsed, total_detections
    )

    # Store inference time and metadata on detections for output_fn
    for detection in detections:
        detection.inference_time_ms = elapsed

    return detections


def output_fn(prediction_output: List[Any], content_type: str) -> str:
    """Format RF-DETR prediction results as JSON.

    Args:
        prediction_output: List of supervision Detections objects from RF-DETR
        content_type: Desired output content type

    Returns:
        JSON string with detections and metadata
    """
    detections = []
    inference_time_ms = 0
    image_shape = None

    # prediction_output is a list of supervision Detections objects
    for detection in prediction_output:
        # Extract inference time if available
        inference_time_ms = getattr(detection, "inference_time_ms", 0)

        # supervision Detections has: xyxy, confidence, class_id (all numpy arrays)
        if (
            hasattr(detection, "xyxy")
            and detection.xyxy is not None
            and len(detection.xyxy) > 0
        ):
            boxes_np = detection.xyxy  # Already numpy
            confidences_np = detection.confidence  # Already numpy
            class_ids_np = detection.class_id  # Already numpy

            for box, conf, cls_id in zip(boxes_np, confidences_np, class_ids_np):
                x1, y1, x2, y2 = box
                cls_id = int(cls_id)

                # Map class id to label using COCO_CLASSES dictionary
                # Model outputs 1-based COCO class IDs directly
                label = COCO_CLASSES.get(cls_id, f"unknown_{cls_id}")

                detections.append(
                    {
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "class_id": cls_id,
                        "label": label,
                    }
                )

    # Build response with metadata
    response = {
        "detections": detections,
        "metadata": {
            "count": len(detections),
            "inference_time_ms": inference_time_ms,
        },
    }

    if image_shape is not None:
        response["metadata"]["image_shape"] = {
            "height": int(image_shape[0]),
            "width": int(image_shape[1]),
        }

    logger.info("Returning %d detections", len(detections))

    return json.dumps(response)
