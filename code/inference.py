import os
import json
import time
import logging
import base64
from io import BytesIO
from typing import Any, List, Tuple, Optional, Dict

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
SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/jpg", "application/json")
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


def input_fn(
    request_body: bytes, request_content_type: str
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decode image from request body and extract optional parameters.

    Supports two formats:
    1. Raw image bytes (image/jpeg, image/png, image/jpg)
    2. JSON with base64 image and parameters (application/json)

    Args:
        request_body: Raw bytes of the image or JSON request
        request_content_type: MIME type of the request

    Returns:
        Tuple of (image array in RGB format, dict of optional parameters)
    """
    if request_content_type not in SUPPORTED_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported content type: {request_content_type}. "
            f"Supported types: {SUPPORTED_CONTENT_TYPES}"
        )

    # Initialize parameters dictionary
    params = {
        "confidence": None,
        "classes": None,  # None means all classes (no filtering)
        "max_detections": None,  # None means no limit
        "min_box_area": None,  # None means no minimum
    }

    # Handle JSON request format
    if request_content_type == "application/json":
        try:
            request_data = json.loads(request_body)

            # Extract base64-encoded image
            if "image" not in request_data:
                raise ValueError(
                    "JSON request must contain 'image' field with base64-encoded image"
                )

            image_data = base64.b64decode(request_data["image"])

            # Extract optional confidence threshold
            if "confidence" in request_data:
                confidence = float(request_data["confidence"])
                if not (0.0 <= confidence <= 1.0):
                    raise ValueError(
                        f"Confidence must be between 0.0 and 1.0, got {confidence}"
                    )
                params["confidence"] = confidence
                logger.info("Using request confidence threshold: %.3f", confidence)

            # Extract optional class filter (list of class names or None for all)
            if "classes" in request_data:
                classes = request_data["classes"]
                if classes is not None:
                    if not isinstance(classes, list):
                        raise ValueError(
                            f"'classes' must be a list or null, got {type(classes).__name__}"
                        )
                    if not all(isinstance(c, str) for c in classes):
                        raise ValueError("All elements in 'classes' must be strings")
                    params["classes"] = classes
                    logger.info("Filtering to classes: %s", classes)
                else:
                    logger.info("classes=null: returning all classes (no filtering)")

            # Extract optional max detections
            if "max_detections" in request_data:
                max_det = request_data["max_detections"]
                if max_det is not None:
                    max_det = int(max_det)
                    if max_det <= 0:
                        raise ValueError(
                            f"'max_detections' must be positive, got {max_det}"
                        )
                    params["max_detections"] = max_det
                    logger.info("Limiting to top %d detections", max_det)

            # Extract optional minimum box area
            if "min_box_area" in request_data:
                min_area = request_data["min_box_area"]
                if min_area is not None:
                    min_area = float(min_area)
                    if min_area < 0:
                        raise ValueError(
                            f"'min_box_area' must be non-negative, got {min_area}"
                        )
                    params["min_box_area"] = min_area
                    logger.info(
                        "Filtering detections with box area >= %.1f px²", min_area
                    )

            request_body = image_data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid JSON request structure: {e}")

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

        return img_array, params

    except Image.UnidentifiedImageError as e:
        raise ValueError(f"Invalid image format or corrupted image data: {e}")
    except Exception as e:
        logger.error("Image decoding failed: %s", e)
        raise ValueError(f"Failed to decode image: {e}")


def predict_fn(
    input_data: Tuple[np.ndarray, Dict[str, Any]], model
) -> Tuple[List[Any], Dict[str, Any]]:
    """Run inference on input image using RF-DETR.

    Args:
        input_data: Tuple of (image as numpy array in RGB format, dict of optional parameters)
        model: Loaded RF-DETR model

    Returns:
        Tuple of (list of supervision Detections objects, request parameters dict)
    """
    # Unpack input data
    img_array, params = input_data

    # Use request confidence if provided, otherwise fall back to model's default
    request_confidence = params.get("confidence")
    conf_threshold = (
        request_confidence
        if request_confidence is not None
        else getattr(model, "conf_threshold", DEFAULT_CONF_THRESHOLD)
    )

    logger.info(
        "Running RF-DETR inference on image shape=%s, conf_threshold=%.3f%s",
        img_array.shape,
        conf_threshold,
        " (from request)" if request_confidence is not None else " (default)",
    )

    start = time.perf_counter()
    try:
        # RF-DETR predict method handles preprocessing internally
        # Input should be RGB numpy array in range [0, 255]
        detections = model.predict(img_array, threshold=conf_threshold)

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

    # Add the actual confidence threshold used to params for metadata
    params["confidence_used"] = conf_threshold

    # Return detections along with params for post-processing in output_fn
    return detections, params


def output_fn(
    prediction_output: Tuple[List[Any], Dict[str, Any]], content_type: str
) -> str:
    """Format RF-DETR prediction results as JSON and apply post-processing filters.

    Args:
        prediction_output: Tuple of (list of supervision Detections objects, request params dict)
        content_type: Desired output content type

    Returns:
        JSON string with detections and metadata
    """
    # Unpack prediction output
    detections_list, params = prediction_output

    detections = []
    inference_time_ms = 0
    image_shape = None

    # Extract filtering parameters
    filter_classes = params.get("classes")  # None means no filtering
    max_detections = params.get("max_detections")
    min_box_area = params.get("min_box_area")

    # Convert class names to lowercase for case-insensitive matching
    if filter_classes is not None:
        filter_classes_lower = [c.lower() for c in filter_classes]

    # detections_list is a list of supervision Detections objects
    for detection in detections_list:
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

                # Apply class filter if specified
                if (
                    filter_classes is not None
                    and label.lower() not in filter_classes_lower
                ):
                    continue

                # Calculate box area for filtering
                box_area = (float(x2) - float(x1)) * (float(y2) - float(y1))

                # Apply minimum box area filter if specified
                if min_box_area is not None and box_area < min_box_area:
                    continue

                detections.append(
                    {
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "class_id": cls_id,
                        "label": label,
                        "area": box_area,  # Include area in response for debugging
                    }
                )

    # Count before max_detections filtering
    total_before_limit = len(detections)

    # Sort by confidence descending before applying max_detections
    if detections:
        detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Apply max detections limit if specified
    if max_detections is not None and len(detections) > max_detections:
        detections = detections[:max_detections]

    # Build response with metadata
    response = {
        "detections": detections,
        "metadata": {
            "count": len(detections),
            "inference_time_ms": inference_time_ms,
        },
    }

    # Add filter information to metadata
    applied_filters = {}

    # Always include confidence threshold (it's always applied)
    confidence_used = params.get("confidence_used")
    if confidence_used is not None:
        applied_filters["confidence"] = confidence_used

    if filter_classes is not None:
        applied_filters["classes"] = filter_classes
    if max_detections is not None:
        applied_filters["max_detections"] = max_detections
        if total_before_limit > max_detections:
            applied_filters["total_before_limit"] = total_before_limit
    if min_box_area is not None:
        applied_filters["min_box_area"] = min_box_area

    if applied_filters:
        response["metadata"]["applied_filters"] = applied_filters

    if image_shape is not None:
        response["metadata"]["image_shape"] = {
            "height": int(image_shape[0]),
            "width": int(image_shape[1]),
        }

    logger.info(
        "Returning %d detections (filters: %s)",
        len(detections),
        applied_filters if applied_filters else "none",
    )

    return json.dumps(response)
