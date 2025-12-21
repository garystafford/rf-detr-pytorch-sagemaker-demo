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

VARIANTS = {
    "nano": {"file": "rf-detr-nano.pth", "type": "rfdetr-nano", "resolution": "384"},
    "small": {"file": "rf-detr-small.pth", "type": "rfdetr-small", "resolution": "512"},
    "medium": {
        "file": "rf-detr-medium.pth",
        "type": "rfdetr-medium",
        "resolution": "576",
    },
    "base": {"file": "rf-detr-base.pth", "type": "rfdetr-base", "resolution": "560"},
    "large": {"file": "rf-detr-large.pth", "type": "rfdetr-large", "resolution": "560"},
}


def model_fn(model_dir: str) -> Any:
    """Load and prepare RF-DETR model for inference.

    Args:
        model_dir: Directory containing the model checkpoint

    Returns:
        Configured RF-DETR model ready for inference
    """
    logger.info("Loading RF-DETR model from %s", model_dir)

    # Resolve variant from single env knob
    variant = os.getenv("RFDETR_VARIANT", "large")
    if variant not in VARIANTS:
        raise ValueError(
            f"Invalid RFDETR_VARIANT: {variant}. Must be one of {list(VARIANTS.keys())}"
        )

    cfg = VARIANTS[variant]
    model_name = cfg["file"]
    model_type = cfg["type"]
    resolution = cfg["resolution"]

    logger.info(
        "Using variant=%s (file=%s, type=%s, resolution=%s)",
        variant,
        model_name,
        model_type,
        resolution,
    )

    checkpoint_path = os.path.join(model_dir, model_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    try:
        model_class = MODEL_CLASSES[model_type]
        model = model_class(
            pretrain_weights=checkpoint_path, device=device, resolution=int(resolution)
        )
    except Exception as e:
        logger.error("Failed to load RF-DETR model: %s", e)
        raise RuntimeError(f"Model loading failed: {e}") from e

    # Apply optimization (enabled by default)
    optimize_enabled = os.getenv("RFDETR_OPTIMIZE", "true").lower() == "true"
    if optimize_enabled:
        compile_flag = os.getenv("RFDETR_COMPILE", "true").lower() == "true"
        logger.info("Optimizing model for inference (compile=%s)", compile_flag)
        try:
            model.optimize_for_inference(
                compile=compile_flag, batch_size=1, dtype=torch.float32
            )
            logger.info("Model optimization completed")
        except Exception as e:
            logger.warning("Could not optimize model: %s", e)

    # Store confidence threshold on model object
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
        "Model loaded (type=%s, resolution=%s, conf=%.2f)",
        model_type,
        resolution,
        model.conf_threshold,
    )

    return model


def _extract_json_params(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate optional parameters from JSON request."""
    params: Dict[str, Any] = {
        "confidence": None,
        "classes": None,
        "max_detections": None,
        "min_box_area": None,
    }

    # Extract confidence threshold
    if "confidence" in request_data:
        confidence = float(request_data["confidence"])
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {confidence}"
            )
        params["confidence"] = confidence
        logger.info("Using request confidence threshold: %.3f", confidence)

    # Extract class filter
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

    # Extract max detections limit
    if "max_detections" in request_data:
        max_det = request_data["max_detections"]
        if max_det is not None:
            max_det = int(max_det)
            if max_det <= 0:
                raise ValueError(f"'max_detections' must be positive, got {max_det}")
            params["max_detections"] = max_det
            logger.info("Limiting to top %d detections", max_det)

    # Extract minimum box area
    if "min_box_area" in request_data:
        min_area = request_data["min_box_area"]
        if min_area is not None:
            min_area = float(min_area)
            if min_area < 0:
                raise ValueError(f"'min_box_area' must be non-negative, got {min_area}")
            params["min_box_area"] = min_area
            logger.info("Filtering detections with box area >= %.1f px²", min_area)

    return params


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

    # Handle JSON request format
    if request_content_type == "application/json":
        try:
            request_data = json.loads(request_body)
            if "image" not in request_data:
                raise ValueError(
                    "JSON request must contain 'image' field with base64-encoded image"
                )
            image_data = base64.b64decode(request_data["image"])
            params = _extract_json_params(request_data)
            request_body = image_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from e
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid JSON request structure: {e}") from e
    else:
        # No parameters for raw image requests
        params = {
            "confidence": None,
            "classes": None,
            "max_detections": None,
            "min_box_area": None,
        }

    # Decode and validate image
    try:
        img = Image.open(BytesIO(request_body))
        width, height = img.size

        if max(width, height) > MAX_IMAGE_DIMENSION:
            raise ValueError(
                f"Image dimensions ({width}x{height}) exceed maximum "
                f"({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
            )

        logger.info("Decoded image: %dx%d, mode=%s", width, height, img.mode)

        if img.mode != "RGB":
            logger.debug("Converting image from %s to RGB", img.mode)
            img = img.convert("RGB")

        return np.array(img), params

    except Image.UnidentifiedImageError as e:
        raise ValueError(f"Invalid image format or corrupted image data: {e}") from e
    except Exception as e:
        logger.error("Image decoding failed: %s", e)
        raise ValueError(f"Failed to decode image: {e}") from e


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
    img_array, params = input_data

    # Use request confidence if provided, otherwise fall back to model's default
    request_conf = params.get("confidence")
    conf_threshold = (
        request_conf
        if request_conf is not None
        else getattr(model, "conf_threshold", DEFAULT_CONF_THRESHOLD)
    )
    conf_source = "from request" if request_conf is not None else "default"

    logger.info(
        "Running inference on image shape=%s, conf=%.3f (%s)",
        img_array.shape,
        conf_threshold,
        conf_source,
    )

    start = time.perf_counter()
    try:
        detections = model.predict(img_array, threshold=conf_threshold)
    except Exception as e:
        logger.error("RF-DETR inference failed: %s", e)
        raise RuntimeError(f"Model inference failed: {e}") from e

    elapsed = (time.perf_counter() - start) * 1000

    # Ensure detections is a list
    detections = detections if isinstance(detections, list) else [detections]

    # Count and log detections
    total_detections = sum(len(d.xyxy) if hasattr(d, "xyxy") else 0 for d in detections)
    logger.info(
        "Inference completed in %.2f ms, detected %d objects", elapsed, total_detections
    )

    # Store metadata for output_fn
    for detection in detections:
        detection.inference_time_ms = elapsed
    params["confidence_used"] = conf_threshold

    return detections, params


def _should_filter_detection(
    label: str,
    box_area: float,
    filter_classes_lower: Optional[List[str]],
    min_box_area: Optional[float],
) -> bool:
    """Check if a detection should be filtered out based on class or area."""
    if filter_classes_lower is not None and label.lower() not in filter_classes_lower:
        return True
    if min_box_area is not None and box_area < min_box_area:
        return True
    return False


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
    detections_list, params = prediction_output

    # Extract filtering parameters
    filter_classes = params.get("classes")
    max_detections = params.get("max_detections")
    min_box_area = params.get("min_box_area")

    # Pre-compute lowercased class list for case-insensitive filtering
    filter_classes_lower = (
        [c.lower() for c in filter_classes] if filter_classes else None
    )

    detections = []
    inference_time_ms = 0

    # Process all detections
    for detection in detections_list:
        inference_time_ms = getattr(detection, "inference_time_ms", 0)

        if not (
            hasattr(detection, "xyxy")
            and detection.xyxy is not None
            and len(detection.xyxy) > 0
        ):
            continue

        for box, conf, cls_id in zip(
            detection.xyxy, detection.confidence, detection.class_id
        ):
            x1, y1, x2, y2 = box
            cls_id = int(cls_id)
            label = COCO_CLASSES.get(cls_id, f"unknown_{cls_id}")
            box_area = (float(x2) - float(x1)) * (float(y2) - float(y1))

            # Apply filters
            if _should_filter_detection(
                label, box_area, filter_classes_lower, min_box_area
            ):
                continue

            detections.append(
                {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class_id": cls_id,
                    "label": label,
                    "area": box_area,
                }
            )

    # Sort by confidence and apply max detections limit
    total_before_limit = len(detections)
    detections.sort(key=lambda d: d["confidence"], reverse=True)
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
    if (confidence_used := params.get("confidence_used")) is not None:
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

    logger.info(
        "Returning %d detections (filters: %s)",
        len(detections),
        applied_filters or "none",
    )

    return json.dumps(response)
