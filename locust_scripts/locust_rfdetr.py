import logging
import json
import time
from io import BytesIO
import random
import os
import base64

import boto3
from locust import task, events
from locust.contrib.fasthttp import FastHttpUser
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

IMG_SIZE = 560  # max image dimension for inference

# Confidence thresholds to test (randomly selected per request)
# Set to None to use endpoint default, or provide a list to randomly select
CONFIDENCE_THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4]

# Class filtering options (randomly selected per request)
# Set to None to disable class filtering, or provide a list of class filter options
# Each option can be None (all classes) or a list of class names
CLASS_FILTERS = [
    None,  # No filtering - return all classes
    ["person"],  # Only people
    ["person", "car", "truck", "bus"],  # People and vehicles
    ["dog", "cat", "bird"],  # Pets and birds
]

# Max detections options (randomly selected per request)
# Set to None to disable limit, or provide a list of max detection values
MAX_DETECTIONS_OPTIONS = [None, 5, 10, 20]  # None means no limit

# Minimum box area options (randomly selected per request)
# Set to None to disable filtering, or provide a list of minimum area values (in pixels²)
MIN_BOX_AREA_OPTIONS = [None, 500, 1000, 2000]  # None means no filtering

region = "us-east-1"
endpoint_name = "<YOUR_SAGEMAKER_ENDPOINT_NAME>"  # replace with your endpoint name


class BotoClient:
    def __init__(self, host):
        self.sagemaker_client = boto3.client("sagemaker-runtime", region_name=region)

        # Load list of sample images
        self.sample_images_dir = "sample_images"
        self.sample_images = [
            os.path.join(self.sample_images_dir, f)
            for f in os.listdir(self.sample_images_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        if not self.sample_images:
            raise ValueError(f"No images found in {self.sample_images_dir}")
        logger.info(f"Loaded {len(self.sample_images)} sample images for testing")

    def resize_long_side(self, image: Image.Image, max_size: int = 560) -> Image.Image:
        w, h = image.size
        long_side = max(w, h)
        if long_side <= max_size:
            return image  # no upscaling
        scale = max_size / float(long_side)
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def send(self):

        request_meta = {
            "request_type": "InvokeEndpoint",
            "name": "SageMaker",
            "start_time": time.time(),
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        start_perf_counter = time.perf_counter()

        try:
            # Prepare image payload - randomly select an image
            selected_image = random.choice(self.sample_images)
            orig_image = Image.open(selected_image)
            logger.debug(f"Selected image: {selected_image}")

            # Downscale client-side: long side = 560, keep aspect ratio
            send_image = self.resize_long_side(orig_image, IMG_SIZE)

            buffer = BytesIO()
            send_image.save(buffer, format="JPEG", quality=90)
            image_bytes = buffer.getvalue()

            # Randomly select parameters if configured
            confidence = None
            if CONFIDENCE_THRESHOLDS:
                confidence = random.choice(CONFIDENCE_THRESHOLDS)

            classes = None
            if CLASS_FILTERS:
                classes = random.choice(CLASS_FILTERS)

            max_detections = None
            if MAX_DETECTIONS_OPTIONS:
                max_detections = random.choice(MAX_DETECTIONS_OPTIONS)

            min_box_area = None
            if MIN_BOX_AREA_OPTIONS:
                min_box_area = random.choice(MIN_BOX_AREA_OPTIONS)

            # Create JSON payload with base64-encoded image and optional parameters
            payload_dict = {"image": base64.b64encode(image_bytes).decode("utf-8")}
            if confidence is not None:
                payload_dict["confidence"] = confidence
            if classes is not None:
                payload_dict["classes"] = classes
            if max_detections is not None:
                payload_dict["max_detections"] = max_detections
            if min_box_area is not None:
                payload_dict["min_box_area"] = min_box_area

            payload = json.dumps(payload_dict)

            # Invoke SageMaker endpoint directly via boto3 runtime client
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=payload,
                ContentType="application/json",
                Accept="application/json",
            )
            response_body = response["Body"].read()
            # Some endpoints return raw JSON, others bytes; handle both
            try:
                result = json.loads(response_body.decode("utf-8"))
            except Exception:
                result = json.loads(response_body)

            # Track response metrics
            request_meta["response_length"] = len(response_body)

            # Extract detection count from response
            detection_count = 0
            if isinstance(result, dict):
                if "metadata" in result and "count" in result["metadata"]:
                    detection_count = result["metadata"]["count"]
                elif "detections" in result:
                    detection_count = len(result["detections"])

            # Format parameters for logging
            confidence_str = (
                f"{confidence:.2f}" if confidence is not None else "default"
            )
            classes_str = f"{classes}" if classes is not None else "all"
            max_det_str = (
                f"{max_detections}" if max_detections is not None else "unlimited"
            )
            min_area_str = f"{min_box_area}px²" if min_box_area is not None else "none"

            # Check for applied filters in response
            applied_filters = result.get("metadata", {}).get("applied_filters", {})
            filtered_info = ""
            if applied_filters:
                if "total_before_limit" in applied_filters:
                    filtered_info = (
                        f" (filtered from {applied_filters['total_before_limit']})"
                    )

            logger.info(
                "Image: %s | Conf: %s | Classes: %s | MaxDet: %s | MinArea: %s | "
                "Response: %d bytes, %d detections%s, inference: %s ms",
                selected_image,
                confidence_str,
                classes_str,
                max_det_str,
                min_area_str,
                request_meta["response_length"],
                detection_count,
                filtered_info,
                result.get("metadata", {}).get("inference_time_ms", "N/A"),
            )
        except Exception as e:
            logger.error(e)
            request_meta["exception"] = e

        end_perf_counter = time.perf_counter()
        request_meta["response_time"] = (end_perf_counter - start_perf_counter) * 1000

        logger.debug(start_perf_counter)
        logger.debug(end_perf_counter)
        logger.info(request_meta["response_time"])

        events.request.fire(**request_meta)


class BotoUser(FastHttpUser):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = BotoClient(self.host)


class MyUser(BotoUser):
    @task
    def send_request(self):
        self.client.send()
