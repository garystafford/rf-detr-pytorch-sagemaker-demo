import logging
import json
import time
from io import BytesIO

import boto3
from locust import task, events
from locust.contrib.fasthttp import FastHttpUser
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

IMG_SIZE = 56*10  # max image dimension for inference

region = "us-east-1"
endpoint_name = "<YOUR_SAGEMAKER_ENDPOINT_NAME>"  # replace with your endpoint name


class BotoClient:
    def __init__(self):
        self.sagemaker_client = boto3.client("sagemaker-runtime", region_name=region)

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
            # Prepare image payload
            orig_image = Image.open("sample_image_00005.jpg")

            # Downscale client-side: long side = 640, keep aspect ratio
            send_image = self.resize_long_side(orig_image, IMG_SIZE)

            buffer = BytesIO()
            send_image.save(buffer, format="JPEG", quality=90)
            payload = buffer.getvalue()

            # Invoke SageMaker endpoint directly via boto3 runtime client
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=payload,
                ContentType="image/jpeg",
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

            logger.info(
                "Response: %d bytes, %d detections, inference_time: %s ms",
                request_meta["response_length"],
                detection_count,
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
