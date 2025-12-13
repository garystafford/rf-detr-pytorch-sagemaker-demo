# Real-Time Object Detection with RF-DETR and Amazon SageMaker

Learn how to deploy and performance test the state-of-the-art RF-DETR object detection model to Amazon SageMaker AI using PyTorch for production-ready, real-time inference with GPU acceleration.

![Sample Output](./previews/sample_images_01_detected.jpg)

![Sample Output](./previews/sample_images_04_detected.jpg)

## Features

- Uses pre-trained RF-DETR-Large model with COCO labels (98 classes)
- Creates SageMaker-compatible model artifact
- Deploys real-time endpoint with GPU acceleration (ml.g4dn.xlarge)
- Custom PyTorch inference handler with optimization enabled by default
- Supports JPEG and PNG image formats
- Includes performance testing with [Locust](https://locust.io/)

## Contents

- `deploy_rf_detr.ipynb` - End-to-end deployment notebook
- `code/` - SageMaker model code directory
  - `inference.py` - Custom SageMaker inference handler for RF-DETR
  - `requirements.txt` - Python dependencies
- `rf-detr-large.pth` - Pre-trained RF-DETR-Large model checkpoint
- `sample_images/` - Sample images for testing
- `locust_scripts/` - Performance testing scripts
- `previews/` - Sample detection results

## Inference Pipeline

```text
┌─────────────────┐
│   Image Input   │ (JPEG/PNG)
│   (HTTP POST)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   input_fn()    │ Decode & validate image
│                 │ Convert to RGB
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  predict_fn()   │ RF-DETR 3-stage inference:
│                 │ 1. pre_process()
│                 │ 2. forward()
│                 │ 3. post_process()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  output_fn()    │ Format detection results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JSON Response  │ Bounding boxes, classes,
│   (HTTP 200)    │ confidence scores
└─────────────────┘
```

## Prerequisites

- AWS account with SageMaker access
- Python 3.12+
- Jupyter notebook environment

## Environment Variables

Configure the RF-DETR model using these environment variables when deploying to SageMaker:

- `RFDETR_MODEL` - Model checkpoint filename (default: "rf-detr-large.pth")
- `RFDETR_MODEL_TYPE` - Model variant: rfdetr-nano, rfdetr-small, rfdetr-medium, rfdetr-base, rfdetr-large (default: "rfdetr-large")
- `RFDETR_LABELS` - Label set to use (default: "coco")
- `RFDETR_CONF` - Confidence threshold 0.0-1.0 (default: "0.25")
- `RFDETR_RESOLUTION` - Input resolution, must be divisible by 56 (optional)
- `RFDETR_OPTIMIZE` - Enable inference optimization for 3-5x speedup (default: "true")
- `RFDETR_COMPILE` - Enable torch.jit.trace compilation (default: "true")

## Usage

1. Login to your AWS account on the commandline: `aws login`
2. Create a Python virtual environment (see below)
3. Run the Jupyter notebook [deploy_rf_detr.ipynb](deploy_rf_detr.ipynb)
   - Uses rf-detr-large.pth model checkpoint
   - Packages model artifact with inference code
   - Deploys to SageMaker real-time endpoint
   - Tests object detection on sample images
4. Optionally, use [Locust](https://locust.io/) to load test your endpoint (see [README.md](locust_scripts/README.md))

## Install Requirements (Mac with `pip`)

```bash
brew install python@3.12

python3.12 -m pip install virtualenv --break-system-packages -Uq
python3.12 -m venv .venv
source .venv/bin/activate

python3.12 -m pip install -r ./requirements.txt -Uq
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

The contents of this repository represent my viewpoints and not those of my past or current employers, including Amazon Web Services (AWS). All third-party libraries, modules, plugins, and SDKs are the property of their respective owners.
