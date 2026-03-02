# DAP – Depth Any Panoramas

Minimal inference-only implementation of [Depth Any Panoramas](https://github.com/Insta360-Research-Team/DAP) (DAP), a foundation model for panoramic depth estimation. This stripped-down version runs faster by skipping visualization and focuses on depth map output.

## What This Is

DAP estimates depth from equirectangular (360°) panorama images. Input: a 2:1 aspect ratio panorama. Output: a depth map (`.npy`) with normalized depth values (0–1, multiply by `max_depth` for meters).

This repo contains only the inference code: `dap_inference.py`, the model networks, and the DINOv3/depth backbone. No training, evaluation, or visualization tooling.

## How It Works

1. **Model**: DINOv3 encoder + Depth Anything V2 decoder, fine-tuned for panoramas.
2. **Input**: BGR image (e.g. from `cv2.imread`), 2:1 aspect ratio.
3. **Output**: Float32 depth map `(H, W)` saved as `test_output/depth_map.npy`.

The `DAPInference` class loads the model from a checkpoint, preprocesses the image via `image2tensor`, runs inference, and returns the depth array.

## Installation (venv)

```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate   # Windows

# Install PyTorch (choose CUDA or CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# pip install torch torchvision   # CPU only

# Install dependencies
pip install -r requirements.txt
```

If `requirements.txt` is missing, install: `opencv-python`, `numpy`, `torch`, `torchvision`, `einops`, `PyYAML`, `tqdm`.

## Model Weights

Download the pretrained weights from [Hugging Face](https://huggingface.co/Insta360-Research/DAP-weights):

```bash
# Option 1: git
git clone https://huggingface.co/Insta360-Research/DAP-weights

# Option 2: huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Insta360-Research/DAP-weights', local_dir='./DAP-weights')"
```

Ensure `model.pth` is at `DAP-weights/model.pth` (or set `load_weights_dir` in `infer.yaml` to your path).

## How to Run

From the repo root:

```bash
python dap_inference.py
```

Or with an explicit venv and paths:

```bash
/path/to/venv/bin/python /path/to/DAP/dap_inference.py
```

**Example** (using venv and default image `~/roadside_vision/dap/panorama.jpg`):

```bash
~/.venv-310/bin/python ~/roadside_vision/dap/dap_inference.py
```

**Arguments:**
- `--config` – Path to YAML config (default: `infer.yaml`)
- `--image` – Input panorama (default: `~/roadside_vision/dap/panorama.jpg`)
- `--output` – Output directory (default: `test_output`)
- `--gpu` – GPU ID (default: `0`)
- `--input_size` – Input height for preprocessing (default: `1024`)

**Example output:**

```
xFormers not available
xFormers not available
============================================================
Initializing DAP Inference Engine
============================================================
✅ Using GPU 0: NVIDIA GeForce RTX 5090 (compute capability 12.0)
🔹 Using device: cuda:0
🔹 Loading model weights from: /mnt/projects/roadside_vision/models/DAP-weights/model.pth
🔹 Model weights have 'module.' prefix, wrapping with DataParallel
🔹 Loading 858/859 weights into model
✅ Model loaded successfully.

Loading image: /home/mkolar/roadside_vision/dap/panorama.jpg

Running inference...
100%|████████████████████████████████████████████████████████| 100/100 [01:04<00:00,  1.55it/s]
✅ Saved depth map: test_output/depth_map.npy
```

## Inference as a Module

```python
from dap_inference import DAPInference
import yaml

with open("infer.yaml") as f:
    config = yaml.safe_load(f)

inferencer = DAPInference(
    load_weights_dir=config["load_weights_dir"],
    config=config,
    device=0,
    input_size=1024
)

import cv2
image_bgr = cv2.imread("panorama.jpg")
depth_map = inferencer.predict(image_bgr)
# depth_map: (H, W) float32
```

## Input Requirements

- Equirectangular panorama, **2:1 aspect ratio** (width = 2 × height)
- Common sizes: 1024×512, 2048×1024, 4096×2048
- Formats: PNG, JPG, JPEG

## Citation

```bibtex
@article{lin2025dap,
  title={Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation},
  author={Lin, Xin and Song, Meixi and Zhang, Dizhe and Lu, Wenxuan and Li, Haodong and Du, Bo and Yang, Ming-Hsuan and Nguyen, Truong and Qi, Lu},
  journal={arXiv},
  year={2025}
}
```

## Acknowledgement

- [Insta360-Research-Team/DAP](https://github.com/Insta360-Research-Team/DAP)
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [PanDA](https://github.com/Insta360-Research-Team/PanDA)
