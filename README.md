<h1 align="center">
Depth Any Panoramas:<br>
A Foundation Model for Panoramic Depth Estimation
</h1>


<p align="center">
  <a href="https://linxin0.github.io"><b>Xin Lin</b></a> ·
  <a href="#"><b>Meixi Song</b></a> ·
  <a href="#"><b>Dizhe Zhang</b></a> ·
  <a href="#"><b>Wenxuan Lu</b></a> ·
  <a href="https://haodong2000.github.io"><b>Haodong Li</b></a>
  <br>
  <a href="#"><b>Bo Du</b></a> ·
  <a href="#"><b>Ming-Hsuan Yang</b></a> ·
  <a href="#"><b>Truong Nguyen</b></a> ·
  <a href="http://luqi.info"><b>Lu Qi</b></a>
</p>


<p align="center">
  <a href='https://arxiv.org/abs/2512.16913'><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://insta360-research-team.github.io/DAP_website/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=insta360&logoColor=white' alt='Project Page'></a>
  <a href=''><img src='https://img.shields.io/badge/%F0%9F%93%88%20Hugging%20Face-Dataset-yellow'></a>
  <a href='https://huggingface.co/spaces/Insta360-Research/DAP'><img src='https://img.shields.io/badge/🚀%20Hugging%20Face-Demo-orange'></a>
</p>

![teaser](assets/depth_teaser2_00.png)



## 🔨 Installation

### Step 1: Clone the repository

```Bash
git clone https://github.com/Insta360-Research-Team/DAP
cd DAP
```

### Step 2: Set up Python environment

Create a conda environment (recommended):

```Bash
conda create -n dap python=3.12
conda activate dap
```

Or use your existing Python environment.

### Step 3: Install PyTorch

Install PyTorch with CUDA support (for GPU) or CPU-only version:

**For CUDA (GPU):**
```Bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```Bash
pip install torch==2.7.1 torchvision==0.22.1
```

**Note:** For newer GPUs (e.g., RTX 5090 with compute capability 12.0), you may need PyTorch 2.9+ with CUDA 12.8+ support:
```Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Step 4: Install dependencies

```Bash
pip install -r requirements.txt
```

## 🤝 Pre-trained Model

Download the pretrained model weights from Hugging Face:

```Bash
# Option 1: Using git (recommended)
git clone https://huggingface.co/Insta360-Research/DAP-weights

# Option 2: Using huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Insta360-Research/DAP-weights', local_dir='./DAP-weights')"
```

**Note:** After downloading, ensure the `model.pth` file (~1.4GB) is located at `./DAP-weights/model.pth` (relative to the repository root).

## 📒 Quick Start: Run Inference on Demo Image

We provide a demo equirectangular panorama image (`assets/schloss_celle.jpg`) that you can use to test the model. This image is **open-licensed** and demonstrates the depth estimation capabilities. The image shows Schloss Celle (Celle Castle) and is included in the repository for easy testing.

### Step 1: Prepare the input file

The `datasets/test.txt` file already points to the demo image. You can verify it contains:
```
assets/schloss_celle.jpg
```

Or create your own text file with paths to panorama images (one per line).

### Step 2: Run inference

**Basic usage (uses default config and demo image):**
```Bash
python test/infer.py --config config/infer.yaml --txt datasets/test.txt --output test_output --gpu 0
```

**With custom options:**
```Bash
python test/infer.py \
    --config config/infer.yaml \
    --txt datasets/test.txt \
    --output test_output \
    --gpu 0 \
    --vis 100m \
    --cmap Spectral \
    --input_size 512
```

**Arguments:**
- `--config`: Path to config file (default: `config/infer.yaml`)
- `--txt`: Text file with image paths, one per line (default: `datasets/test.txt`)
- `--output`: Output directory (default: `test_output`)
- `--gpu`: GPU ID to use (default: `0`, use `-1` or ensure CUDA is unavailable for CPU)
- `--vis`: Visualization range - `100m` or `10m` (default: `100m`)
- `--cmap`: Matplotlib colormap name, e.g., `Spectral`, `Turbo`, `Viridis` (default: `Spectral`)
- `--input_size`: Input height for preprocessing (width will be 2x for 2:1 aspect ratio, default: `256`, config uses `512`)

### Step 3: Check the results

After inference completes, you'll find the outputs in the specified output directory:

- **`depth_npy/`**: Raw depth predictions as NumPy arrays (`.npy` files)
- **`depth_vis_gray_100m/`**: Grayscale depth visualizations (`.png` files)
- **`depth_vis_color_100m/`**: Colorized depth visualizations (`.png` files)
- **`example_plot.jpg`**: Side-by-side comparison of input image and depth map (generated for the first image)

### Input Requirements

The model expects **equirectangular panorama images** with:
- **2:1 aspect ratio** (width = 2 × height)
- Common sizes: 1024×512, 2048×1024, 4096×2048, etc.
- Standard image formats: PNG, JPG, JPEG

The model will automatically resize images while preserving the aspect ratio.

## 🖼️ Dataset

The training dataset will be open soon.


## 🚀 Evaluation

For evaluation on benchmark datasets:

```Bash
python test/eval.py --config config/test.yaml
```

## 📝 Example Output

When you run inference on `assets/schloss_celle.jpg`, you should see:

1. **Console output** showing:
   - Model loading progress
   - GPU/CPU device selection
   - Inference progress bar
   - Completion message with output paths

2. **Generated files** in `test_output/`:
   - `depth_npy/000001.npy` - Raw depth values (0-1 normalized, multiply by max_depth for meters)
   - `depth_vis_gray_100m/000001.png` - Grayscale depth visualization
   - `depth_vis_color_100m/000001.png` - Colorized depth map
   - `example_plot.jpg` - Side-by-side comparison plot

## 🔧 Troubleshooting

### GPU not detected
- Ensure PyTorch was installed with CUDA support
- Check GPU compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
- For newer GPUs (compute capability >= 12.0), use PyTorch 2.9+ with CUDA 12.8+

### Model weights not found
- Ensure `DAP-weights/model.pth` exists
- Download from: https://huggingface.co/Insta360-Research/DAP-weights

### Wrong aspect ratio warning
- Input images must be equirectangular panoramas with 2:1 aspect ratio
- The model will still process them but results may be suboptimal

### Out of memory errors
- Reduce `--input_size` (e.g., use `256` instead of `512`)
- Process fewer images at a time




## 🤝 Acknowledgement

We appreciate the open source of the following projects:

* [PanDA](https://caozidong.github.io/PanDA_Depth/)
* [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)


## Citation
```
@article{lin2025dap,
          title={Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation},
          author={Lin, Xin and Song, Meixi and Zhang, Dizhe and Lu, Wenxuan and Li, Haodong and Du, Bo and Yang, Ming-Hsuan and Nguyen, Truong and Qi, Lu},
          journal={arXiv},
          year={2025}
        }
```

