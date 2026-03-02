from __future__ import absolute_import, division, print_function

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from networks.models import make  # 建议用 make，而不是 import *

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SSH/headless environments
import matplotlib.pyplot as plt
from PIL import Image

def colorize_depth_fixed(depth_u8: np.ndarray, cmap: str = "Spectral") -> np.ndarray:
    """
    depth_u8: uint8, 0~255
    return: RGB uint8
    """
    disp = depth_u8.astype(np.float32) / 255.0
    colored = matplotlib.colormaps[cmap](disp)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    return np.ascontiguousarray(colored)

def ensure_dir_for_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_model(config, device_id=0):
    model_path = os.path.join(config["load_weights_dir"], "model.pth")
    print(f"🔹 Loading model weights from: {model_path}")

    # Determine device - use GPU if available, otherwise CPU
    device = "cpu"
    if torch.cuda.is_available():
        try:
            # Use the specified GPU or default to GPU 0
            gpu_to_use = int(device_id) if device_id is not None else 0
            
            if gpu_to_use < torch.cuda.device_count():
                # Try to use the GPU - PyTorch will handle compatibility
                try:
                    # Test if we can create a tensor on this GPU
                    test_tensor = torch.zeros(1).to(f"cuda:{gpu_to_use}")
                    device = f"cuda:{gpu_to_use}"
                    gpu_name = torch.cuda.get_device_name(gpu_to_use)
                    cap = torch.cuda.get_device_capability(gpu_to_use)
                    print(f"✅ Using GPU {gpu_to_use}: {gpu_name} (compute capability {cap[0]}.{cap[1]})")
                    del test_tensor
                except RuntimeError as e:
                    # If GPU fails, fall back to CPU
                    print(f"⚠️ Cannot use GPU {gpu_to_use}: {e}")
                    print("⚠️ Falling back to CPU")
                    device = "cpu"
            else:
                print(f"⚠️ GPU {gpu_to_use} not available (only {torch.cuda.device_count()} GPUs found), using CPU")
        except Exception as e:
            device = "cpu"
            print(f"⚠️ CUDA error detected: {e}, using CPU")
    else:
        device = "cpu"
        print("⚠️ CUDA not available, using CPU")
    
    print(f"🔹 Using device: {device}")
    state = torch.load(model_path, map_location=device)

    m = make(config["model"])
    # Apply DataParallel if weights have "module." prefix (from multi-GPU training)
    # This must be done BEFORE moving to device and loading state dict
    has_module_prefix = any(k.startswith("module.") for k in state.keys())
    if has_module_prefix:
        m = nn.DataParallel(m)
        print("🔹 Model weights have 'module.' prefix, wrapping with DataParallel")

    m = m.to(device)
    m_state = m.state_dict()
    
    # Count how many keys match before filtering
    state_keys = set(state.keys())
    model_keys = set(m_state.keys())
    matching_keys = state_keys & model_keys
    
    # If weights have module prefix but model doesn't, strip the prefix
    if has_module_prefix and not isinstance(m, nn.DataParallel):
        # This shouldn't happen with our logic above, but handle it
        state = {k.replace("module.", ""): v for k, v in state.items()}
        state_keys = set(state.keys())
        matching_keys = state_keys & model_keys
    
    if len(matching_keys) == 0:
        print(f"❌ ERROR: No matching keys between model and state dict!")
        print(f"   State dict has {len(state_keys)} keys (sample: {list(state_keys)[:3]})")
        print(f"   Model has {len(model_keys)} keys (sample: {list(model_keys)[:3]})")
        raise RuntimeError("Model weights do not match model architecture. Check if DataParallel wrapping is correct.")
    
    loaded_state = {k: v for k, v in state.items() if k in m_state}
    print(f"🔹 Loading {len(loaded_state)}/{len(state_keys)} weights into model")
    
    missing_keys, unexpected_keys = m.load_state_dict(loaded_state, strict=False)
    if missing_keys and len(missing_keys) < 20:
        print(f"⚠️ Warning: Missing keys: {list(missing_keys)[:5]}...")
    elif missing_keys:
        print(f"⚠️ Warning: {len(missing_keys)} keys not found in model (this may be normal)")
    if unexpected_keys and len(unexpected_keys) < 20:
        print(f"⚠️ Warning: Unexpected keys: {list(unexpected_keys)[:5]}...")
    elif unexpected_keys:
        print(f"⚠️ Warning: {len(unexpected_keys)} unexpected keys in state dict")
    m.eval()

    print("✅ Model loaded successfully.\n")
    return m, device

def infer_raw(model, device, img_bgr_u8: np.ndarray, input_size=256, max_depth=1.0) -> np.ndarray:
    """
    img_bgr_u8: HWC uint8 BGR (as read by cv2.imread)
    return: pred float32 (H,W) - depth map at original image resolution
    """
    # Use the model's built-in preprocessing which handles:
    # - Resize to 2:1 aspect ratio (width=input_size*2, height=input_size)
    # - Normalization with ImageNet stats
    # - Proper tensor preparation
    h_orig, w_orig = img_bgr_u8.shape[:2]
    
    # Get the actual model (unwrap DataParallel if needed)
    actual_model = model.module if hasattr(model, 'module') else model
    
    # The model's image2tensor expects BGR input
    tensor, (h_orig, w_orig) = actual_model.image2tensor(img_bgr_u8, input_size=input_size)
    tensor = tensor.to(device)

    with torch.inference_mode():
        outputs = model(tensor)

        if isinstance(outputs, dict) and "pred_depth" in outputs:
            pred_depth = outputs["pred_depth"]
            # Interpolate back to original image size
            pred_depth = F.interpolate(pred_depth, (h_orig, w_orig), mode="bilinear", align_corners=True)
            pred = pred_depth[0, 0].detach().cpu().numpy()
            
            # Apply mask if available
            if "pred_mask" in outputs:
                mask = 1 - outputs["pred_mask"]
                mask = mask > 0.5
                mask_resized = F.interpolate(mask.float(), (h_orig, w_orig), mode="nearest")
                mask_resized = mask_resized[0, 0].detach().cpu().numpy() > 0.5
                pred[~mask_resized] = max_depth  # Set masked areas to max depth
        else:
            # Fallback for models that don't return dict
            pred = outputs[0].detach().cpu().squeeze().numpy()
            if pred.ndim == 2:
                # Resize if needed
                pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
                pred_tensor = F.interpolate(pred_tensor, (h_orig, w_orig), mode="bilinear", align_corners=True)
                pred = pred_tensor[0, 0].numpy()

    return pred.astype(np.float32)

def pred_to_vis(pred: np.ndarray, vis_range: str = "100m", cmap: str = "Spectral"):
    """
    return:
      depth_gray_u8: (H,W) uint8
      depth_color_rgb: (H,W,3) uint8 RGB
    """
    if vis_range == "100m":
        pred_clip = np.clip(pred, 0.0, 1.0)
        depth_gray = (pred_clip * 255).astype(np.uint8)
    elif vis_range == "10m":
        pred_clip = np.clip(pred, 0.0, 0.1)
        depth_gray = (pred_clip * 10.0 * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown vis_range: {vis_range} (use '100m' or '10m')")

    depth_color = colorize_depth_fixed(depth_gray, cmap=cmap)
    return depth_gray, depth_color

def create_example_plot(img_path: str, depth_npy_path: str, out_root: str, max_depth: float = 1.0):
    """
    Create a side-by-side plot showing the original image and depth map with colorbar.
    Saves to test_output/example_plot.jpg
    """
    try:
        # Load the image
        img = np.array(Image.open(img_path))
        
        # Load the depth map
        depth = np.load(depth_npy_path)
        
        # Convert normalized depth (0-1) to meters (assuming max_depth=1.0 means 100m)
        max_depth_meters = 100.0
        depth_meters = depth * max_depth_meters
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top subplot: original image
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        # Bottom subplot: depth map with colorbar
        im = ax2.imshow(depth_meters, cmap='Spectral', vmin=0, vmax=max_depth_meters)
        ax2.set_title('Depth Map (Distance from Camera)', fontsize=14)
        ax2.axis('off')
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Distance (meters)', fontsize=12)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(out_root, 'example_plot.jpg')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Example plot saved to {plot_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not create example plot: {e}")

def infer_and_save(model, device, img_path, out_root, idx, vis_range="100m", cmap="Spectral", input_size=256, max_depth=1.0):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠️ Cannot read image: {img_path}")
        return

    # Check if image is a panorama (should have 2:1 aspect ratio)
    h, w = img_bgr.shape[:2]
    aspect_ratio = w / h
    if abs(aspect_ratio - 2.0) > 0.1:  # Allow 10% tolerance
        print(f"⚠️ Warning: Image {img_path} has aspect ratio {aspect_ratio:.2f}, expected ~2.0 (panorama format)")
        print(f"   Image size: {w}x{h}. The model expects equirectangular panoramas with 2:1 aspect ratio.")

    # Pass BGR image directly (model's image2tensor expects BGR)
    pred = infer_raw(model, device, img_bgr, input_size=input_size, max_depth=max_depth)

    # Normalize depth for visualization (pred_to_vis expects [0, 1] range)
    # The raw pred values are saved as-is in .npy file
    pred_normalized = pred / max_depth if max_depth > 0 else pred
    depth_gray, depth_color_rgb = pred_to_vis(pred_normalized, vis_range=vis_range, cmap=cmap)

    filename = f"{idx:06d}"

    pred_npy_path = os.path.join(out_root, "depth_npy", filename + ".npy")
    gray_png_path = os.path.join(out_root, f"depth_vis_gray_{vis_range}", filename + ".png")
    color_png_path = os.path.join(out_root, f"depth_vis_color_{vis_range}", filename + ".png")

    ensure_dir_for_file(pred_npy_path)
    ensure_dir_for_file(gray_png_path)
    ensure_dir_for_file(color_png_path)

    np.save(pred_npy_path, pred)

    cv2.imwrite(gray_png_path, depth_gray)

    cv2.imwrite(color_png_path, cv2.cvtColor(depth_color_rgb, cv2.COLOR_RGB2BGR))
    
    # Generate example plot for the first image
    if idx == 1:
        create_example_plot(img_path, pred_npy_path, out_root, max_depth)


def main(config_path, txt_path, out_root, vis_range="100m", cmap="Spectral", input_size=256, gpu_id=0):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print("✅ Config loaded.")

    model, device = load_model(config, device_id=gpu_id)
    
    # Get input size and max_depth from config if available
    if "input" in config:
        input_size = config["input"].get("height", input_size)
    
    max_depth = config.get("model", {}).get("args", {}).get("max_depth", 1.0)

    with open(txt_path, "r") as f:
        img_list = [l.strip() for l in f.readlines() if l.strip()]

    print(f"🔹 Total images to infer: {len(img_list)}")
    print(f"🔹 Visualization: {vis_range}, colormap: {cmap}")
    print(f"🔹 Input size (height): {input_size} (panoramas will be resized to {input_size}x{input_size*2})")
    print(f"🔹 Max depth: {max_depth}\n")

    for idx, img_path in enumerate(tqdm(img_list, desc="Inferencing"), start=1):
        infer_and_save(model, device, img_path, out_root, idx, vis_range=vis_range, cmap=cmap, input_size=input_size, max_depth=max_depth)

    print("\n🎯 推理完成！")
    print(f"   depth npy: {out_root}/depth_npy")
    print(f"   depth gray png: {out_root}/depth_vis_gray_{vis_range}")
    print(f"   depth color png: {out_root}/depth_vis_color_{vis_range}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/infer.yaml")
    parser.add_argument("--txt", default="datasets/test.txt")
    parser.add_argument("--output", default="test_output")
    parser.add_argument("--gpu", default="0", help="使用的GPU编号")

    parser.add_argument("--vis", default="100m", choices=["100m", "10m"], help="可视化范围（只影响png，不影响npy）")
    parser.add_argument("--cmap", default="Spectral", help="matplotlib colormap name, e.g. Spectral, Turbo, Viridis")
    parser.add_argument("--input_size", type=int, default=256, help="Input height for preprocessing (width will be 2x this for 2:1 aspect ratio)")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # After setting CUDA_VISIBLE_DEVICES, GPU indices are remapped, so use 0
    main(args.config, args.txt, args.output, vis_range=args.vis, cmap=args.cmap, input_size=args.input_size, gpu_id=0)
