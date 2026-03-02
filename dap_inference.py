"""
Standalone inference module for DAP (Depth Anything Panorama) model.

This module provides a clean API for running depth inference on panoramic images
and can be easily integrated into other repositories.

Usage:
    from dap_inference import DAPInference
    
    inferencer = DAPInference(
        load_weights_dir="./DAP-weights",
        config={
            "model": {
                "name": "dap",
                "args": {
                    "midas_model_type": "vitl",
                    "fine_tune_type": "hypersim",
                    "max_depth": 1.0,
                    "min_depth": 0.01,
                    "train_decoder": True
                }
            }
        },
        device="cuda:0"
    )
    
    image_bgr = cv2.imread("panorama.jpg")
    depth_map = inferencer.predict(image_bgr)
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Union, Tuple, Optional
from tqdm import tqdm

# Add dap directory to Python path for absolute imports to work
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# These imports would need to be available in the target repo
try:
    from .networks.models import make
except ImportError:
    # Fallback for when running as script directly
    from networks.models import make


class DAPInference:
    """
    Standalone inference class for DAP (Depth Anything Panorama) model.
    
    This class encapsulates model loading and inference functionality,
    providing a clean API for depth prediction on panoramic images.
    """
    
    def __init__(
        self,
        load_weights_dir: str,
        config: dict,
        device: Optional[Union[str, int]] = None,
        input_size: int = 256
    ):
        """
        Initialize DAP inference engine.
        
        Args:
            load_weights_dir: Directory containing model.pth checkpoint file
            config: Model configuration dict (same format as YAML)
            device: Device to use ("cuda:0", "cpu", or None for auto-detect, or int for GPU ID)
            input_size: Input height for preprocessing (width will be 2x for 2:1 aspect ratio)
        """
        self.load_weights_dir = load_weights_dir
        self.config = config
        self.input_size = input_size
        self.device = self._setup_device(device)
        self.model = None
        self.max_depth = config.get("model", {}).get("args", {}).get("max_depth", 1.0)
        
        self._load_model()
    
    def _setup_device(self, device: Optional[Union[str, int]]) -> str:
        """Setup and validate device."""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        elif isinstance(device, int):
            # Integer device ID - treat as GPU ID
            if torch.cuda.is_available():
                gpu_to_use = device
                if gpu_to_use < torch.cuda.device_count():
                    try:
                        # Test if we can create a tensor on this GPU
                        test_tensor = torch.zeros(1).to(f"cuda:{gpu_to_use}")
                        device = f"cuda:{gpu_to_use}"
                        gpu_name = torch.cuda.get_device_name(gpu_to_use)
                        cap = torch.cuda.get_device_capability(gpu_to_use)
                        print(f"✅ Using GPU {gpu_to_use}: {gpu_name} (compute capability {cap[0]}.{cap[1]})")
                        del test_tensor
                    except RuntimeError as e:
                        print(f"⚠️ Cannot use GPU {gpu_to_use}: {e}")
                        print("⚠️ Falling back to CPU")
                        device = "cpu"
                else:
                    print(f"⚠️ GPU {gpu_to_use} not available (only {torch.cuda.device_count()} GPUs found), using CPU")
                    device = "cpu"
            else:
                device = "cpu"
        elif isinstance(device, str):
            # String device like "cuda:0" or "cpu"
            if device.startswith("cuda:"):
                gpu_id = int(device.split(":")[1])
                if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    device = device
                else:
                    print(f"⚠️ {device} not available, using CPU")
                    device = "cpu"
            elif device == "cpu":
                device = "cpu"
            else:
                print(f"⚠️ Unknown device {device}, using CPU")
                device = "cpu"
        
        if device == "cpu":
            print("⚠️ CUDA not available, using CPU")
        else:
            print(f"🔹 Using device: {device}")
        
        return device
    
    def _load_model(self):
        """Load model from checkpoint."""
        model_path = os.path.join(self.load_weights_dir, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
        print(f"🔹 Loading model weights from: {model_path}")
        
        state = torch.load(model_path, map_location=self.device)
        
        # Create model
        m = make(self.config["model"])
        
        # Handle DataParallel wrapping
        has_module_prefix = any(k.startswith("module.") for k in state.keys())
        if has_module_prefix:
            m = nn.DataParallel(m)
            print("🔹 Model weights have 'module.' prefix, wrapping with DataParallel")
        
        m = m.to(self.device)
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
        
        # Filter and load state dict
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
        self.model = m
        print("✅ Model loaded successfully.\n")
    
    def predict(
        self,
        image_bgr: np.ndarray,
        return_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on a single image.
        
        Args:
            image_bgr: Input image as BGR numpy array (H, W, 3) uint8
            return_mask: If True, also return the mask prediction
            
        Returns:
            depth_map: Depth prediction as float32 array (H, W)
            mask (optional): Mask prediction as bool array (H, W) if return_mask=True
        """
        h_orig, w_orig = image_bgr.shape[:2]
        
        # Get the actual model (unwrap DataParallel if needed)
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Preprocess image
        tensor, (h_orig, w_orig) = actual_model.image2tensor(
            image_bgr, 
            input_size=self.input_size
        )
        tensor = tensor.to(self.device)
        
        # Run inference
        with torch.inference_mode():
            outputs = self.model(tensor)
            
            if isinstance(outputs, dict) and "pred_depth" in outputs:
                pred_depth = outputs["pred_depth"]
                pred_depth = F.interpolate(
                    pred_depth, 
                    (h_orig, w_orig), 
                    mode="bilinear", 
                    align_corners=True
                )
                depth_map = pred_depth[0, 0].detach().cpu().numpy().astype(np.float32)
                
                # Apply mask if available
                if "pred_mask" in outputs:
                    mask = 1 - outputs["pred_mask"]
                    mask = mask > 0.5
                    mask_resized = F.interpolate(
                        mask.float(), 
                        (h_orig, w_orig), 
                        mode="nearest"
                    )
                    mask_resized = mask_resized[0, 0].detach().cpu().numpy() > 0.5
                    depth_map[~mask_resized] = self.max_depth
                    
                    if return_mask:
                        return depth_map, mask_resized
            else:
                # Fallback for models that don't return dict
                pred = outputs[0].detach().cpu().squeeze().numpy()
                if pred.ndim == 2:
                    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
                    pred_tensor = F.interpolate(
                        pred_tensor, 
                        (h_orig, w_orig), 
                        mode="bilinear", 
                        align_corners=True
                    )
                    depth_map = pred_tensor[0, 0].numpy().astype(np.float32)
                else:
                    depth_map = pred.astype(np.float32)
        
        if return_mask:
            return depth_map, np.ones_like(depth_map, dtype=bool)
        return depth_map


def load_dap_inference(
    load_weights_dir: str,
    config_path: Optional[str] = None,
    config_dict: Optional[dict] = None,
    device: Optional[Union[str, int]] = None,
    input_size: int = 256
) -> DAPInference:
    """
    Convenience function to load DAP inference engine.
    
    Args:
        load_weights_dir: Directory containing model.pth
        config_path: Path to YAML config file (optional)
        config_dict: Config dict (optional, alternative to config_path)
        device: Device to use
        input_size: Input size for preprocessing
        
    Returns:
        DAPInference instance
    """
    import yaml
    
    if config_dict is None:
        if config_path is None:
            raise ValueError("Either config_path or config_dict must be provided")
        with open(config_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    return DAPInference(
        load_weights_dir=load_weights_dir,
        config=config_dict,
        device=device,
        input_size=input_size
    )


# ============================================================================
# Sample usage code
# ============================================================================

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="DAP Inference Sample")
    parser.add_argument("--config", default="infer.yaml", help="Path to config YAML file")
    parser.add_argument("--image", default=os.path.expanduser("~/roadside_vision/dap/panorama.jpg"), help="Path to input panorama image")
    parser.add_argument("--output", default="test_output", help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--input_size", type=int, default=1024, help="Input height for preprocessing")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create inference engine
    print("=" * 60)
    print("Initializing DAP Inference Engine")
    print("=" * 60)
    inferencer = DAPInference(
        load_weights_dir=config["load_weights_dir"],
        config=config,
        device=args.gpu,
        input_size=args.input_size
    )
    
    # Load image
    print(f"\nLoading image: {args.image}")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    
    # aspect ration  must be 2:1
    h, w = image_bgr.shape[:2]
    aspect_ratio = w / h
    assert aspect_ratio == 2.0, f"Image must have aspect ratio 2:1, but has {aspect_ratio:.2f}"
    
    # Run inference
    print(f"\nRunning inference...")
    for tqdm in tqdm(range(100)):
        depth_map = inferencer.predict(image_bgr)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    depth_map_path = os.path.join(args.output, "depth_map.npy")
    np.save(depth_map_path, depth_map)
    print(f"✅ Saved depth map: {depth_map_path}")
