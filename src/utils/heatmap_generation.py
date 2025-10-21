import numpy as np
from typing import List, Tuple

# --- CONSTANTS (Must match dino_pipeline.py) ---
NUM_KEYPOINTS = 14
HEAD_FINAL_RES = 64 # Output heatmap resolution (e.g., 64x64)
CROP_SIZE = 256 # Fixed input size for the cropped image
SIGMA = 2 # Standard deviation for Gaussian kernel


def generate_single_heatmap(center_x: float, center_y: float, 
                            H: int, W: int, sigma: float) -> np.ndarray:
    """
    Generates a single 2D Gaussian heatmap.
    
    Args:
        center_x, center_y: Normalized center coordinates (0 to W-1, 0 to H-1)
        H, W: Height and Width of the output map
    """
    center_x, center_y = int(center_x), int(center_y)
    
    # Create meshgrid for coordinates
    y = np.arange(0, H, 1, dtype=np.float32)
    x = np.arange(0, W, 1, dtype=np.float32)
    yy, xx = np.meshgrid(y, x)
    
    # Calculate Gaussian value: exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
    # We use a threshold to zero out low values for efficiency, but here we keep it simple.
    heatmap = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
    
    # Ensure the heatmap maximum is 1.0 (standard for ground truth)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
        
    return heatmap.transpose() # Transpose back to (H, W)


def generate_target_heatmaps(
    keypoints_xyv: np.ndarray, # (14*3) array from COCO
    bbox: np.ndarray,          # (x, y, w, h)
    img_width: int,
    img_height: int,
    target_res: int = HEAD_FINAL_RES,
    crop_size: int = CROP_SIZE
) -> np.ndarray:
    """
    Converts COCO keypoints for a single person into a stack of 14 heatmaps.
    """
    final_heatmaps = np.zeros((NUM_KEYPOINTS, target_res, target_res), dtype=np.float32)
    
    # 1. Normalize keypoints to the CROP_SIZE
    x_min, y_min, w, h = bbox
    
    for i in range(NUM_KEYPOINTS):
        kx = keypoints_xyv[3 * i]
        ky = keypoints_xyv[3 * i + 1]
        v = int(keypoints_xyv[3 * i + 2])
        
        # Skip invisible or unlabeled keypoints
        if v == 0 or kx == 0 or ky == 0:
            continue
            
        # a) Map keypoint from original image space to the Bounding Box Crop space
        # KP_crop = KP_orig - (x_min, y_min)
        kp_crop_x = kx - x_min
        kp_crop_y = ky - y_min
        
        # b) Map keypoint from BBox Crop space to the fixed CROP_SIZE space (e.g., 256x256)
        # KP_fixed = KP_crop * (CROP_SIZE / w)
        kp_fixed_x = kp_crop_x * (crop_size / w)
        kp_fixed_y = kp_crop_y * (crop_size / h)

        # c) Map keypoint from CROP_SIZE to the FINAL HEATMAP RESOLUTION (e.g., 64x64)
        # This is where the downsampling factor is applied (e.g., 64/256 = 0.25)
        scale_factor = target_res / crop_size
        center_x = kp_fixed_x * scale_factor
        center_y = kp_fixed_y * scale_factor
        
        # Generate the Gaussian blob
        heatmap = generate_single_heatmap(
            center_x=center_x,
            center_y=center_y,
            H=target_res,
            W=target_res,
            sigma=SIGMA
        )
        
        # Assign to the final stack
        final_heatmaps[i] = heatmap
        
    return final_heatmaps