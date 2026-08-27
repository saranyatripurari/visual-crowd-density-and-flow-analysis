import os
import numpy as np
import scipy.io as io
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import cv2


def load_gt_points(mat_path: str) -> np.ndarray:
    """
    Parses ShanghaiTech ground-truth .mat annotation file.
    Returns:
        points: (N, 2) numpy array of [x, y] head coordinates.
    """
    if not os.path.exists(mat_path):
        return np.empty((0, 2), dtype=np.float32)

    try:
        mat = io.loadmat(mat_path)
        # ShanghaiTech stores points inside image_info[0,0][0,0][0] or annPoints
        if "image_info" in mat:
            points = mat["image_info"][0, 0][0, 0][0]
        elif "annPoints" in mat:
            points = mat["annPoints"]
        else:
            # Fallback scan for 2D coordinate array
            for k in mat.keys():
                if not k.startswith("__") and isinstance(mat[k], np.ndarray) and mat[k].ndim == 2 and mat[k].shape[1] == 2:
                    points = mat[k]
                    break
            else:
                return np.empty((0, 2), dtype=np.float32)

        return np.asarray(points, dtype=np.float32)
    except Exception as e:
        print(f"[density_map] Error loading {mat_path}: {e}")
        return np.empty((0, 2), dtype=np.float32)


def generate_density_map_fixed(img_shape: tuple, points: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """
    Generates a continuous density map using a fixed Gaussian kernel spread.
    Ideal for Part B (sparse / uniform crowd density).

    Args:
        img_shape: (H, W) or (H, W, C)
        points: (N, 2) array of [x, y] coordinates
        sigma: Standard deviation for Gaussian kernel (default: 15.0)
    Returns:
        density_map: (H, W) float32 array where sum(density_map) == len(points)
    """
    h, w = img_shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)
    num_gt = len(points)

    if num_gt == 0:
        return density_map

    # Place delta pulses at head locations
    for pt in points:
        x = min(w - 1, max(0, int(round(pt[0]))))
        y = min(h - 1, max(0, int(round(pt[1]))))
        density_map[y, x] += 1.0

    # Apply 2D Gaussian blur
    density_map = gaussian_filter(density_map, sigma=sigma, mode="constant")

    # Re-normalize to strictly preserve total headcount integral
    d_sum = density_map.sum()
    if d_sum > 0:
        density_map = density_map * (num_gt / d_sum)

    return density_map


def generate_density_map_adaptive(
    img_shape: tuple,
    points: np.ndarray,
    k: int = 3,
    beta: float = 0.3,
    min_sigma: float = 3.0,
    max_sigma: float = 25.0,
) -> np.ndarray:
    """
    Generates a Geometry-Adaptive Gaussian Kernel density map as described in the MCNN paper.
    For each head coordinate x_i:
      sigma_i = beta * mean_distance_to_k_nearest_neighbors
    Ideal for Part A (highly dense and congested crowd scenarios).

    Args:
        img_shape: (H, W) or (H, W, C)
        points: (N, 2) array of [x, y] coordinates
        k: Number of nearest neighbors to query (default: 3)
        beta: Spread factor constant (default: 0.3)
        min_sigma: Lower bound clamp for sigma
        max_sigma: Upper bound clamp for sigma
    Returns:
        density_map: (H, W) float32 array
    """
    h, w = img_shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)
    num_gt = len(points)

    if num_gt == 0:
        return density_map

    # If only 1 or 2 points exist, fallback to fixed kernel
    if num_gt <= k:
        return generate_density_map_fixed(img_shape, points, sigma=15.0)

    # Build KD-Tree for fast nearest-neighbor spatial queries
    tree = KDTree(points.copy())
    # Query (k + 1) because the 1st neighbor is the point itself (dist=0)
    distances, _ = tree.query(points, k=k + 1)

    for i, pt in enumerate(points):
        x = min(w - 1, max(0, int(round(pt[0]))))
        y = min(h - 1, max(0, int(round(pt[1]))))

        # Average distance to k nearest neighbors
        mean_dist = distances[i][1:].mean()
        sigma = np.clip(beta * mean_dist, min_sigma, max_sigma)

        # Local Gaussian window radius (3 * sigma)
        radius = int(np.ceil(3.0 * sigma))
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)

        # Generate 2D Gaussian patch
        y_grid, x_grid = np.ogrid[y_min - y : y_max - y, x_min - x : x_max - x]
        patch = np.exp(-(x_grid**2 + y_grid**2) / (2.0 * sigma**2))
        patch_sum = patch.sum()
        if patch_sum > 0:
            patch = patch / patch_sum

        density_map[y_min:y_max, x_min:x_max] += patch

    # Re-normalize to strictly preserve total headcount integral
    d_sum = density_map.sum()
    if d_sum > 0:
        density_map = density_map * (num_gt / d_sum)

    return density_map


def generate_density_map(
    img_shape: tuple,
    points: np.ndarray,
    method: str = "adaptive",
    sigma: float = 15.0,
) -> np.ndarray:
    """
    Unified entrypoint to generate continuous density maps.
    Args:
        img_shape: (H, W)
        points: (N, 2) array of coordinates
        method: 'adaptive' (Part A recommended) or 'fixed' (Part B recommended)
        sigma: Used when method == 'fixed'
    """
    if method == "adaptive" and len(points) > 3:
        return generate_density_map_adaptive(img_shape, points)
    else:
        return generate_density_map_fixed(img_shape, points, sigma=sigma)


def density_to_heatmap(density_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Converts a continuous 2D density map into a vibrant RGB/BGR heatmap visualization.
    High density regions are mapped to intense red/yellow, while background is deep blue.
    """
    max_val = density_map.max()
    if max_val > 0:
        norm = (np.clip(density_map / max_val, 0, 1) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(density_map, dtype=np.uint8)

    heatmap_bgr = cv2.applyColorMap(norm, colormap)
    return heatmap_bgr
