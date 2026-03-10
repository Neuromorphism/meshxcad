"""Compare two engineering drawings as images.

Metrics are based on edge-pixel sets rather than raw pixel diffs, which
gives robustness to line-weight variation and anti-aliasing.
"""

import numpy as np
from scipy.spatial import KDTree


def extract_edge_pixels(image, threshold=128) -> np.ndarray:
    """Binarise image and extract dark pixel coordinates.

    Args:
        image: (H, W) or (H, W, 3) uint8 image. Dark = edges.
        threshold: pixels darker than this are edges.

    Returns:
        (N, 2) array of (row, col) coordinates.
    """
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.astype(np.float64)
    coords = np.argwhere(gray < threshold)
    return coords


def chamfer_distance_2d(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Mean bidirectional nearest-neighbour distance between point sets.

    Returns 0.0 if both sets are empty. Returns inf if one is empty and
    the other is not.
    """
    if len(points_a) == 0 and len(points_b) == 0:
        return 0.0
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")
    tree_a = KDTree(points_a)
    tree_b = KDTree(points_b)
    dist_a_to_b, _ = tree_b.query(points_a)
    dist_b_to_a, _ = tree_a.query(points_b)
    return float(0.5 * np.mean(dist_a_to_b) + 0.5 * np.mean(dist_b_to_a))


def compare_drawings(drawing_a: np.ndarray, drawing_b: np.ndarray,
                     threshold=128) -> dict:
    """Compare two engineering drawings.

    Args:
        drawing_a: (H, W) or (H, W, 3) uint8 image.
        drawing_b: same.
        threshold: edge pixel threshold.

    Returns:
        dict with pixel_iou, chamfer_distance, edge_precision, edge_recall.
    """
    edges_a = extract_edge_pixels(drawing_a, threshold)
    edges_b = extract_edge_pixels(drawing_b, threshold)

    # Chamfer distance
    chamfer = chamfer_distance_2d(edges_a, edges_b)

    # Pixel IoU — build binary masks at the union of both image sizes
    h = max(drawing_a.shape[0], drawing_b.shape[0])
    w = max(drawing_a.shape[1], drawing_b.shape[1])

    mask_a = np.zeros((h, w), dtype=bool)
    mask_b = np.zeros((h, w), dtype=bool)
    if len(edges_a) > 0:
        mask_a[edges_a[:, 0], edges_a[:, 1]] = True
    if len(edges_b) > 0:
        mask_b[edges_b[:, 0], edges_b[:, 1]] = True

    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    iou = float(intersection / union) if union > 0 else 1.0

    # Precision / recall with tolerance (within 3 pixels)
    tolerance = 3.0
    if len(edges_a) > 0 and len(edges_b) > 0:
        tree_b = KDTree(edges_b)
        dist_a, _ = tree_b.query(edges_a)
        precision = float(np.mean(dist_a <= tolerance))

        tree_a = KDTree(edges_a)
        dist_b, _ = tree_a.query(edges_b)
        recall = float(np.mean(dist_b <= tolerance))
    elif len(edges_a) == 0 and len(edges_b) == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = 0.0
        recall = 0.0

    return {
        "pixel_iou": iou,
        "chamfer_distance": chamfer,
        "edge_precision": precision,
        "edge_recall": recall,
    }
