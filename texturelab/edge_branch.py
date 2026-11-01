"""Student-facing exact-direction NMS, adaptive thresholds, and hysteresis."""

from __future__ import annotations
from collections import deque
import numpy as np
from .config import EdgeConfig
from .supplied import gaussian_smooth, sobel_gradients


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a 2-D image at floating-point image coordinates."""
    # YOUR CODE HERE
    #
    # TODO 1 — Resolve safe coordinates
    #   - Read image height/width and clamp y/x to the valid closed intervals.
    #   - Compute floor coordinates (y0, x0) and the next coordinates (y1, x1).
    #   - Clamp y1/x1 as well so samples at the last row/column remain valid.
    
    height, width = image.shape
    y = np.clip(y, 0, height - 1)
    x = np.clip(x, 0, width - 1)

    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)

    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)

    # TODO 2 — Compute interpolation weights
    #   - wy and wx are the fractional offsets from (y0, x0).
    #   - Form the four complementary weights for top-left, top-right,
    #     bottom-left, and bottom-right pixels.
    wy = y - y0
    wx = x - x0
    top_left = (1 - wy) * (1 - wx)
    top_right = (1 - wy) * wx
    bottom_left = wy * (1 - wx)
    bottom_right = wy * wx
    pix_top_left = image[y0, x0]
    pix_top_right = image[y0, x1]
    pix_bottom_left = image[y1, x0]
    pix_bottom_right = image[y1, x1]

    # TODO 3 — Return the weighted sample
    #   - Support array-valued y/x so a whole image grid can be sampled at once.
    #   - An exact integer coordinate should reproduce the corresponding pixel.
    sample = (
        top_left * pix_top_left + top_right * pix_top_right + bottom_left * pix_bottom_left + bottom_right * pix_bottom_right
    )

    return sample

def nms_interpolated(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin gradient magnitude along the exact image-coordinate direction."""
    # YOUR CODE HERE
    #
    # TODO 1 — Validate inputs
    #   - Require matching two-dimensional magnitude and direction arrays.
    if((magnitude.ndim != 2) or \
       (direction.ndim != 2) or \
        (magnitude.shape != direction.shape)):
        raise ValueError("magnitude and direction must have matching shapes")
    
    # TODO 2 — Construct exact-direction comparison coordinates
    #   - Build row/column index grids for the complete image.
    #   - In image coordinates, the unit step is dy=sin(theta), dx=cos(theta).
    #   - Bilinearly sample magnitude one step forward and one step backward.
    height, width = magnitude.shape
    y, x = np.indices((height, width), dtype=np.float32)
    
    dy = np.sin(direction)
    dx = np.cos(direction)

    forward = bilinear_sample(magnitude, y + dy, x + dx)
    backward = bilinear_sample(magnitude, y - dy, x - dx)

    # TODO 3 — Suppress non-maxima
    #   - Keep the original magnitude when it is >= both interpolated neighbours.
    #   - Write zero elsewhere, return float32, and explicitly zero all four borders.
    mask = (magnitude >= forward) & (magnitude >= backward)
    suppress = np.where(mask, magnitude, 0).astype(np.float32)

    suppress[0, :] = 0
    suppress[-1, :] = 0
    suppress[:, 0] = 0
    suppress[:, -1] = 0

    return suppress


def adaptive_thresholds(nms: np.ndarray, config: EdgeConfig) -> tuple[float, float]:
    """Select reproducible low/high thresholds without test labels."""
    # YOUR CODE HERE
    #
    # TODO 1 — Isolate useful responses
    #   - Estimate thresholds from strictly positive NMS values only.
    #   - If none exist, return a pair that produces no strong/weak edges.
    pos_nms = nms[nms > 0]
    if len(pos_nms) == 0:
        return (float("inf"), float("inf"))

    # TODO 2 — Compute the high threshold
    #   - "percentile": use config.high_percentile on positive values.
    #   - "robust": use median + 2.5 * 1.4826 * median absolute deviation.
    #       - 1.4826 is a mathematically motivated conversion from MAD to a standard-deviation-like scale.
    #       - 2.5 is a scale multiplier for robust outlier rejection.
    #   - Reject unknown threshold_method values with ValueError.
    if (config.threshold_method == "percentile"):
        high = np.percentile(pos_nms, config.high_percentile)
    elif (config.threshold_method == "robust"):
        median = np.median(pos_nms)
        mad = np.median(np.abs(pos_nms - median))
        high = median + 2.5 * 1.4826 * mad
    else:
        raise ValueError("Unknown threshold method")

    # TODO 3 — Compute and return the low threshold
    #   - low is config.low_ratio multiplied by high.
    #   - Return ordinary Python floats in (low, high) order.

    low = config.low_ratio * high
    return (float(low), float(high))


def hysteresis(strong: np.ndarray, weak: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep all weak pixels reachable from strong seeds."""
    # YOUR CODE HERE
    #
    # TODO 1 — Validate and initialise
    #   - Require matching mask shapes and connectivity equal to 4 or 8.
    #   - Convert inputs to Boolean masks without modifying the caller's arrays.
    #   - Seed the result and a deque with every strong-pixel coordinate.
    
    if (strong.shape != weak.shape):
        raise ValueError("[Hysteresis] Mask shapes are not matching")

    if connectivity not in (4, 8):
        raise ValueError("[Hysteresis] Connectivity not 4 or 8")

    strong_mask = strong.astype(bool, copy=True)
    weak_mask = weak.astype(bool, copy=True)

    result = strong_mask.copy()
    queue = deque(map(tuple, np.argwhere(strong_mask)))

    # TODO 2 — Define the neighbourhood
    #   - Four-connectivity uses vertical/horizontal offsets.
    #   - Eight-connectivity additionally includes all diagonal offsets.
    if (connectivity == 4):
        neighbourhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbourhood = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # TODO 3 — Traverse the full connected component
    #   - Pop a coordinate, check in-bounds allowed neighbours, and accept every
    #     unvisited weak pixel connected to a seed.
    #   - Enqueue newly accepted pixels so multi-pixel weak chains are retained.
    #   - Return the final Boolean mask after the queue is exhausted.

    height, width = strong.shape

    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbourhood:
            y_neighbour = y + dy
            x_neighbour = x + dx

            if (0 <= y_neighbour < height and \
                0 <= x_neighbour < width and \
                weak_mask[y_neighbour, x_neighbour] and \
                    not result[y_neighbour, x_neighbour]):
                result[y_neighbour, x_neighbour] = True
                queue.append((y_neighbour, x_neighbour))

    return result


def detect_edges(gray: np.ndarray, config: EdgeConfig) -> dict[str, np.ndarray | float]:
    """Run the supplied smoothing/Sobel stages and student edge stages."""
    smoothed = gaussian_smooth(gray, config.gaussian_sigma)
    gx, gy, magnitude, direction = sobel_gradients(smoothed)
    # YOUR CODE HERE
    #
    # TODO 1 — Thin and threshold
    #   - Run nms_interpolated on magnitude/direction.
    #   - Obtain (low, high) from adaptive_thresholds.
    #   - Strong pixels meet/exceed high; weak pixels meet/exceed low but are not strong.

    nms = nms_interpolated(magnitude, direction)
    low, high = adaptive_thresholds(nms, config)
    strong = nms >= high
    weak = (nms >= low) & (nms < high)
    
    # TODO 2 — Link and package the result
    #   - Run hysteresis with config.connectivity.
    #   - Return gx, gy, magnitude, direction, nms, low, high, and edges using
    #     exactly those dictionary keys so downstream feature code remains stable.
    edges = hysteresis(strong, weak, config.connectivity)

    return {
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "direction": direction,
        "nms": nms,
        "low": low,
        "high": high,
        "edges": edges
    }
