import cv2
import numpy as np
from typing import Tuple


# --- ROI Definitions ---
ROI_X_COORDS = (223, 402, 107, 172)  # (x, y, w, h)
ROI_Y_COORDS = (337, 402, 117, 175)


# --- Image Loading ---
def load_image(image_path: str) -> np.ndarray:
    """Load an image from a file path."""
    return cv2.imread(image_path)


# --- ROI Extraction ---
def extract_roi(image: np.ndarray, coords: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract a single ROI given coordinates."""
    x, y, w, h = coords
    return image[y:y+h, x:x+w]


# --- Image Preprocessing ---
def process_roi(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresholded_roi = cv2.threshold(gray, 30, 225, cv2.THRESH_BINARY_INV)
    return thresholded_roi

# --- ROI Processing Wrappers (Independent) ---
def process_roi_x(image: np.ndarray) -> np.ndarray:
    """Extract and process the X-ROI."""
    roi_x = extract_roi(image, ROI_X_COORDS)
    return process_roi(roi_x)


def process_roi_y(image: np.ndarray) -> np.ndarray:
    """Extract and process the Y-ROI."""
    roi_y = extract_roi(image, ROI_Y_COORDS)
    return process_roi(roi_y)