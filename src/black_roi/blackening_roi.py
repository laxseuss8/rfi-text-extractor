import numpy as np

def black_roi(image: np.ndarray) -> np.ndarray:
    """
    Applies a black rectangular region to the given image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Image with the ROI blacked out.
    """
    x, y, w, h = 1, 136, 86, 21
    modified = image.copy()

    if len(modified.shape) == 3:  # Color
        modified[y:y+h, x:x+w, :] = 0
    else:  # Grayscale
        modified[y:y+h, x:x+w] = 0

    return modified
