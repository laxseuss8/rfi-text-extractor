import re
from typing import Tuple, List


def clean_text(text_x: str, text_y: str) -> Tuple[List[str], List[str]]:
    """
    Cleans OCR text from ROI_X and ROI_Y.
    
    - ROI_X: extracts numeric values, converts to float, multiplies by 1000, strips trailing zeros.
    - ROI_Y: skips first two lines, replaces lines starting with '=', and extracts non-integer floats.

    Args:
        text_x (str): OCR result from ROI_X.
        text_y (str): OCR result from ROI_Y.

    Returns:
        Tuple[List[str], List[str]]: Cleaned values from ROI_X and ROI_Y.
    """
    # --- Clean ROI_Y ---
    lines_y = text_y.strip().splitlines()
    data_lines_y = lines_y[2:] if len(lines_y) > 2 else lines_y
    cleaned_text_y = '\n'.join(data_lines_y)

    # Replace leading '=' with '-'
    cleaned_text_y = re.sub(r'^\s*=\s*', '-', cleaned_text_y, flags=re.MULTILINE)

    # Extract floats (including negative)
    matches_y = re.findall(r'-?\d+\.\d+', cleaned_text_y)

    processed_y = [
        match for match in matches_y if int(match.split('.')[1]) != 0
    ]

    # --- Clean ROI_X ---
    matches_x = re.findall(r'^(\d+(?:\.\d+)?)', text_x, flags=re.MULTILINE)

    processed_x = []
    for val in matches_x:
        try:
            value = float(val) * 1000
            formatted = f"{value:.3f}".rstrip('0').rstrip('.')
            processed_x.append(formatted)
        except ValueError:
            continue

    # Debug logs (optional — remove if integrating into production pipeline)
    print("Ref X:", ', '.join(processed_x))
    print("Ref Y:", ', '.join(processed_y))

    return processed_x, processed_y
