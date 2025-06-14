import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


def process_images(input_folder: Path, process_fn: Callable, output_folder_name="output", ocr_results=None) -> list[str]:
    """
    Process images in a folder with the given function and save results.

    Args:
        input_folder (Path): Directory to process.
        process_fn (Callable): Image processing function.
        output_folder_name (str): Output folder name.
        ocr_results (dict, optional): Dictionary mapping image stems to OCR results.

    Returns:
        list[str]: List of original image stem names.
    """
    output_folder = input_folder / output_folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    original_stems = []

    for root, dirs, files in os.walk(input_folder):
        root_path = Path(root)
        if output_folder in root_path.parents or root_path == output_folder:
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                input_path = root_path / file
                relative_path = input_path.relative_to(input_folder)

                stem, suffix = relative_path.stem, relative_path.suffix
                
                # Get OCR results for this image if available
                ocr_text = ""
                if ocr_results and stem in ocr_results:
                    x_vals = ocr_results[stem].get('X', [])
                    if len(x_vals) >= 2:  # Check if we have at least 2 X values
                        ocr_text = f"{x_vals[0]}_{x_vals[1]}_"
                
                # Create new filename with OCR text
                processed_name = f"{ocr_text}{stem}{suffix}"
                processed_path = output_folder / relative_path.parent / processed_name
                processed_path.parent.mkdir(parents=True, exist_ok=True)

                with Image.open(input_path) as img:
                    array = np.array(img)
                    processed_array = process_fn(array)
                    Image.fromarray(processed_array).save(processed_path)

                print(f"🖼️ Saved: {processed_path}")
                original_stems.append(stem)

    return original_stems