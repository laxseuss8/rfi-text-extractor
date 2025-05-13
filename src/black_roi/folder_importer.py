import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


def process_images(input_folder: Path, process_fn: Callable, output_folder_name="output", ocr_results=None) -> list[str]:
    """
    Process images in a folder with the given function and save results.
    Output structure keeps immediate subfolders: output_folder/folder1/images.png
    Deeper nested folders are flattened into the immediate subfolder.

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

    print("\nDEBUG - process_images received OCR results:")
    print(ocr_results)

    original_stems = []

    for root, dirs, files in os.walk(input_folder):
        root_path = Path(root)
        if output_folder in root_path.parents or root_path == output_folder:
            continue

        # Get the immediate subfolder name (first level only)
        relative_to_input = root_path.relative_to(input_folder)
        if relative_to_input == Path('.'):
            target_folder = output_folder
        else:
            # Take only the first part of the path for the subfolder
            immediate_folder = relative_to_input.parts[0]
            target_folder = output_folder / immediate_folder
            target_folder.mkdir(exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                input_path = root_path / file
                relative_path = input_path.relative_to(input_folder)

                # If the image is in a deeper folder, include that in the filename
                deeper_path = relative_path.parent.relative_to(immediate_folder) if relative_to_input != Path('.') else relative_path.parent
                folder_prefix = f"{str(deeper_path).replace(os.sep, '_')}_" if str(deeper_path) != '.' else ""

                stem, suffix = relative_path.stem, relative_path.suffix
                
                # Get OCR results for this image if available
                ocr_text = ""
                if ocr_results and stem in ocr_results:
                    x_vals = ocr_results[stem].get('X', [])
                    print(f"\nDEBUG - Processing {stem}:")
                    print(f"  Found X values: {x_vals}")
                    if len(x_vals) >= 2:  # Check if we have at least 2 X values
                        ocr_text = f"{x_vals[0]}_{x_vals[1]}_"
                        print(f"  Using OCR text: {ocr_text}")
                    else:
                        print(f"  Not enough X values (need 2, got {len(x_vals)})")
                else:
                    print(f"\nDEBUG - No OCR results found for {stem}")
                
                # Create new filename with deeper folder structure and OCR text
                processed_name = f"{ocr_text}{folder_prefix}{stem}{suffix}"
                processed_path = target_folder / processed_name
                print(f"  Final filename: {processed_name}")

                with Image.open(input_path) as img:
                    array = np.array(img)
                    processed_array = process_fn(array)
                    Image.fromarray(processed_array).save(processed_path)

                print(f"🖼️ Saved: {processed_path}")
                original_stems.append(stem)

    return original_stems