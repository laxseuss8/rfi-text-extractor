import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


def process_images(input_folder: Path, process_fn: Callable, output_folder_name="output") -> list[str]:
    """
    Process images in a folder with the given function and save results.

    Args:
        input_folder (Path): Directory to process.
        process_fn (Callable): Image processing function.
        output_folder_name (str): Output folder name.

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
                processed_name = f"{stem}{suffix}"
                processed_path = output_folder / relative_path.parent / processed_name
                processed_path.parent.mkdir(parents=True, exist_ok=True)

                with Image.open(input_path) as img:
                    array = np.array(img)
                    processed_array = process_fn(array)
                    Image.fromarray(processed_array).save(processed_path)

                print(f"🖼️ Saved: {processed_path}")
                original_stems.append(stem)

    return original_stems


def save_dictionary_to_csv(original_images: list[str], folder_label: str, csv_path: Path) -> None:
    """
    Saves matched vertical and horizontal images in a structured CSV format.

    Args:
        original_images (list[str]): List of image stems.
        folder_label (str): Used for column headers.
        csv_path (Path): Output CSV path.
    """
    v_images, h_images = [], []

    for name in sorted(original_images):
        if " V" in name:
            v_images.append(name)
        elif " H" in name:
            h_images.append(name)

    max_len = max(len(v_images), len(h_images))
    v_images.extend([""] * (max_len - len(v_images)))
    h_images.extend([""] * (max_len - len(h_images)))

    rows = []
    for v, h in zip(v_images, h_images):
        rows.append([v] + [""] * 6 + [h])
        rows.extend([[""] * 8] * 2)  # two blank rows

    columns = pd.Index([f"{folder_label}_V"] + [""] * 6 + [f"{folder_label}_H"])
    df = pd.DataFrame(rows, columns=columns)

    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
