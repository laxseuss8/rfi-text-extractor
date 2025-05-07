import streamlit as st
from black_roi.blackening_roi import black_roi
from black_roi.folder_importer import process_images, save_dictionary_to_csv

from ocr_process.image_processor import process_roi_x, process_roi_y
from ocr_process.text_extractor import extract_from_image
from ocr_process.text_cleaner import clean_text
from ocr_process.save_to_csv import save_side_by_side_csv

from pathlib import Path
import os
import cv2

def collect_image_files(input_folder: Path, exclude_dir_name="_output") -> list[Path]:
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_files = []

    for root, dirs, files in os.walk(input_folder):
        dirs[:] = [d for d in dirs if exclude_dir_name not in d]
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(Path(root) / file)

    return image_files

def run_pipeline(input_folder: Path):
    input_name = input_folder.name
    output_folder = input_folder / f"{input_name}_output"
    csv_path = output_folder / f"{input_name}.csv"

    original_stems = process_images(input_folder, black_roi, output_folder_name=output_folder.name)
    save_dictionary_to_csv(original_stems, input_name, csv_path)

    image_files = collect_image_files(input_folder)
    ocr_results = {}

    for image_path in image_files:
        image = cv2.imread(str(image_path))
        roi_x = process_roi_x(image)
        roi_y = process_roi_y(image)

        text_x, text_y = extract_from_image(roi_x, roi_y)
        cleaned_text_x, cleaned_text_y = clean_text(text_x, text_y)

        image_stem = image_path.stem
        ocr_results[image_stem] = {
            "X": cleaned_text_x,
            "Y": cleaned_text_y,
        }

    output_csv_path = output_folder / f"{input_name}_ocr.csv"
    save_side_by_side_csv(ocr_results, output_csv_path)

    return output_folder, output_csv_path

# === Streamlit UI ===
st.set_page_config(page_title="OCR Text Extractor", page_icon="🧠", layout="centered")
st.title("🧠 OCR Text Extraction Pipeline")
st.write("Select a local folder containing images to run the full ROI + OCR pipeline.")

folder_path = st.text_input("📁 Enter full directory path:", "")

if folder_path:
    path_obj = Path(folder_path)
    if not path_obj.exists() or not path_obj.is_dir():
        st.error("Invalid path or not a directory.")
    else:
        st.success(f"Selected folder: {path_obj.name}")
        if st.button("🚀 Run OCR Pipeline"):
            with st.spinner("Processing..."):
                output_folder, result_csv = run_pipeline(path_obj)
            st.success("✅ Processing complete!")

            with open(result_csv, "rb") as f:
                st.download_button("📥 Download OCR CSV", f, file_name=result_csv.name)
