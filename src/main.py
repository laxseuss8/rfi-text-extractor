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

# === Utility for handling uploaded files ===
def save_uploaded_files(uploaded_files, temp_dir="uploaded_images"):
    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        file_path = temp_path / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return temp_path, saved_paths

# === Modified pipeline to work with uploaded images ===
def run_pipeline_from_uploads(image_folder: Path, image_files: list[Path]):
    input_name = image_folder.name
    output_folder = image_folder / f"{input_name}_output"
    csv_path = output_folder / f"{input_name}.csv"

    # ROI blackening and saving original stems
    original_stems = process_images(image_folder, black_roi, output_folder_name=output_folder.name)
    save_dictionary_to_csv(original_stems, input_name, csv_path)

    ocr_results = {}

    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            continue  # skip unreadable files

        roi_x = process_roi_x(image)
        roi_y = process_roi_y(image)

        text_x, text_y = extract_from_image(roi_x, roi_y)
        cleaned_text_x, cleaned_text_y = clean_text(text_x, text_y)

        image_stem = image_path.stem
        ocr_results[image_stem] = {
            "X": cleaned_text_x,
            "Y": cleaned_text_y,
        }

    output_folder.mkdir(exist_ok=True)
    output_csv_path = output_folder / f"{input_name}_ocr.csv"
    save_side_by_side_csv(ocr_results, output_csv_path)

    return output_folder, output_csv_path

# === Streamlit UI ===
st.set_page_config(page_title="OCR Text Extractor", page_icon="🧠", layout="centered")
st.title("🧠 OCR Text Extraction Pipeline")
st.write("Upload image files to run the full ROI + OCR pipeline.")

uploaded_files = st.file_uploader("📁 Upload image files", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} image(s).")
    if st.button("🚀 Run OCR Pipeline"):
        with st.spinner("Processing..."):
            try:
                image_folder, image_paths = save_uploaded_files(uploaded_files)
                output_folder, result_csv = run_pipeline_from_uploads(image_folder, image_paths)
                st.success("✅ Processing complete!")

                with open(result_csv, "rb") as f:
                    st.download_button("📥 Download OCR CSV", f, file_name=result_csv.name)
            except Exception as e:
                st.error(f"❌ Error: {e}")
