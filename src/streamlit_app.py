'''
OCR Text Extraction Streamlit Application

This module implements a complete OCR pipeline with a Streamlit user interface. Users can upload images or archive files (.zip, .7z, .rar). 
Uploaded files are saved locally, archives are extracted, and images are processed using a black ROI filter and OCR steps. 
Outputs include processed images preview, a side-by-side CSV of OCR results, and downloadable ZIP and CSV files.
'''

import streamlit as st
from black_roi.blackening_roi import black_roi
from black_roi.folder_importer import process_images

from ocr_process.image_processor import process_roi_x, process_roi_y
from ocr_process.text_extractor import extract_from_image
from ocr_process.text_cleaner import clean_text
from ocr_process.save_to_csv import save_side_by_side_csv

from pathlib import Path
import cv2
import zipfile
import py7zr
import shutil
import tempfile
import os
from PIL import Image

# === Utilities ===
def save_uploaded_files(uploaded_files, temp_dir="uploaded_data"):
    """
    Save uploaded Streamlit files to a local directory.

    Args:
        uploaded_files (list of UploadedFile): Files uploaded via Streamlit's file_uploader.
        temp_dir (str): Name of temporary directory for saving files.

    Returns:
        Path: Path object of the directory containing saved files.
        list[Path]: List of Paths to each saved file.
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        file_path = temp_path / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return temp_path, saved_paths

def extract_archives_if_needed(file_paths, extract_dir):
    """
    Extract supported archives (.zip, .7z) into dedicated subdirectories.

    Args:
        file_paths (list[Path]): List of file Paths that may include archives.
        extract_dir (Path): Directory under which to create extraction folders.

    Returns:
        list[Path]: Paths of folders containing extracted files.
    """
    extracted_folders = []
    for file_path in file_paths:
        suffix = file_path.suffix.lower()
        extracted_folder = extract_dir / file_path.stem
        extracted_folder.mkdir(exist_ok=True)

        try:
            if suffix == ".zip":
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_folder)
                extracted_folders.append(extracted_folder)

            elif suffix == ".7z":
                with py7zr.SevenZipFile(file_path, mode='r') as archive:
                    archive.extractall(path=extracted_folder)
                extracted_folders.append(extracted_folder)

            elif suffix == ".rar":
                # .rar extraction requires unrar; skip if unavailable
                print(f"Skipping .rar archive: {file_path} (no 'unrar' CLI installed)")

        except Exception as e:
            print(f"Failed to extract {file_path.name}: {e}")
    return extracted_folders

def collect_image_files_recursive(root_folder: Path, image_extensions=('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
    """
    Recursively gather image files from a directory.

    Args:
        root_folder (Path): Base folder to search for image files.
        image_extensions (tuple): Allowed image file extensions.

    Returns:
        list[Path]: List of Paths matching allowed extensions.
    """
    return [p for p in root_folder.rglob("*") if p.suffix.lower() in image_extensions and p.is_file()]

def zip_folder(folder_path: Path, zip_name: Path):
    """
    Compress a folder into a ZIP archive.

    Args:
        folder_path (Path): Directory to compress.
        zip_name (Path): Destination ZIP file name (without .zip suffix).

    Returns:
        Path: Path to created ZIP archive.
    """
    shutil.make_archive(str(zip_name.with_suffix('')), 'zip', str(folder_path))
    return zip_name

def run_pipeline(image_folder: Path, image_files: list[Path], base_name: str):
    """
    Execute the full OCR pipeline: 
    1. ROI blackening,
    2. OCR extraction from the output of ROI blackening,
    3. Cleaning of extracted text,
    4. Save results to CSV outputs.

    Args:
        image_folder (Path): Working directory for image processing.
        image_files (list[Path]): List of image Paths to process.
        base_name (str): Base filename for output CSVs and folders.

    Returns:
        Path: Output folder containing processed images and files.
        Path: CSV file path with cleaned OCR results.
    """
    # Prepare output folder and temporary CSV path
    output_folder = image_folder / f"{base_name}_output"


    # Apply black ROI filter and save stems dictionary
    original_stems = process_images(image_folder, black_roi, output_folder_name=output_folder.name)


    # OCR processing on each image
    ocr_results = {}
    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        # Extract regions of interest
        roi_x = process_roi_x(image)
        roi_y = process_roi_y(image)
        # Perform OCR
        text_x, text_y = extract_from_image(roi_x, roi_y)
        # Clean extracted text
        cleaned_text_x, cleaned_text_y = clean_text(text_x, text_y)
        ocr_results[image_path.stem] = {"X": cleaned_text_x, "Y": cleaned_text_y}

    # Ensure output directory exists and save final OCR CSV
    output_folder.mkdir(exist_ok=True)
    output_csv_path = output_folder / f"{base_name}.csv"
    save_side_by_side_csv(ocr_results, output_csv_path)

    return output_folder, output_csv_path

# === Streamlit UI ===
# Page configuration and title
st.set_page_config(page_title="OCR Text Extractor", page_icon="🧠", layout="centered")
st.title("🧠 OCR Text Extraction Pipeline")

# File uploader for images and archives
uploaded_files = st.file_uploader(
    "📁 Upload images or archives (.png, .jpg, .rar, .7z, .zip)",
    type=["png", "jpg", "jpeg", "tif", "tiff", "rar", "7z", "zip"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s).")
    if st.button("🚀 Run OCR on All Uploaded Files"):
        with st.spinner("Processing..."):
            try:
                temp_dir = Path(tempfile.mkdtemp())
                # save uploads
                image_folder, file_paths = save_uploaded_files(uploaded_files, temp_dir=str(temp_dir))
                # extract archives
                extracted_dirs = extract_archives_if_needed(file_paths, temp_dir)
                # collect all images
                all_images = [f for f in file_paths if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
                for d in extracted_dirs:
                    all_images += collect_image_files_recursive(d)

                if not all_images:
                    st.warning("No valid image files found.")
                else:
                    # group files by base_name (handles multiple archives/images)
                    groups: dict[str, list[Path]] = {}
                    for img in all_images:
                        # use stem up to first underscore for grouping, or full stem
                        base = img.stem.rsplit('_', 1)[0]
                        groups.setdefault(base, []).append(img)

                    # process each group separately
                    for base, imgs in groups.items():
                        st.markdown(f"### Results for `{base}`")
                        out_folder, out_csv = run_pipeline(temp_dir, imgs, base)
                        # zip images
                        zip_path = temp_dir / f"{base}_output.zip"
                        zip_folder(out_folder, zip_path)
    
                        with open(out_csv, "rb") as f:
                            st.download_button(f"📥 Download `{base}` OCR CSV", f, file_name=out_csv.name)
                        with open(zip_path, "rb") as f:
                            st.download_button(f"🖼️ Download `{base}` Images ZIP", f, file_name=zip_path.name)

            except Exception as e:
                st.error(f"❌ Error: {e}")
