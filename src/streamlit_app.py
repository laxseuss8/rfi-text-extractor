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
    4. Save results to CSV outputs,
    5. Flatten any nested directories under output_folder.

    Args:
        image_folder (Path): Working directory for image processing.
        image_files (list[Path]): List of image Paths to process.
        base_name (str): Base filename for output CSVs and folders.

    Returns:
        Path: Output folder containing processed images and files.
        Path: CSV file path with cleaned OCR results.
    """
    # 1) Prepare output folder path
    output_folder = image_folder / f"{base_name}_output"

    # 2) Apply black ROI filter (creates output_folder/<orig structure>)
    original_stems = process_images(
        image_folder,
        black_roi,
        output_folder_name=output_folder.name
    )

    # 3) OCR processing
    ocr_results = {}
    for image_path in image_files:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        roi_x = process_roi_x(img)
        roi_y = process_roi_y(img)
        text_x, text_y = extract_from_image(roi_x, roi_y)
        cleaned_x, cleaned_y = clean_text(text_x, text_y)
        ocr_results[image_path.stem] = {"X": cleaned_x, "Y": cleaned_y}

    # 4) Ensure output folder exists
    output_folder.mkdir(exist_ok=True)

    # 5) Flatten any nested directories
    for sub in list(output_folder.iterdir()):
        if sub.is_dir():
            for file in sub.rglob("*"):
                if file.is_file():
                    shutil.move(str(file), str(output_folder / file.name))
            sub.rmdir()

    # 6) Save the consolidated CSV
    output_csv = output_folder / f"{base_name}.csv"
    save_side_by_side_csv(ocr_results, output_csv)

    return output_folder, output_csv


def show_image_gallery(folder: Path):
    """
    Display processed images in a gallery layout within Streamlit.

    Args:
        folder (Path): Directory containing images to preview.
    """
    image_files = sorted(folder.glob("*"))
    images = [Image.open(img) for img in image_files if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
    if images:
        st.markdown("### 🖼️ Preview of Processed Images:")
        cols = st.columns(3)
        for i, img in enumerate(images):
            with cols[i % 3]:
                st.image(img, use_column_width=True, caption=image_files[i].name)

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

# Main processing trigger
if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s).")
    if st.button("🚀 Run OCR on Uploaded Files / Archives"):
        with st.spinner("Processing..."):
            try:
                # Create temporary working directory
                temp_dir = tempfile.mkdtemp()
                image_folder, file_paths = save_uploaded_files(uploaded_files, temp_dir=temp_dir)
                # Extract archives if present
                extracted_dirs = extract_archives_if_needed(file_paths, Path(temp_dir))

                # Determine base filename for outputs
                archive_file = next((f for f in file_paths if f.suffix.lower() in [".zip", ".rar", ".7z"]), None)
                base_name = archive_file.stem if archive_file else image_folder.name

                # Collect all image files from uploads and extractions
                all_images = [f for f in file_paths if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
                for folder in extracted_dirs:
                    all_images += collect_image_files_recursive(folder)

                if not all_images:
                    st.warning("No valid image files found.")
                else:
                    # Run processing pipeline
                    output_folder, result_csv = run_pipeline(Path(temp_dir), all_images, base_name)

                    # Create ZIP of processed images
                    zip_path = Path(temp_dir) / f"{base_name}_output.zip"
                    zip_folder(output_folder, zip_path)

                    st.success("✅ Processing complete!")

                    # Show gallery and provide downloads
                    show_image_gallery(output_folder)

                    with open(zip_path, "rb") as zip_file:
                        st.download_button("🖼️ Download Processed Images", zip_file, file_name=zip_path.name)

            except Exception as e:
                st.error(f"❌ Error: {e}")
