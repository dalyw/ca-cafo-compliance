#!/usr/bin/env python3

import os
import glob
import csv
import json
import tempfile
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import pandas as pd
from helper_functions.read_report_helpers import clean_common_errors, YEARS, REGIONS
from paddleocr import PaddleOCR
from multiprocessing import Pool

# All scripts generated with help from Claude 3.5


# OCR engine selection: "paddleocr" (default) or "tesseract"
ocr_engine = "paddleocr"

test_mode = True  # Set to True to process only 5 files for testing
max_pages_per_document = 10  # Limit processing to first N pages of each document

# Performance optimization settings
use_angle_cls = False  # Disable angle classification for speed
use_doc_orientation_classify = False  # Disable document orientation classification
use_doc_unwarping = False  # Disable document unwarping
text_det_limit_side_len = 960  # Limit image size for faster processing
text_rec_score_thresh = 0.5  # Lower threshold for faster processing

# Global OCR model to avoid reinitialization
global_ocr_model = None

# --- Helper Functions ---
def clean_text(text):
    """Clean up OCR text output while preserving original line structure."""
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Clean common OCR errors
        line = clean_common_errors(line)

        # Preserve leading spaces and clean up the rest
        leading_spaces = len(line) - len(line.lstrip())
        line = " " * leading_spaces + " ".join(line.strip().split())
        line = "".join(char for char in line if char.isprintable())
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def preprocess_image(image):
    """Preprocess image to improve OCR accuracy."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated


def detect_orientation(image):
    """Detect page orientation using Tesseract OSD (psm 1). 
    Returns angle in degrees (0, 90, 180, 270)."""
    try:
        osd = pytesseract.image_to_osd(image)
        for line in osd.splitlines():
            if "Rotate:" in line:
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 0


def process_page_tesseract(image, rotation=0):
    """Process a page image with Tesseract OCR."""
    processed_image = preprocess_image(image)
    angle = detect_orientation(processed_image)
    if angle != 0:
        if angle == 90:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_180)
        elif angle == 270:
            processed_image = cv2.rotate(
                processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE
            )
    # OCR Config notes:
    # https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
    ocr_config = (
        r'--oem 1 --psm 4 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        r'abcdefghijklmnopqrstuvwxyz.,()-_&/ " -c preserve_interword_spaces=1'
    )
    raw_text = pytesseract.image_to_string(
        processed_image, config=ocr_config, lang="eng"
    )
    return clean_text(raw_text)


def process_page_paddleocr(image, ocr_model):
    """Process a page image with PaddleOCR."""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize image if too large for faster processing
    height, width = img_array.shape[:2]
    max_size = 2048  # Maximum dimension for processing
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Run PaddleOCR
    result = ocr_model.ocr(img_array)
    print('page done')
    return result


def extract_text_from_paddleocr_result(result):
    """Extract plain text from PaddleOCR result."""
    if not result or not result[0]:
        return ""
    
    text_lines = []
    for line in result[0]:
        if line and len(line) >= 2:
            # line[1][0] contains the recognized text
            text_lines.append(line[1][0])
    
    return "\n".join(text_lines)


def process_pdf(pdf_path, file_list_df, ocr_engine="paddleocr", max_pages=10):
    """Process a single PDF file and extract text using OCR."""
    global global_ocr_model
    
    print(f"Processing {pdf_path}")

    pdf_dir = os.path.dirname(pdf_path)
    parent_dir = os.path.dirname(pdf_dir)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Create paddleocr_output directory
    paddleocr_dir = os.path.join(parent_dir, "paddleocr_output")
    os.makedirs(paddleocr_dir, exist_ok=True)
    
    # Create folder named after the PDF file under paddleocr_output
    pdf_folder = os.path.join(paddleocr_dir, pdf_name)
    os.makedirs(pdf_folder, exist_ok=True)
    
    # Output file paths
    text_file = os.path.join(paddleocr_dir, f"{pdf_name}.txt")
    json_file = os.path.join(paddleocr_dir, f"{pdf_name}.json")
    
    # Check if already processed (for paddleocr only)
    if os.path.exists(text_file) and os.path.exists(json_file):
        print(f"Already processed: {pdf_name}")
        return
    
    # Convert PDF to images (reduced DPI for faster processing)
    images = convert_from_path(pdf_path, dpi=300)
    
    # Limit to max_pages
    images = images[:max_pages]
    print(f"Processing first {len(images)} pages (max: {max_pages})")
    
    # Initialize PaddleOCR
    ocr_model = None
    if ocr_engine == "paddleocr":
        if global_ocr_model is None:
            print("Initializing PaddleOCR model...")
            global_ocr_model = PaddleOCR(
                use_angle_cls=use_angle_cls,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                text_det_limit_side_len=text_det_limit_side_len,
                text_rec_score_thresh=text_rec_score_thresh,
                lang="en"
            )
            print("PaddleOCR model initialized")
        ocr_model = global_ocr_model
    
    # Process each page
    all_results = []
    combined_text = []
    
    for i, image in enumerate(images):
        page_num = i + 1
        print(f"  Processing page {page_num}")
        
        page_result = {
            "page": page_num,
            "tesseract_text": "",
            "paddleocr_result": None,
            "paddleocr_text": ""
        }
        
        # Process with Tesseract if requested
        if ocr_engine == "tesseract":
            tesseract_text = process_page_tesseract(image)
            page_result["tesseract_text"] = tesseract_text
            if ocr_engine == "tesseract":
                combined_text.append(tesseract_text)
        
        # Process with PaddleOCR if requested
        if ocr_engine == "paddleocr" and ocr_model:
            paddleocr_result = process_page_paddleocr(image, ocr_model)
            page_result["paddleocr_result"] = paddleocr_result
            paddleocr_text = extract_text_from_paddleocr_result(paddleocr_result)
            page_result["paddleocr_text"] = paddleocr_text
            if ocr_engine == "paddleocr":
                combined_text.append(paddleocr_text)
        
        all_results.append(page_result)
        
        # Save individual page results
        page_folder = os.path.join(pdf_folder, f"page_{page_num:03d}")
        os.makedirs(page_folder, exist_ok=True)
        
        # Save page text
        page_text_file = os.path.join(page_folder, f"page_{page_num:03d}.txt")
        with open(page_text_file, "w", encoding="utf-8") as f:
            if ocr_engine == "tesseract":
                f.write(page_result["tesseract_text"])
            else:
                f.write(page_result["paddleocr_text"])
        
        # Save page JSON with full OCR result
        page_json_file = os.path.join(page_folder, f"page_{page_num:03d}.json")
        with open(page_json_file, "w", encoding="utf-8") as f:
            json.dump(page_result, f, indent=2, ensure_ascii=False)
        
        # Save intermediate image files in test mode
        if test_mode:
            print(f"    Saving intermediate images for page {page_num}")
            # Save original image
            image_file = os.path.join(page_folder, f"page_{page_num:03d}_original.png")
            image.save(image_file, "PNG")
            
            # Save preprocessed image if using Tesseract
            if ocr_engine == "tesseract":
                processed_image = preprocess_image(image)
                processed_image_file = os.path.join(page_folder, f"page_{page_num:03d}_processed.png")
                cv2.imwrite(processed_image_file, processed_image)
        
        # Add page break separator
        if i < len(images) - 1:
            combined_text.append(f"\nPDF PAGE BREAK {page_num}\n")
    
    # Save plain text (combined)
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_text))
    
    # Save full JSON output
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved results to {paddleocr_dir}")


def main():
    """Process all PDF files in the data directory."""
    # Load file list
    file_list_path = "ca_cafo_compliance/outputs/file_list.csv"
    if not os.path.exists(file_list_path):
        print(
            f"File list not found at {file_list_path}. "
            "Please run generate_file_list.py first."
        )
        return

    file_list_df = pd.read_csv(file_list_path)

    pdf_files = []
    for year in YEARS:
        base_data_path = f"ca_cafo_compliance/data/{year}"
        for region in REGIONS:
            region_data_path = os.path.join(base_data_path, region)
            if not os.path.exists(region_data_path):
                continue

            for county in [
                d
                for d in os.listdir(region_data_path)
                if os.path.isdir(os.path.join(region_data_path, d))
            ]:
                county_data_path = os.path.join(region_data_path, county)

                for template in [
                    d
                    for d in os.listdir(county_data_path)
                    if os.path.isdir(os.path.join(county_data_path, d))
                ]:
                    folder = os.path.join(county_data_path, template, "original")
                    if os.path.exists(folder):
                        pdf_files.extend(glob.glob(os.path.join(folder, "*.pdf")))

    if not pdf_files:
        print("No PDF files found")
        return

    # Count already processed files (check both combined files and individual page folders)
    processed_files = sum(
        1
        for pdf_path in pdf_files
        if (os.path.exists(
            os.path.join(
                os.path.dirname(os.path.dirname(pdf_path)),
                "paddleocr_output",
                f"{os.path.splitext(os.path.basename(pdf_path))[0]}.txt",
            )
        ) or os.path.exists(
            os.path.join(
                os.path.dirname(os.path.dirname(pdf_path)),
                "paddleocr_output",
                os.path.splitext(os.path.basename(pdf_path))[0],
            )
        ))
    )

    files_to_process = [
        pdf_path
        for pdf_path in pdf_files
        if not (os.path.exists(
            os.path.join(
                os.path.dirname(os.path.dirname(pdf_path)),
                "paddleocr_output",
                f"{os.path.splitext(os.path.basename(pdf_path))[0]}.txt",
            )
        ) or os.path.exists(
            os.path.join(
                os.path.dirname(os.path.dirname(pdf_path)),
                "paddleocr_output",
                os.path.splitext(os.path.basename(pdf_path))[0],
            )
        ))
    ]

    print(f"\nFound {len(pdf_files)} PDF files total")
    print(f"Already processed: {processed_files}")
    print(f"Files to process: {len(files_to_process)}")
    print(f"Using OCR engine: {ocr_engine}")
    print(f"Processing first {max_pages_per_document} pages per document")
    print(f"Performance optimizations: DPI=300, max_size=2048px, disabled angle/doc classification")

    if test_mode:
        print("Test mode active: only processing 5 files.")
        files_to_process = files_to_process[:5]

    if not files_to_process:
        print("No new files to process")
        return

    # Process PDFs in parallel

    args_list = [(pdf_path, file_list_df, ocr_engine, max_pages_per_document) for pdf_path in files_to_process]
    
    with Pool(1) as pool:
        pool.starmap(process_pdf, args_list)
    print("\nOCR processing complete")


def update_reports_available():
    """Count PDFs in R5 subfolders by region/county and update reports_available.csv."""
    # Define region-county mapping
    region_county_map = {
        "5F": ["kern"],
        "5S": ["fresno_madera", "kings", "tulare_west"],
        "5R": [],  # Add counties for 5R if needed
    }
    base_path = "ca_cafo_compliance/data/2023/R5"
    pdf_counts = {}
    for region, counties in region_county_map.items():
        total = 0
        for county in counties:
            county_path = os.path.join(base_path, county)
            for root, dirs, files in os.walk(county_path):
                total += sum(1 for f in files if f.lower().endswith(".pdf"))
        pdf_counts[region] = total
    # Read and update reports_available.csv
    csv_path = "ca_cafo_compliance/data/reports_available.csv"
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = row["region"]
            if region in pdf_counts:
                row["acquired"] = str(pdf_counts[region])
            rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["region", "acquired", "total"])
        writer.writeheader()
        writer.writerows(rows)
    print("Updated reports_available.csv with PDF counts:", pdf_counts)


if __name__ == "__main__":
    update_reports_available()
    main()
