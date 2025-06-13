#!/usr/bin/env python3

import os
import glob
import re
import csv
import tempfile
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import multiprocessing as mp
from functools import partial
import sys
from helper_functions.read_report_helpers import *

# All scripts generated with help from Claude 3.5

# Add import for Google Document AI
try:
    from google.cloud import documentai_v1 as documentai
    import PyPDF2
    GOOGLE_DOC_AI_AVAILABLE = True
except ImportError:
    GOOGLE_DOC_AI_AVAILABLE = False

# Set to True to use Google Document AI, False for Tesseract
use_google_doc_ai = False

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
        line = ' ' * leading_spaces + ' '.join(line.strip().split())
        line = ''.join(char for char in line if char.isprintable())
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def preprocess_image(image):
    """Preprocess image to improve OCR accuracy."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

def detect_orientation(image):
    """Detect page orientation using Tesseract OSD (psm 1). Returns angle in degrees (0, 90, 180, 270)."""
    try:
        osd = pytesseract.image_to_osd(image)
        for line in osd.splitlines():
            if "Rotate:" in line:
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 0

def process_page(image, rotation=0):
    """Process a page image with orientation detection and specified rotation."""
    processed_image = preprocess_image(image)
    angle = detect_orientation(processed_image)
    if angle != 0:
        if angle == 90:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_180)
        elif angle == 270:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # OCR Config notes https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
    ocr_config = r'--oem 1 --psm 4 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-_&/ " -c preserve_interword_spaces=1'
    raw_text = pytesseract.image_to_string(processed_image, config=ocr_config, lang='eng')
    return clean_text(raw_text)

def get_first_n_pages(pdf_path, n=15, start_page=1):
    with open(pdf_path, "rb") as infile:
        reader = PyPDF2.PdfReader(infile)
        writer = PyPDF2.PdfWriter()
        start_idx = max(0, start_page - 1)
        for i in range(start_idx, min(start_idx + n, len(reader.pages))):
            writer.add_page(reader.pages[i])
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with open(temp.name, "wb") as outfile:
            writer.write(outfile)
        return temp.name

def get_first_key_page_for_template(pdf_path):
    parts = pdf_path.split(os.sep)
    template = None
    for i, part in enumerate(parts):
        if part == 'original' and i > 0:
            template = parts[i-1]
            break
    if not template:
        return 1
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates_pages.csv')
    try:
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['template'] == template and row['first_key_page']:
                    return int(row['first_key_page'])
    except Exception:
        pass
    return 1

def extract_text_with_google_docai(file_path, output_text_path, project_id, location, processor_id, service_account_path):
    """Extract text from a document using Google Document AI and save to file."""
    if not GOOGLE_DOC_AI_AVAILABLE:
        raise ImportError("google-cloud-documentai is not installed. Please install it with 'pip install google-cloud-documentai'.")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
    start_page = get_first_key_page_for_template(file_path)
    temp_pdf_path = get_first_n_pages(file_path, n=15, start_page=start_page)
    client = documentai.DocumentProcessorServiceClient()
    resource_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
    with open(temp_pdf_path, "rb") as f:
        file_content = f.read()
    request = documentai.ProcessRequest(
        name=resource_name,
        raw_document=documentai.RawDocument(
            content=file_content,
            mime_type="application/pdf"
        )
    )
    result = client.process_document(request=request)
    document = result.document
    with open(output_text_path, "w", encoding="utf-8") as out_file:
        out_file.write(document.text)
    print(f"Extracted text saved to {output_text_path}")
    return document.text

def process_pdf(pdf_path, file_list_df, ocr_engine="tesseract", google_docai_config=None):
    """Process a single PDF file and extract text using OCR (Tesseract or Google DocAI)."""
    print(f'Processing {pdf_path}')
    try:
        pdf_dir = os.path.dirname(pdf_path)
        parent_dir = os.path.dirname(pdf_dir)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        ocr_dir = os.path.join(parent_dir, 'ocr_output')
        ai_ocr_dir = os.path.join(parent_dir, 'ai_ocr_output')
        text_file_ocr = os.path.join(ocr_dir, f'{pdf_name}.txt')
        text_file_handwriting = os.path.join(ai_ocr_dir, f'{pdf_name}.txt')
        if os.path.exists(text_file_ocr) or os.path.exists(text_file_handwriting):
            return
        os.makedirs(ocr_dir, exist_ok=True)
        if ocr_engine == "google":
            if google_docai_config is None:
                raise ValueError("google_docai_config must be provided when using Google Document AI.")
            extract_text_with_google_docai(
                file_path=pdf_path,
                output_text_path=text_file_ocr,
                project_id=google_docai_config["project_id"],
                location=google_docai_config["location"],
                processor_id=google_docai_config["processor_id"],
                service_account_path=google_docai_config["service_account_path"]
            )
            return
        images = convert_from_path(pdf_path, dpi=400)
        combined_text = []
        for i, image in enumerate(images):
            text = process_page(image, rotation=0)
            combined_text.append(text)
            if i < len(images) - 1:
                combined_text.append(f"\nPDF PAGE BREAK {i+1}\n")
        with open(text_file_ocr, 'w') as f:
            f.write('\n'.join(combined_text))
    except ImportError:
        print("\nOCR dependencies not found. To use OCR:")
        print("1. Install Tesseract: brew install tesseract")
        print("2. Install Python packages: poetry add pytesseract pdf2image opencv-python numpy")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def main():
    """Process all PDF files in the data directory."""
    # Load file list
    file_list_path = 'ca_cafo_compliance/outputs/file_list.csv'
    if not os.path.exists(file_list_path):
        print(f"File list not found at {file_list_path}. Please run generate_file_list.py first.")
        return
    
    file_list_df = pd.read_csv(file_list_path)
    
    pdf_files = []
    for year in YEARS:
        base_data_path = f"ca_cafo_compliance/data/{year}"
        for region in REGIONS:
            region_data_path = os.path.join(base_data_path, region)
            if not os.path.exists(region_data_path):
                continue
                
            for county in [d for d in os.listdir(region_data_path) if os.path.isdir(os.path.join(region_data_path, d))]:
                county_data_path = os.path.join(region_data_path, county)
                
                for template in [d for d in os.listdir(county_data_path) if os.path.isdir(os.path.join(county_data_path, d))]:
                    folder = os.path.join(county_data_path, template, 'original')
                    if os.path.exists(folder):
                        pdf_files.extend(glob.glob(os.path.join(folder, '*.pdf')))
    
    if not pdf_files:
        print("No PDF files found")
        return
    
    # Count already processed files
    processed_files = sum(1 for pdf_path in pdf_files 
                         if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(pdf_path)), 
                                                      'ocr_output', 
                                                      f'{os.path.splitext(os.path.basename(pdf_path))[0]}.txt'))
                         or os.path.exists(os.path.join(os.path.dirname(os.path.dirname(pdf_path)),
                                                      'ai_ocr_output',
                                                      f'{os.path.splitext(os.path.basename(pdf_path))[0]}.txt')))

    files_to_process = [pdf_path for pdf_path in pdf_files 
                       if not (os.path.exists(os.path.join(os.path.dirname(os.path.dirname(pdf_path)), 
                                                        'ocr_output', 
                                                        f'{os.path.splitext(os.path.basename(pdf_path))[0]}.txt'))
                                or os.path.exists(os.path.join(os.path.dirname(os.path.dirname(pdf_path)),
                                                               'ai_ocr_output',
                                                               f'{os.path.splitext(os.path.basename(pdf_path))[0]}.txt')))]
    
    print(f"\nFound {len(pdf_files)} PDF files total")
    print(f"Already processed: {processed_files}")
    print(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        print("No new files to process")
        return
    
    # Example Google DocAI config (update with your actual values)
    google_docai_config = {
        "project_id": "831414910366",
        "location": "us",
        "processor_id": "48615fe9e055a014",
        "service_account_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "plasma-geode-462802-k1-005fc515d4c5.json")
    }
    
    def build_args(pdf_path):
        if use_google_doc_ai:
            return (pdf_path, file_list_df, "google", google_docai_config)
        else:
            return (pdf_path, file_list_df, "tesseract", None)
    args_list = [build_args(pdf_path) for pdf_path in files_to_process]
    # Process PDFs in parallel
    from multiprocessing import Pool
    with Pool(3) as pool:
        pool.starmap(process_pdf, args_list)
    print("\nOCR processing complete")

def update_reports_available():
    """Count PDFs in R5 subfolders by region/county and update reports_available.csv."""
    # Define region-county mapping
    region_county_map = {
        '5F': ['kern'],
        '5S': ['fresno_madera', 'kings', 'tulare_west'],
        '5R': []  # Add counties for 5R if needed
    }
    base_path = 'ca_cafo_compliance/data/2023/R5'
    pdf_counts = {}
    for region, counties in region_county_map.items():
        total = 0
        for county in counties:
            county_path = os.path.join(base_path, county)
            for root, dirs, files in os.walk(county_path):
                total += sum(1 for f in files if f.lower().endswith('.pdf'))
        pdf_counts[region] = total
    # Read and update reports_available.csv
    csv_path = 'ca_cafo_compliance/data/reports_available.csv'
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = row['region']
            if region in pdf_counts:
                row['acquired'] = str(pdf_counts[region])
            rows.append(row)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['region', 'acquired', 'total'])
        writer.writeheader()
        writer.writerows(rows)
    print('Updated reports_available.csv with PDF counts:', pdf_counts)

if __name__ == "__main__":
    update_reports_available()
    main() 