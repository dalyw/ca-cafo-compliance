#!/usr/bin/env python3

import os
import glob
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import re
import pandas as pd
from conversion_factors import *

def clean_text(text):
    """Clean up OCR text output while preserving original line structure."""
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        # Fix common OCR errors
        line = line.replace('|', 'I')
        line = line.replace('0O', 'O')
        line = line.replace('1I', 'I')
        line = line.replace('S5', 'S')
        
        # Fix common OCR errors in numbers
        line = re.sub(r'(\d)O(\d)', r'\1O\2', line)
        line = re.sub(r'(\d)l(\d)', r'\1l\2', line)
        line = re.sub(r'(\d)I(\d)', r'\1I\2', line)
        
        # Fix common OCR errors in text
        line = re.sub(r'([a-zA-Z])0([a-zA-Z])', r'\1O\2', line)
        line = re.sub(r'([a-zA-Z])l([a-zA-Z])', r'\1I\2', line)
        line = re.sub(r'([a-zA-Z])I([a-zA-Z])', r'\1I\2', line)
        
        # Remove extra spaces while preserving indentation
        leading_spaces = len(line) - len(line.lstrip())
        line = ' ' * leading_spaces + ' '.join(line.strip().split())
        
        # Remove non-printable characters
        line = ''.join(char for char in line if char.isprintable())
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def preprocess_image(image, fast_mode=OCR_FAST_MODE):
    """Preprocess image to improve OCR accuracy."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if fast_mode:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    else:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(thresh)
        kernel = np.ones((1,1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        return dilated

def get_ocr_config(fast_mode=OCR_FAST_MODE):
    """Get Tesseract OCR configuration based on mode."""
    base_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-_&/ " -c preserve_interword_spaces=1'
    if not fast_mode:
        base_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-_&/ " --dpi 300 -c preserve_interword_spaces=1'
    return base_config

def process_page(image, rotation=0):
    """Process a page image with specified rotation."""
    processed_image = preprocess_image(image, fast_mode=OCR_FAST_MODE)
    
    # Apply rotation if needed
    if rotation == 90:
        processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == -90:
        processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    text = pytesseract.image_to_string(processed_image, config=get_ocr_config(OCR_FAST_MODE), lang='eng')
    return clean_text(text)

def process_pdf(pdf_path, file_list_df):
    """Process a single PDF file and extract text using OCR."""
    try:
        # Set up directories
        pdf_dir = os.path.dirname(pdf_path)
        parent_dir = os.path.dirname(pdf_dir)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        ocr_dir = os.path.join(parent_dir, 'ocr_output')
        
        # Check if OCR has already been performed
        text_file = os.path.join(ocr_dir, f'{pdf_name}.txt')
        if os.path.exists(text_file):
            return
        
        # Get rotation from file list
        filename = os.path.basename(pdf_path)
        rotation = file_list_df[file_list_df['filename'] == filename]['rotation'].iloc[0] if not file_list_df.empty else 0
        
        # Perform OCR
        os.makedirs(ocr_dir, exist_ok=True)
        images = convert_from_path(pdf_path, dpi=OCR_DPI)
        
        # Process each page
        combined_text = []
        for i, image in enumerate(images):
            text = process_page(image, rotation)
            combined_text.append(text)
            if i < len(images) - 1:
                combined_text.append(f"\nPDF PAGE BREAK {i+1}\n")
        
        # Save combined text
        with open(text_file, 'w') as f:
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
    file_list_path = 'outputs/file_list.csv'
    if not os.path.exists(file_list_path):
        print(f"File list not found at {file_list_path}. Please run generate_file_list.py first.")
        return
    
    file_list_df = pd.read_csv(file_list_path)
    
    pdf_files = []
    for year in YEARS:
        base_data_path = f"data/{year}"
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
                                                      f'{os.path.splitext(os.path.basename(pdf_path))[0]}.txt')))
    
    files_to_process = [pdf_path for pdf_path in pdf_files 
                       if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(pdf_path)), 
                                                        'ocr_output', 
                                                        f'{os.path.splitext(os.path.basename(pdf_path))[0]}.txt'))]
    
    print(f"\nFound {len(pdf_files)} PDF files total")
    print(f"Already processed: {processed_files}")
    print(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        print("No new files to process")
        return
    
    # Process PDFs in parallel
    from multiprocessing import Pool
    with Pool(OCR_NUM_CORES) as pool:
        pool.starmap(process_pdf, [(pdf_path, file_list_df) for pdf_path in files_to_process])
    
    print("\nOCR processing complete")

if __name__ == "__main__":
    main() 