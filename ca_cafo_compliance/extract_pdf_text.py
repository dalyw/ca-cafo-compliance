#!/usr/bin/env python3

import os
import glob
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import multiprocessing as mp
from functools import partial
import cv2
import numpy as np
import re

def clean_text(text):
    """Clean up OCR text output while preserving original line structure."""
    # Split into lines to preserve structure
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        # Fix common OCR errors
        line = line.replace('|', 'I')
        line = line.replace('0O', 'O')  # Common confusion between 0 and O
        line = line.replace('1I', 'I')  # Common confusion between 1 and I
        line = line.replace('S5', 'S')  # Common confusion between S and 5
        
        # Fix common OCR errors in numbers - use raw strings for regex patterns
        line = re.sub(r'(\d)O(\d)', r'\1O\2', line)  # Fix 0 in numbers
        line = re.sub(r'(\d)l(\d)', r'\1l\2', line)  # Fix l in numbers
        line = re.sub(r'(\d)I(\d)', r'\1I\2', line)  # Fix I in numbers
        
        # Fix common OCR errors in text - use raw strings for regex patterns
        line = re.sub(r'([a-zA-Z])0([a-zA-Z])', r'\1O\2', line)  # Fix 0 in text
        line = re.sub(r'([a-zA-Z])l([a-zA-Z])', r'\1I\2', line)  # Fix l in text
        line = re.sub(r'([a-zA-Z])I([a-zA-Z])', r'\1I\2', line)  # Fix I in text
        
        # Remove extra spaces within line while preserving indentation
        leading_spaces = len(line) - len(line.lstrip())
        line = ' ' * leading_spaces + ' '.join(line.strip().split())
        
        # Remove any non-printable characters
        line = ''.join(char for char in line if char.isprintable())
        
        cleaned_lines.append(line)
    
    # Join lines back together
    return '\n'.join(cleaned_lines)

def preprocess_image(image, fast_mode=True):
    """Preprocess image to improve OCR accuracy for handwritten text."""
    # Convert PIL Image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if fast_mode:
        # Enhanced fast mode preprocessing
        # Use OTSU thresholding with additional contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    else:
        # Full preprocessing for better quality
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Dilation to make text more prominent
        kernel = np.ones((1,1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        
        return dilated

def detect_text_orientation(text):
    """Detect if text appears to be in a sensible orientation.
    Returns True if text appears horizontal, False if it might be vertical."""
    
    # Split into lines and count non-empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True  # Default to horizontal if no text
    
    # Livingston template specific checks
    if any("Livingston Dairy Consulting" in line for line in lines):
        # Check if the text appears rotated (vertical)
        # In rotated text, lines are typically very short and contain few spaces
        short_lines = 0
        for line in lines:
            if len(line) <= 3 and ' ' not in line:
                short_lines += 1
        
        # If more than 70% of lines are very short, it's likely rotated
        if short_lines / len(lines) > 0.7:
            return False
    
    # General text orientation checks
    normal_lines = 0
    for line in lines:
        # Check if line has spaces and reasonable length
        if ' ' in line and 5 <= len(line) <= 100:
            normal_lines += 1
    
    # If more than 50% of lines look normal, consider it horizontal
    return (normal_lines / len(lines)) > 0.5

def process_page_with_rotation(image, fast_mode=True, custom_config=None):
    """Process a page image trying different rotations if needed."""
    # Try original orientation first
    processed_image = preprocess_image(image, fast_mode=fast_mode)
    text = pytesseract.image_to_string(
        processed_image,
        config=custom_config,
        lang='eng'
    )
    text = clean_text(text)
    
    # If text doesn't look right, try 90 and -90 degree rotations
    if not detect_text_orientation(text):
        # Try 90 degrees
        rotated_90 = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
        text_90 = pytesseract.image_to_string(
            rotated_90,
            config=custom_config,
            lang='eng'
        )
        text_90 = clean_text(text_90)
        
        # Try -90 degrees
        rotated_neg90 = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        text_neg90 = pytesseract.image_to_string(
            rotated_neg90,
            config=custom_config,
            lang='eng'
        )
        text_neg90 = clean_text(text_neg90)
        
        # Choose the orientation that produces the most sensible text
        if detect_text_orientation(text_90):
            return text_90
        elif detect_text_orientation(text_neg90):
            return text_neg90
        
        # If neither rotation looks right, try the one with more normal-looking lines
        lines_90 = [line for line in text_90.splitlines() if line.strip()]
        lines_neg90 = [line for line in text_neg90.splitlines() if line.strip()]
        
        normal_lines_90 = sum(1 for line in lines_90 if ' ' in line and 5 <= len(line) <= 100)
        normal_lines_neg90 = sum(1 for line in lines_neg90 if ' ' in line and 5 <= len(line) <= 100)
        
        if normal_lines_90 > normal_lines_neg90:
            return text_90
        else:
            return text_neg90
            
    return text

def process_pdf(pdf_path, dpi=200, fast_mode=True):
    """Process a single PDF file and extract text using OCR."""
    try:
        # Set up directories
        pdf_dir = os.path.dirname(pdf_path)  # This is the 'original' folder
        parent_dir = os.path.dirname(pdf_dir)  # This is the template folder
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        ocr_dir = os.path.join(parent_dir, 'ocr_output')  # Create ocr_output at template level
        
        # Check if OCR has already been performed
        if os.path.exists(ocr_dir):
            text_file = os.path.join(ocr_dir, f'{pdf_name}.txt')
            if os.path.exists(text_file):
                print(f"Found {pdf_path}")
                return
        
        # If no existing OCR file found, perform OCR
        print(f"Performing OCR {pdf_path}")
        os.makedirs(ocr_dir, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)
        
        # Configure Tesseract
        if fast_mode:
            custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-_&/ " -c preserve_interword_spaces=1'
        else:
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-_&/ " --dpi 300 -c preserve_interword_spaces=1'
        
        # Perform OCR on each page and combine text
        combined_text = []
        
        for i, image in enumerate(images):
            # Process page with rotation detection
            text = process_page_with_rotation(image, fast_mode=fast_mode, custom_config=custom_config)
            
            # Add page separator
            combined_text.append(text)
            if i < len(images) - 1:
                combined_text.append("\nPDF PAGE BREAK " + str(i+1) + "\n")
        
        # Save combined text to a single file
        text_file = os.path.join(ocr_dir, f'{pdf_name}.txt')
        with open(text_file, 'w') as f:
            f.write('\n'.join(combined_text))
            
    except ImportError:
        print("\nOCR dependencies not found. To use OCR:")
        print("1. Install Tesseract: brew install tesseract")
        print("2. Install Python packages: poetry add pytesseract pdf2image opencv-python numpy")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        # Print the full traceback for debugging
        import traceback
        traceback.print_exc()

def main():
    """Process all PDF files in the data directory."""
    years = [2023, 2024]
    regions = ['R2', 'R5', 'R7']
    
    # Get all PDF files
    pdf_files = []
    for year in years:
        base_data_path = f"data/{year}"
        for region in regions:
            region_data_path = os.path.join(base_data_path, region)
            if not os.path.exists(region_data_path):
                continue
                
            # Process each county
            for county in [d for d in os.listdir(region_data_path) if os.path.isdir(os.path.join(region_data_path, d))]:
                county_data_path = os.path.join(region_data_path, county)
                
                # Process each template folder
                for template in [d for d in os.listdir(county_data_path) if os.path.isdir(os.path.join(county_data_path, d))]:
                    folder = os.path.join(county_data_path, template, 'original')
                    print(f"Processing folder: {folder}")
                    if not os.path.exists(folder):
                        continue
                    pdf_files.extend(glob.glob(os.path.join(folder, '*.pdf')))
    
    if not pdf_files:
        print("No PDF files found")
        return
    
    # Count already processed files
    processed_files = 0
    files_to_process = []
    for pdf_path in pdf_files:
        # Get the template folder (parent of 'original')
        template_dir = os.path.dirname(os.path.dirname(pdf_path))
        pdf_name = os.path.basename(pdf_path)
        text_file = os.path.join(template_dir, 'ocr_output', f'{os.path.splitext(pdf_name)[0]}.txt')
        if os.path.exists(text_file):
            processed_files += 1
            print(f"Found existing OCR text: {text_file}")
        else:
            files_to_process.append(pdf_path)
    
    print(f"\nFound {len(pdf_files)} PDF files total")
    print(f"Already processed: {processed_files}")
    print(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        print("No new files to process")
        return
    
    # Process PDFs in parallel with fast mode
    num_cores = 3
    print(f"\nUsing {num_cores} cores for parallel processing")
    
    # Create partial function with fast mode settings
    process_pdf_fast = partial(process_pdf, dpi=200, fast_mode=True)
    
    with mp.Pool(num_cores) as pool:
        pool.map(process_pdf_fast, files_to_process)
    
    print("\nOCR processing complete")

if __name__ == "__main__":
    main() 