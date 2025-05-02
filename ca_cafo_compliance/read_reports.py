#!/usr/bin/env python3

import pandas as pd
from pypdf import PdfReader
import pdfplumber
import sys
import glob
import numpy as np
import os
import re
import contextlib
from conversion_factors import *
import multiprocessing as mp
from functools import partial
import pickle
from datetime import datetime

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output"""
    stderr = sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = stderr

def try_alternate_extraction(pdf_path, page_number):
    """Try different methods to extract text from a PDF page."""
    try:
        # 1. Try PyPDF2 first
        with suppress_stderr():
            reader = PdfReader(pdf_path)
            if page_number < len(reader.pages):
                text = reader.pages[page_number].extract_text()
                if text.strip():
                    return text
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")

    # 2. Try pdfplumber with different text extraction options
    try:
        with suppress_stderr(), pdfplumber.open(pdf_path) as pdf:
            if page_number < len(pdf.pages):
                page = pdf.pages[page_number]
                # Try with different y tolerances
                text = page.extract_text(y_tolerance=5)  # More lenient line joining
                if text.strip():
                    return text
                text = page.extract_text(y_tolerance=10)  # Even more lenient
                if text.strip():
                    return text
                
                # If still no text, try extracting words directly
                words = page.extract_words()
                if words:
                    return ' '.join(word['text'] for word in words)
    except Exception as e:
        print(f"pdfplumber alternate extraction failed: {e}")
    
    print(f"\nWARNING: Text extraction failed for page {page_number} in {pdf_path}")
    print("Consider using OCR if this is a recurring issue.")
    print("To use OCR:")
    print("1. Install Tesseract: brew install tesseract")
    print("2. Install Python packages: poetry add pytesseract pdf2image")
    return None

def convert_to_float_list(text, ignore_before=None):
    if not text:
        return []
        
    # Split text by whitespace and remove empty strings
    components = [c for c in text.split() if c]
    
    float_numbers = []
    for component in components:
        # If ignore_before is specified and is a string, only take text after that character
        if ignore_before and isinstance(component, str) and isinstance(ignore_before, str):
            if ignore_before in component:
                _, component = component.split(ignore_before, 1)
                component = component.strip()
            
        # Remove any non-numeric characters except decimal points and negative signs
        cleaned = ''.join(c for c in component if c.isdigit() or c in '.-')
        
        # Skip if we don't have any digits left
        if not any(c.isdigit() for c in cleaned):
            continue
            
        try:
            # Convert the cleaned component to a float and append to the list
            float_numbers.append(float(cleaned))
        except ValueError:
            # Skip invalid numbers but continue processing
            continue
    
    return float_numbers

def extract_text_adjacent_to_phrase(page, row, data_type):
    """Extract text either to the right or below the search phrase, handling both horizontal and vertical text."""
    
    # Try horizontal text first with different extraction settings
    text = page.extract_text(x_tolerance=3, y_tolerance=3)
    phrase = row['row_search_text']
    item_order = int(row['item_order'])
    find_value_by = row['find_value_by']
    offset = float(row['offset']) if not pd.isna(row['offset']) else 0
    
    # Try finding the phrase in horizontal text
    if phrase in text:
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if phrase in line:
                if 'right' in find_value_by:
                    parts = line.split(phrase)
                    if len(parts) > 1:
                        right_text = parts[1].strip()
                        # For manifest fields, clean up the text
                        if row['template'] == 'manifest_R5_2007':
                            # Remove any non-alphanumeric characters except spaces and common punctuation
                            right_text = re.sub(r'[^a-zA-Z0-9\s\.,\-\'&]', '', right_text)
                            right_text = ' '.join(right_text.split())  # Normalize whitespace
                        return right_text
                elif 'below' in find_value_by:
                    # Skip empty first line
                    current_idx = i + 1
                    if current_idx < len(lines) and not lines[current_idx].strip():
                        current_idx += 1
                    
                    following_lines = []
                    while current_idx < len(lines):
                        next_line = lines[current_idx].strip()
                        if not next_line:
                            break
                        following_lines.append(next_line)
                        current_idx += 1
                    
                    if not following_lines:
                        return "0" if data_type == 'numeric' else None
                    
                    if row['separator'] == 'line':
                        if item_order == -1:
                            return following_lines[-1]
                        elif item_order < len(following_lines):
                            return following_lines[item_order]
                        else:
                            return "0" if data_type == 'numeric' else None
    
    # If not found in horizontal text, try with different extraction settings
    try:
        # Try with different extraction settings
        text = page.extract_text(x_tolerance=1, y_tolerance=1)  # Very precise
        if phrase in text:
            lines = text.split('\n')
            for line in lines:
                if phrase in line:
                    parts = line.split(phrase)
                    if len(parts) > 1:
                        right_text = parts[1].strip()
                        if row['template'] == 'manifest_R5_2007':
                            right_text = re.sub(r'[^a-zA-Z0-9\s\.,\-\'&]', '', right_text)
                            right_text = ' '.join(right_text.split())
                        return right_text
                        
        # Try extracting words directly if still not found
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        phrase_word = None
        for i, word in enumerate(words):
            if phrase in word['text']:
                phrase_word = word
                # Get words at the specified offset distance
                target_x = word['x0'] + offset if 'right' in find_value_by else word['x0']
                target_y = word['y0'] + offset if 'below' in find_value_by else word['y0']
                
                # Find the closest word at the target coordinates
                closest_word = None
                min_distance = float('inf')
                
                for j, other_word in enumerate(words):
                    if j != i:  # Skip the phrase word itself
                        if 'right' in find_value_by:
                            # For right direction, only consider words to the right
                            if other_word['x0'] > word['x1']:
                                distance = abs(other_word['y0'] - word['y0'])  # Vertical distance
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_word = other_word
                        elif 'below' in find_value_by:
                            # For below direction, only consider words below
                            if other_word['y0'] > word['y1']:
                                distance = abs(other_word['x0'] - word['x0'])  # Horizontal distance
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_word = other_word
                
                if closest_word:
                    result = closest_word['text']
                    if row['template'] == 'manifest_R5_2007':
                        result = re.sub(r'[^a-zA-Z0-9\s\.,\-\'&]', '', result)
                        result = ' '.join(result.split())
                    return result
                break
                    
    except Exception as e:
        print(f"Error in alternate extraction: {e}")
    
    return "0" if data_type == 'numeric' else None

def perform_ocr_on_pdf(pdf_path):
    """Perform OCR on a PDF and return the extracted text for each page.
    Saves all OCR text in a single file with page breaks."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        import os
        
        # Set up directories
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        ocr_dir = os.path.join(pdf_dir, 'ocr_output')
        
        # Check if OCR has already been performed
        if os.path.exists(ocr_dir):
            # Look for existing text file
            text_file = os.path.join(ocr_dir, f'{pdf_name}_text.txt')
            if os.path.exists(text_file):
                print(f"Found existing OCR file for {pdf_name}")
                with open(text_file, 'r') as f:
                    text = f.read()
                # Split text by page breaks and return as list
                return text.split('PDF PAGE BREAK')
        
        # If no existing OCR file found, perform OCR
        print(f"Performing new OCR for {pdf_name}")
        os.makedirs(ocr_dir, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        # Perform OCR on each page and combine text
        ocr_texts = []
        combined_text = []
        
        for i, image in enumerate(images):
            # Perform OCR
            text = pytesseract.image_to_string(image)
            ocr_texts.append(text)
            combined_text.append(text)
            if i < len(images) - 1:  # Add page break except after last page
                combined_text.append(f"\nPDF PAGE BREAK {i+1}\n")
        
        # Save combined text to a single file
        text_file = os.path.join(ocr_dir, f'{pdf_name}_text.txt')
        with open(text_file, 'w') as f:
            f.write(''.join(combined_text))
            
        return ocr_texts
    except ImportError:
        print("\nOCR dependencies not found. To use OCR:")
        print("1. Install Tesseract: brew install tesseract")
        print("2. Install Python packages: poetry add pytesseract pdf2image")
        return None
    except Exception as e:
        print(f"Error performing OCR: {e}")
        return None

def find_page_by_text(pdf, search_text, page_cache, ocr_texts=None):
    """Find the page number containing the specified text.
    Returns the page number (1-based index) or None if not found."""
    
    # Check if we've already found this text
    if search_text in page_cache:
        return page_cache[search_text]
            
    # First try normal text extraction
    for i in range(len(pdf.pages)):
        text = pdf.pages[i].extract_text()
        if search_text in text:
            page_number = i + 1  # Convert to 1-based index
            page_cache[search_text] = page_number
            # print(f"Found '{search_text}' on page {page_number}")
            return page_number
            
    # If text not found and OCR texts available, search OCR texts
    if ocr_texts:
        for i, text in enumerate(ocr_texts):
            if search_text in text:
                page_number = i + 1  # Convert to 1-based index
                page_cache[search_text] = page_number
                print(f"Found '{search_text}' using OCR on page {page_number}")
                return page_number
                
    print(f"WARNING: Text '{search_text}' not found in PDF using either method")
    page_cache[search_text] = None  # Cache the failure too
    return None

def is_problematic_unicode(text):
    """Check if text contains problematic Unicode values that indicate PDF encoding issues."""
    if not text:
        return True
    # Check for common problematic Unicode patterns
    if any(ord(c) > 0xf000 for c in text):  # Private Use Area unicode
        return True
    # Check if string is mostly unicode control/formatting characters
    non_printable = sum(1 for c in text if ord(c) > 127)
    if non_printable > len(text) / 2:
        return True
    return False

def find_value_by_coordinates(page, row, data_type):
    """Extract a value from a PDF page by coordinates."""
    if page is None:
        print("Invalid page")
        return np.nan
        
    page_width = float(page.width)
    page_height = float(page.height)
    
    # Get coordinates from row and convert to float
    try:
        x0 = float(row['x0']) if not pd.isna(row['x0']) else 0
        y0 = float(row['y0']) if not pd.isna(row['y0']) else 0
        x1 = float(row['x1']) if not pd.isna(row['x1']) else page_width
        y1 = float(row['y1']) if not pd.isna(row['y1']) else page_height
    except (ValueError, TypeError) as e:
        print(f"Error converting coordinates to float: {e}")
        return np.nan
    
    # Bound coordinates to page dimensions
    x0 = max(0, min(x0, page_width))
    x1 = max(0, min(x1, page_width))
    y0 = max(0, min(y0, page_height))
    y1 = max(0, min(y1, page_height))
    
    text = page.within_bbox((x0, y0, x1, y1)).extract_text()
    
    if data_type == 'text':
        return text
    else:
        text = text.strip().replace(",", "")
        if text == '' or is_problematic_unicode(text):
            return np.nan
        try:
            return float(text)
        except ValueError:
            # If conversion fails, try alternate extraction
            try:
                # Try extracting with different settings
                text = page.within_bbox((x0, y0, x1, y1)).extract_text(x_tolerance=3, y_tolerance=3)
                text = text.strip().replace(",", "")
                if text == '' or is_problematic_unicode(text):
                    return np.nan
                return float(text)
            except ValueError:
                return np.nan

def find_value_by_text(page, row, data_type, ocr_texts=None):
    """Extract a value from a PDF page by searching for text and getting a value from the same row or below."""
    if page is None:
        print("Invalid page")
        return np.nan if data_type == 'text' else 0
        
    # First try normal text extraction
    text = page.extract_text()
    
    # If OCR texts are available, try those too
    if ocr_texts and page.page_number - 1 < len(ocr_texts):
        ocr_text = ocr_texts[page.page_number - 1]
        if ocr_text:
            text = ocr_text  # Use OCR text if available
    
    # For exports section, check if there are no exports
    if "NUTRIENT EXPORTS" in text:
        if "No solid nutrient exports entered" in text and "No liquid nutrient exports entered" in text:
            return 0

    item_order = int(row['item_order'])    
    # Look for the row text
    if str(row['row_search_text']) in text:
        # Extract the text to the right or below where the string was found
        extracted_text = extract_text_adjacent_to_phrase(page, row, data_type)
        if extracted_text:
            if pd.isna(item_order) or item_order == -1:
                return extracted_text.strip() # raw text
            else:
                # Convert to list of floats
                values = convert_to_float_list(extracted_text, row['ignore_before'])
                if len(values) > item_order:
                    return values[item_order]
    return np.nan if data_type == 'text' else 0

def find_value_from_table(page, row, data_type, table_cache=None):
    """Extract a value from a table in the PDF page based on row and column labels."""
    if page is None:
        print("Invalid page")
        return np.nan
    
    row_search_text = row['row_search_text']
    column_search_text = row['column_search_text']

    # Define table extraction settings based on find_value_by parameter
    table_settings = {"snap_tolerance": 4}
    if 'vert_lines' in row['find_value_by']:
        table_settings["vertical_strategy"] = "lines"
    
    # Create cache key from page number and settings
    cache_key = (page.page_number, frozenset(table_settings.items()))
    
    # Try to get tables from cache first
    if table_cache is not None and cache_key in table_cache:
        tables = table_cache[cache_key]
    else:
        # Extract tables with appropriate settings
        tables = page.extract_tables(table_settings)
        if table_cache is not None:
            table_cache[cache_key] = tables
    
    # Find the table containing the search text
    target_table = None
    for table in tables:
        # Convert table to string representation to search
        table_str = '\n'.join([' '.join(filter(None, row)) for row in table])
        if row_search_text in table_str and column_search_text in table_str:
            target_table = table
            break
    
    if target_table is None:
        print(f"Table not found with search texts: {row_search_text}, {column_search_text}")
        return np.nan

    # Find row index - look for empty row_search_text or match
    row_idx = None
    if pd.isna(row_search_text) or row_search_text == '':
        row_idx = 1  # Use first data row if no row search text
    else:
        for i, row in enumerate(target_table):
            if row_search_text in ' '.join(filter(None, row)):
                row_idx = i
                break
            
    # Find column index
    col_idx = None
    header_row = target_table[0]  # Assume first row is header
    for i, col in enumerate(header_row):
        if col and column_search_text in col:
            col_idx = i
            break
            
    if row_idx is None or col_idx is None:
        print(f"Could not find row_idx={row_idx} or col_idx={col_idx}")
        return np.nan
        
    value = target_table[row_idx][col_idx] # value at intersection
    
    if value is None:
        return np.nan
        
    value = str(value).strip()
    if data_type == 'text':
        return value
    else:
        try:
            return float(value.replace(',', ''))
        except (ValueError, AttributeError):
            return np.nan

def find_page_number(pdf, row, page_cache):
    """Extract the page number for a parameter based on find_page_by method.
    Returns the page number and updated page cache."""
    
    # Return None if find_page_by is NA
    if pd.isna(row['find_page_by']):
        return None, page_cache
        
    if row['find_page_by'] == 'number':
        page_number = int(row['page_number_or_text'])
        page_cache[row['page_number_or_text']] = page_number
        return page_number, page_cache
    elif row['find_page_by'] == 'text':
        search_text = row['page_number_or_text']
        page_number = find_page_by_text(pdf, search_text, page_cache)
        return page_number, page_cache
    else:
        print(f"Unknown find_page_by method: {row['find_page_by']}")
        return None, page_cache

def find_parameter_value(pdf, row, page_number=None, param_types=None, table_cache=None, ocr_texts=None):
    """Extract a parameter value from a PDF based on the specified row from parameter_locations."""
    # Return NA if find_value_by is NA
    if pd.isna(row['find_value_by']):
        return np.nan
        
    data_type = param_types.get(row['parameter_key'], 'text')
    
    # Skip if no valid page number
    if page_number is None:
        return np.nan if data_type == 'text' else 0
        
    try:
        # Convert from 1-based to 0-based index for pdfplumber
        plumber_page_number = page_number - 1
        if plumber_page_number < 0 or plumber_page_number >= len(pdf.pages):
            print(f"WARNING: Invalid page number {page_number} for parameter {row['parameter_key']}")
            return np.nan if data_type == 'text' else 0
            
        # Get the page
        page = pdf.pages[plumber_page_number]
        
        # Extract value using the page
        if 'text' in row['find_value_by']:
            value = find_value_by_text(page=page, row=row, data_type=data_type, ocr_texts=ocr_texts)
        elif row['find_value_by'] == 'coordinates':
            value = find_value_by_coordinates(page=page, row=row, data_type=data_type)
        elif 'table' in row['find_value_by']:
            value = find_value_from_table(page=page, row=row, data_type=data_type, table_cache=table_cache)
        else:
            print(f"Unknown find_value_by method: {row['find_value_by']}")
            return np.nan if data_type == 'text' else 0
            
        return value
        
    except Exception as e:
        print(f"Error processing page {page_number} for parameter {row['parameter_key']}: {str(e)}")
        return np.nan if data_type == 'text' else 0
    
def identify_manifest_pages(pdf):
    """Identify pages that contain manifests by looking for the manifest header.
    Returns a list of starting page numbers (1-indexed) for each manifest."""
    manifest_starts = []
    manifest_header = "Manure / Process Wastewater Tracking Manifest"
    
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if manifest_header in text and "NAME OF OPERATOR" in text.upper() or "OPERATOR INFORMATION" in text.upper():
            # print('found manifest')
            manifest_starts.append(i + 1)  # Convert to 1-indexed
    
    return manifest_starts

def process_manifest_pages(pdf, manifest_params):
    """Process a pair of manifest pages and extract parameters."""
    manifest_data = {}
    
    print("Processing manifest with params:", manifest_params['parameter_key'].tolist())
    
    # Process first page parameters
    for _, row in manifest_params.iterrows():
        # Always try both pages for each parameter since layouts can vary
        for page_num in [0, 1]:  # 0-based page numbers
            value = find_parameter_value(pdf, row, page_number=page_num+1, param_types={'text': 'text', 'numeric': 'numeric'})
            if value and not pd.isna(value):  # If we found a valid value, use it
                manifest_data[row['parameter_key']] = value
                break  # Stop looking for this parameter once found
    
    # Extract the date (always on first page)
    text = pdf.pages[0].extract_text()
    date_match = re.search(r'Last date hauled:\s*(\d{2}/\d{2}/\d{4})', text)
    if date_match:
        manifest_data['Last Date Hauled'] = datetime.strptime(date_match.group(1), '%m/%d/%Y').date()
    
    print("Extracted manifest data:", manifest_data)
    return manifest_data

def extract_manifests(pdf_path, manifest_params):
    """Extract all manifests from a PDF file."""
    manifests = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Each manifest is 2 pages
            total_pages = len(pdf.pages)
            manifest_starts = identify_manifest_pages(pdf)
            print(f"\nFound {len(manifest_starts)} manifests in {pdf_path}")
            
            for i, start_page in enumerate(manifest_starts):
                if start_page + 1 <= total_pages:  # Ensure we have both pages of the manifest
                    print(f"Processing manifest {i+1} starting at page {start_page}")
                    # Create a temporary PDF object with just these 2 pages
                    manifest_pdf = type('obj', (), {'pages': pdf.pages[start_page-1:start_page+1]})
                    
                    # Process this manifest
                    manifest_data = process_manifest_pages(manifest_pdf, manifest_params)
                    if manifest_data:
                        manifest_data['Page Numbers'] = f"{start_page}-{start_page+1}"
                        manifests.append(manifest_data)
                        print(f"Added manifest {i+1} to list")
            
            print(f"Total manifests processed: {len(manifests)}")
            
    except Exception as e:
        print(f"Error processing manifests in {pdf_path}: {e}")
        
    return manifests

def save_manifests(manifests, output_path):
    """Save manifests to a pickle file."""
    print('saving manifests')
    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(manifests, f)

def process_pdf(pdf_path, template_params, columns, param_types):
    """Process a single PDF file and extract all parameters."""
    result = {col: None for col in columns}
    result['filename'] = os.path.basename(pdf_path)
    print(f"Processing {pdf_path}")
    
    # Load OCR texts if available
    ocr_texts = None
    ocr_dir = os.path.join(os.path.dirname(pdf_path), 'ocr_output')
    if os.path.exists(ocr_dir):
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        text_file = os.path.join(ocr_dir, f'{pdf_name}_text.txt')
        if os.path.exists(text_file):
            with open(text_file, 'r') as f:
                ocr_texts = f.read().split('PDF PAGE BREAK')
    
    with suppress_stderr(), pdfplumber.open(pdf_path) as pdf:
        # First identify manifest pages
        manifest_starts = identify_manifest_pages(pdf)
        total_pages = len(pdf.pages)
        
        # If we found manifests, adjust the page range for the main report
        main_report_pages = total_pages
        if manifest_starts:
            main_report_pages = manifest_starts[0] - 1
            print(f"Found {len(manifest_starts)} manifests starting at page {manifest_starts[0]}")
        
        # Create a temporary PDF object with just the main report pages
        main_pdf = type('obj', (), {'pages': pdf.pages[:main_report_pages]})
        
        # Initialize caches
        page_cache = {}
        table_cache = {}
        
        # Process main report parameters
        for _, row in template_params.iterrows():
            if not row['manifest_param']:  # Skip manifest parameters
                param_key = row['parameter_key']
                page_number, page_cache = find_page_number(main_pdf, row, page_cache)
                # Pass table_cache only if we need it
                if not pd.isna(row['find_value_by']) and 'table' in row['find_value_by']:
                    value = find_parameter_value(main_pdf, row, page_number=page_number, param_types=param_types, table_cache=table_cache, ocr_texts=ocr_texts)
                else:
                    value = find_parameter_value(main_pdf, row, page_number=page_number, param_types=param_types, ocr_texts=ocr_texts)
                result[param_key] = value
        
        # Process manifests if found
        manifest_params = template_params[template_params['manifest_param']]
        if len(manifest_starts) > 0 and not manifest_params.empty:
            print(f"Processing {len(manifest_starts)} manifests with {len(manifest_params)} parameters")
            manifests = extract_manifests(pdf_path, manifest_params)
            if manifests:
                result['manifests'] = manifests
                print(f"Added {len(manifests)} manifests to result")
            
    return result

def main(test_mode=False, process_manifests=True):
    years = [2023, 2024]
    regions = ['R2', 'R3', 'R5', 'R7']
    parameter_locations = pd.read_csv('ca_cafo_compliance/parameter_locations.csv')
    parameter_locations['item_order'] = parameter_locations['item_order'].fillna(-1).astype(float)
    # Ensure manifest_param column exists with default False
    if 'manifest_param' not in parameter_locations.columns:
        parameter_locations['manifest_param'] = False
    parameters = pd.read_csv('ca_cafo_compliance/parameters.csv')
    param_types = dict(zip(parameters['parameter_key'], parameters['data_type']))
    
    available_templates = parameter_locations['template'].unique()
    
    if test_mode:
        num_cores = 1
    else:
        num_cores = max(1, mp.cpu_count() - 3)
        
    print(f"\nUsing {num_cores} cores for parallel processing")
    
    for year in years:
        base_data_path = f"data/{year}"
        base_output_path = f"outputs/{year}"
        for region in regions:
            region_data_path = os.path.join(base_data_path, region)
            region_output_path = os.path.join(base_output_path, region)
            # Skip if region folder doesn't exist
            if not os.path.exists(region_data_path):
                print(f"Region folder not found: {region_data_path}")
                continue
            
            # Process each county
            for county in [d for d in os.listdir(region_data_path) if os.path.isdir(os.path.join(region_data_path, d))]:
                county_data_path = os.path.join(region_data_path, county)
                county_output_path = os.path.join(region_output_path, county)
                
                # Process each template folder
                for template in [d for d in os.listdir(county_data_path) if os.path.isdir(os.path.join(county_data_path, d))]:
                    # Skip if template is not in parameter_locations
                    if template not in available_templates:
                        print(f"Skipping template '{template}' - not found in parameter_locations")
                        continue
                    
                    print(f"\nProcessing {template} template in {county}")
                    
                    folder = os.path.join(county_data_path, template)
                    output_folder = os.path.join(county_output_path, template)
                    name = f"{county.capitalize()}_{year}_{template}"
                    
                    template_params = parameter_locations[parameter_locations['template'].isin([template, 'manifest'])]
                    pdf_files = glob.glob(os.path.join(folder, '*.pdf'))
                    
                    if test_mode:
                        max_cores = 1
                        print(f"Running in test mode - processing only 2 files for {template}")
                        pdf_files = pdf_files[:2]
                        
                    if not pdf_files:
                        print(f"No PDF files found in {folder}")
                        continue
                        
                    # Get non-manifest columns for the main DataFrame
                    non_manifest_params = template_params[template_params['manifest_param'] == False]
                    columns = non_manifest_params['parameter_key'].unique().tolist()
                    
                    # Process PDFs in parallel
                    with mp.Pool(num_cores) as pool:
                        # Create a partial function with template_params, columns, and param_types
                        process_pdf_partial = partial(process_pdf, template_params=template_params, 
                                                   columns=columns, param_types=param_types)
                        # Process PDFs in parallel and collect results
                        results = pool.map(process_pdf_partial, pdf_files)
                    
                    # Convert results to DataFrame
                    df = pd.DataFrame([r for r in results if r is not None])
                    
                    # Extract manifests to separate DataFrame and pickle file
                    if process_manifests and 'manifests' in df.columns:
                        all_manifests = []
                        print("\nProcessing manifests from results:")
                        for _, row in df.iterrows():
                            if isinstance(row['manifests'], list) and row['manifests']:
                                print(f"Found {len(row['manifests'])} manifests in {row['filename']}")
                                for manifest in row['manifests']:
                                    manifest['Source File'] = row['filename']
                                all_manifests.extend(row['manifests'])
                                
                        if all_manifests:
                            print(f"\nTotal manifests collected: {len(all_manifests)}")
                            os.makedirs(output_folder, exist_ok=True)
                            # Save manifests to pickle
                            manifest_pickle = os.path.join(output_folder, f"{name}_manifests.pickle")
                            print(f"Saving manifests to {manifest_pickle}")
                            save_manifests(all_manifests, manifest_pickle)
                            
                            # Save manifests to CSV
                            manifest_df = pd.DataFrame(all_manifests)
                            csv_path = os.path.join(output_folder, f"{name}_manifests.csv")
                            print(f"Saving manifests to {csv_path}")
                            manifest_df.to_csv(csv_path, index=False)
                        else:
                            print("No manifests found in results")
                        
                        # Remove manifests column from main DataFrame
                        df = df.drop('manifests', axis=1)
                    
                    # Convert numeric fields to numeric based on data_type in parameters
                    for col in df.columns:
                        if param_types.get(col) == 'numeric':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Rename columns from parameter_key to parameter_name
                    df = df.rename(columns=dict(zip(parameters['parameter_key'], parameters['parameter_name'])))

                    calculate_all_metrics(df)

                    os.makedirs(output_folder, exist_ok=True)
                    df.to_csv(os.path.join(output_folder, f"{name}.csv"), index=False)

def calculate_all_metrics(df):
    """Calculate all possible metrics, filling with NA where not applicable"""
    
    # General Order metrics
    try:
        df["Total Herd Size"] = df["Average Milk Cows"].fillna(0) + df["Average Dry Cows"].fillna(0) + \
                               df["Average Bred Heifers"].fillna(0) + df["Average Heifers"].fillna(0) + \
                               df["Average Calves (4-6 mo.)"].fillna(0) + df["Average Calves (0-3 mo.)"].fillna(0) +\
                               df["Average Other"].fillna(0)
    except:
        df["Total Herd Size"] = np.nan

    # Innovative Ag metrics
    try:
        df["Total Head"] = df["Average No. of Head"]
        df["Total AUs"] = df["Average No. of AUs"]
    except:
        df["Total Head"] = np.nan
        df["Total AUs"] = np.nan

    # Common nutrient calculations for both templates
    nutrient_types = ["N", "P", "K", "Salt"]
    for nutrient in nutrient_types:
        # Total Applied
        try:
            df[f"Total Applied {nutrient} (lbs)"] = df[f"Applied {nutrient} Dry Manure (lbs)"].fillna(0) + \
                                                   df[f"Applied Process Wastewater {nutrient} (lbs)"].fillna(0)
        except:
            df[f"Total Applied {nutrient} (lbs)"] = np.nan

        # Total Reported
        try:
            if nutrient == "N":
                df[f"Total Reported {nutrient} (lbs)"] = df[f"Total Dry Manure Generated {nutrient} After Ammonia Losses (lbs)"].fillna(0) + \
                                                        df[f"Total Process Wastewater Generated {nutrient} (lbs)"].fillna(0)
            else:
                df[f"Total Reported {nutrient} (lbs)"] = df[f"Total Dry Manure Generated {nutrient} (lbs)"].fillna(0) + \
                                                        df[f"Total Process Wastewater Generated {nutrient} (lbs)"].fillna(0)
        except:
            df[f"Total Reported {nutrient} (lbs)"] = np.nan

        # Unaccounted for
        try:
            if nutrient == "N":
                df[f"Unaccounted-for {nutrient} (lbs)"] = df[f"Total Dry Manure Generated {nutrient} After Ammonia Losses (lbs)"].fillna(0) + \
                                                         df[f"Total Process Wastewater Generated {nutrient} (lbs)"].fillna(0) - \
                                                         df[f"Total Applied {nutrient} (lbs)"].fillna(0) - \
                                                         df[f"Total Exports {nutrient} (lbs)"].fillna(0)
            else:
                df[f"Unaccounted-for {nutrient} (lbs)"] = df[f"Total Dry Manure Generated {nutrient} (lbs)"].fillna(0) + \
                                                         df[f"Total Process Wastewater Generated {nutrient} (lbs)"].fillna(0) - \
                                                         df[f"Total Applied {nutrient} (lbs)"].fillna(0) - \
                                                         df[f"Total Exports {nutrient} (lbs)"].fillna(0)
        except:
            df[f"Unaccounted-for {nutrient} (lbs)"] = np.nan

    # Wastewater calculations
    try:
        df["Total Process Wastewater Generated (L)"] = df["Total Process Wastewater Generated (gals)"].fillna(0) * 3.78541
    except:
        df["Total Process Wastewater Generated (L)"] = np.nan

    try:
        df["Ratio of Wastewater to Milk (L/L)"] = df["Total Process Wastewater Generated (L)"] / df["Total Annual Milk Production (L)"]
    except:
        df["Ratio of Wastewater to Milk (L/L)"] = np.nan

if __name__ == "__main__":
    # Set test_mode=True to process only 5 files
    main(test_mode=True, process_manifests=True)