#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import glob
import re
import sys
import contextlib
from conversion_factors import *
import multiprocessing as mp
from functools import partial
import pickle
from datetime import datetime

def extract_text_adjacent_to_phrase(page_text, row, data_type):
    """Extract text either to the right or below the search phrase from OCR text."""
    phrase = row['row_search_text']
    item_order = int(row['item_order']) if not pd.isna(row['item_order']) else -1
    find_value_by = str(row['search_direction']).lower() if not pd.isna(row['search_direction']) else ''
    
    # Try finding the phrase in text
    if phrase in page_text:
        lines = page_text.split('\n')
        for i, line in enumerate(lines):
            if phrase in line:
                if 'right' in find_value_by:
                    # For table structures, split by whitespace and get the appropriate column
                    parts = line.split()
                    if len(parts) > 1:
                        # If we have an item_order, use it to get the right column
                        if item_order >= 0 and item_order < len(parts):
                            # Check if this is a row with animal type (contains letters)
                            if any(c.isalpha() for c in parts[0]):
                                # This is a row with animal type, get the value from the next column
                                if item_order + 1 < len(parts):
                                    return parts[item_order + 1]
                            else:
                                # This is a row without animal type, get the value directly
                                return parts[item_order]
                        # Otherwise try to get text after the phrase
                        else:
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
                    
                    if row.get('separator') == 'line':
                        if item_order == -1:
                            return following_lines[-1]
                        elif item_order < len(following_lines):
                            return following_lines[item_order]
                        else:
                            return "0" if data_type == 'numeric' else None
    
    return "0" if data_type == 'numeric' else None

def find_value_by_text(page_text, row, data_type):
    """Extract a value from OCR text by searching for text and getting a value from the same row or below."""
    if not page_text:
        print("Invalid page text")
        return np.nan if data_type == 'text' else 0
        
    # For exports section, check if there are no exports
    if "NUTRIENT EXPORTS" in page_text:
        if "No solid nutrient exports entered" in page_text and "No liquid nutrient exports entered" in page_text:
            return 0

    item_order = int(row['item_order']) if not pd.isna(row['item_order']) else -1
    # Look for the row text
    if str(row['row_search_text']) in page_text:
        # Extract the text to the right or below where the string was found
        extracted_text = extract_text_adjacent_to_phrase(page_text, row, data_type)
        if extracted_text:
            if pd.isna(item_order) or item_order == -1:
                return extracted_text.strip() # raw text
            else:
                # Convert to list of floats
                values = convert_to_float_list(extracted_text, row['ignore_before'])
                if len(values) > item_order:
                    return values[item_order]
    return np.nan if data_type == 'text' else 0

def load_ocr_text(pdf_path):
    """Load OCR text from file."""
    # Get the directory containing the PDF file
    pdf_dir = os.path.dirname(pdf_path)
    # Get the parent directory (which contains both 'original' and 'ocr_output')
    parent_dir = os.path.dirname(pdf_dir)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # First try handwriting_ocr_output directory
    handwriting_ocr_dir = os.path.join(parent_dir, 'handwriting_ocr_output')
    text_file = os.path.join(handwriting_ocr_dir, f'{pdf_name}.txt')
    
    # If not found in handwriting_ocr_output, try ocr_output
    if not os.path.exists(text_file):
        ocr_dir = os.path.join(parent_dir, 'ocr_output')
        text_file = os.path.join(ocr_dir, f'{pdf_name}.txt')
    
    if not os.path.exists(text_file):
        print(f"OCR text file not found for {pdf_name}")
        return None
    
    with open(text_file, 'r') as f:
        text = f.read()
    
    # Return the entire text as a single string
    return text

def find_page_by_text(ocr_texts, search_text, page_cache, pdf_path):
    """Find the page number containing the specified text in OCR text.
    Returns the page number (1-based index) or None if not found."""
    
    # Check if we've already found this text
    if search_text in page_cache:
        return page_cache[search_text]
    
    # Clean up the search text for comparison
    clean_search = ' '.join(search_text.split())
    
    # Search OCR texts
    for i, text in enumerate(ocr_texts):
        # Clean up the page text for comparison
        clean_page = ' '.join(text.split())
        
        if clean_search in clean_page:
            page_number = i + 1  # Convert to 1-based index
            page_cache[search_text] = page_number
            return page_number
    
    print(f"WARNING: Text '{search_text}' not found in text for {pdf_path}")
    page_cache[search_text] = None  # Cache the failure too
    return None

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

def find_page_number(ocr_texts, row, page_cache, pdf_path):
    """Extract the page number for a parameter based on page_search_text.
    Returns the page number and updated page cache."""
    
    # Return None if page_search_text is NA
    if pd.isna(row['page_search_text']):
        return None, page_cache
        
    search_text = row['page_search_text']
    page_number = find_page_by_text(ocr_texts, search_text, page_cache, pdf_path)
    return page_number, page_cache

def find_parameter_value(ocr_text, row, param_types=None):
    """Extract a parameter value from OCR text based on the specified row from parameter_locations."""
    # Return NA if find_value_by is NA
    if pd.isna(row['search_direction']):
        return np.nan
        
    data_type = param_types.get(row['parameter_key'], 'text')
    
    try:
        # If we have a page_search_text, find the section after it
        if not pd.isna(row['page_search_text']):
            search_text = row['page_search_text']
            clean_search = ' '.join(search_text.split())
            clean_text = ' '.join(ocr_text.split())
            
            # Find the position of the search text
            pos = clean_text.find(clean_search)
            if pos != -1:
                # Get the text after the search text
                section_text = ocr_text[pos + len(search_text):]
                
                # Now look for the row_search_text in this section
                if not pd.isna(row['row_search_text']):
                    row_text = row['row_search_text']
                    clean_row = ' '.join(row_text.split())
                    clean_section = ' '.join(section_text.split())
                    
                    # Find the position of the row text
                    row_pos = clean_section.find(clean_row)
                    if row_pos != -1:
                        # Get the text after the row text
                        value_text = section_text[row_pos + len(row_text):]
                        
                        # Extract value using the value text
                        search_direction = str(row['search_direction']).lower()
                        if search_direction in ['right', 'below', 'table', 'coordinates']:
                            value = find_value_by_text(page_text=value_text, row=row, data_type=data_type)
                            return value
            return np.nan if data_type == 'text' else 0
        else:
            # If no page_search_text, look for row_search_text in the entire text
            if not pd.isna(row['row_search_text']):
                row_text = row['row_search_text']
                clean_row = ' '.join(row_text.split())
                clean_text = ' '.join(ocr_text.split())
                
                # Find the position of the row text
                row_pos = clean_text.find(clean_row)
                if row_pos != -1:
                    # Get the text after the row text
                    value_text = ocr_text[row_pos + len(row_text):]
                    
                    # Extract value using the value text
                    search_direction = str(row['search_direction']).lower()
                    if search_direction in ['right', 'below', 'table', 'coordinates']:
                        value = find_value_by_text(page_text=value_text, row=row, data_type=data_type)
                        return value
            return np.nan if data_type == 'text' else 0
            
    except Exception as e:
        print(f"Error processing parameter {row['parameter_key']}: {str(e)}")
        return np.nan if data_type == 'text' else 0

def identify_manifest_pages(ocr_texts):
    """Identify pages that contain manifests by looking for the manifest header.
    Returns a list of starting page numbers (1-indexed) for each manifest."""
    manifest_starts = []
    manifest_header = "Manure / Process Wastewater Tracking Manifest"
    
    for i, text in enumerate(ocr_texts):
        if manifest_header in text and ("NAME OF OPERATOR" in text.upper() or "OPERATOR INFORMATION" in text.upper()):
            manifest_starts.append(i + 1)  # Convert to 1-indexed
    
    return manifest_starts

def process_manifest_pages(ocr_texts, manifest_params):
    """Process a pair of manifest pages and extract parameters."""
    manifest_data = {}
    
    print("Processing manifest with params:", manifest_params['parameter_key'].tolist())
    
    # Process first page parameters
    for _, row in manifest_params.iterrows():
        # Always try both pages for each parameter since layouts can vary
        for page_num in [0, 1]:  # 0-based page numbers
            value = find_parameter_value(ocr_texts[page_num], row, param_types={'text': 'text', 'numeric': 'numeric'})
            if value and not pd.isna(value):  # If we found a valid value, use it
                manifest_data[row['parameter_key']] = value
                break  # Stop looking for this parameter once found
    
    # Extract the date (always on first page)
    text = ocr_texts[0]
    date_match = re.search(r'Last date hauled:\s*(\d{2}/\d{2}/\d{4})', text)
    if date_match:
        manifest_data['Last Date Hauled'] = datetime.strptime(date_match.group(1), '%m/%d/%Y').date()
    
    print("Extracted manifest data:", manifest_data)
    return manifest_data

def extract_manifests(pdf_path, manifest_params):
    """Extract all manifests from OCR text."""
    manifests = []
    
    try:
        # Load OCR text
        ocr_texts = load_ocr_text(pdf_path)
        if not ocr_texts:
            return manifests
            
        # Each manifest is 2 pages
        total_pages = len(ocr_texts)
        manifest_starts = identify_manifest_pages(ocr_texts)
        print(f"\nFound {len(manifest_starts)} manifests in {pdf_path}")
        
        for i, start_page in enumerate(manifest_starts):
            if start_page + 1 <= total_pages:  # Ensure we have both pages of the manifest
                print(f"Processing manifest {i+1} starting at page {start_page}")
                # Process this manifest
                manifest_data = process_manifest_pages(ocr_texts[start_page-1:start_page+1], manifest_params)
                if manifest_data:
                    manifest_data['Page Numbers'] = f"{start_page}-{start_page+1}"
                    manifests.append(manifest_data)
                    print(f"Added manifest {i+1} to list")
        
        print(f"Total manifests processed: {len(manifests)}")
        
    except Exception as e:
        print(f"Error processing manifests in {pdf_path}: {e}")
        
    return manifests

def process_pdf(pdf_path, template_params, columns, param_types):
    """Process a single PDF file and extract all parameters from OCR text."""
    result = {col: None for col in columns}
    result['filename'] = os.path.basename(pdf_path)
    
    # Load OCR text
    ocr_text = load_ocr_text(pdf_path)
    if not ocr_text:
        return result
    
    # Process main report parameters
    for _, row in template_params.iterrows():
        if not row['manifest_param']:  # Skip manifest parameters
            param_key = row['parameter_key']
            value = find_parameter_value(ocr_text, row, param_types=param_types)
            result[param_key] = value
    
    # Process manifests if found
    manifest_params = template_params[template_params['manifest_param']]
    if not manifest_params.empty:
        print(f"Processing manifests with {len(manifest_params)} parameters")
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
                    
                    # Look for OCR text files in the ocr_output folder
                    ocr_folder = os.path.join(folder, 'ocr_output')
                    handwriting_ocr_folder = os.path.join(folder, 'handwriting_ocr_output')
                    
                    if not os.path.exists(ocr_folder) and not os.path.exists(handwriting_ocr_folder):
                        print(f"No OCR output folders found in {folder}")
                        continue
                        
                    # Get list of PDF files that have corresponding OCR text files
                    pdf_files = []
                    for text_file in glob.glob(os.path.join(ocr_folder, '*.txt')):
                        pdf_name = os.path.basename(text_file).replace('.txt', '.pdf')
                        pdf_path = os.path.join(folder, 'original', pdf_name)
                        if os.path.exists(pdf_path):
                            pdf_files.append(pdf_path)
                    
                    # Also check handwriting_ocr_output if it exists
                    if os.path.exists(handwriting_ocr_folder):
                        for text_file in glob.glob(os.path.join(handwriting_ocr_folder, '*.txt')):
                            pdf_name = os.path.basename(text_file).replace('.txt', '.pdf')
                            pdf_path = os.path.join(folder, 'original', pdf_name)
                            if os.path.exists(pdf_path) and pdf_path not in pdf_files:
                                pdf_files.append(pdf_path)
                    
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
                            with open(manifest_pickle, 'wb') as f:
                                pickle.dump(all_manifests, f)
                            
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

def convert_to_float_list(text, ignore_before=None):
    """Convert text to a list of float numbers, handling various formats and separators."""
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

if __name__ == "__main__":
    # Set test_mode=True to process only 2 files
    main(test_mode=False, process_manifests=True)