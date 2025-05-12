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

def extract_text_adjacent_to_phrase(text, phrase, direction='right', row_search_text=None, column_search_text=None, item_order=None, ignore_before=None, value_pattern=None, ignore_after=None):
    """
    Extract text adjacent to a phrase in the specified direction.
    
    Args:
        text (str): The text to search in
        phrase (str): The phrase to find
        direction (str): Direction to search ('right', 'below', 'table')
        row_search_text (str): Text to find in the row
        column_search_text (str): Text to find in the column
        item_order (int): Order of the item in a list
        ignore_before (str): Text to ignore before
        value_pattern (str): Regex pattern to match numeric values
        ignore_after (str): Text to ignore after (and including)
        
    Returns:
        str: The extracted text
    """
    if not text or not phrase:
        print('text or phrase not defined')
        return None
        
    # Split text into lines and remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Find the line containing the phrase
    phrase_line_idx = None
    for i, line in enumerate(lines):
        if phrase.lower() in line.lower():
            phrase_line_idx = i
            break
            
    if phrase_line_idx is None:
        print(f'{phrase} not found')
        return None
    
    # if 'DAIRY:' in phrase:
        # print(lines[phrase_line_idx])

    if direction == 'right':
        # Extract text to the right of the phrase
        line = lines[phrase_line_idx]
        # print(line)
        phrase_idx = line.lower().find(phrase.lower())
        if phrase_idx != -1:
            # Get text after the phrase
            # print(line)
            # print(phrase)
            text_after = line[phrase_idx + len(phrase):].strip()
            
            # # Special handling for facility name and address
            # if phrase.lower() in ['facility name:', 'facility address']:
            #     # For facility name, get everything until the next field
            #     if phrase.lower() == 'facility name:':
            #         next_field_idx = text_after.lower().find('facility address')
            #         if next_field_idx != -1:
            #             text_after = text_after[:next_field_idx].strip()
            #     # For facility address, get everything until the next line
            #     elif phrase.lower() == 'facility address':
            #         next_line_idx = text_after.find('\n')
            #         if next_line_idx != -1:
            #             text_after = text_after[:next_line_idx].strip()
            
            # If ignore_after is specified, cut off text at that point
            if ignore_after and not pd.isna(ignore_after):
                ignore_idx = text_after.lower().find(ignore_after.lower())
                if ignore_idx != -1:
                    text_after = text_after[:ignore_idx].strip()
            
            # If we have a value pattern, use it to extract the numeric value
            if value_pattern and not pd.isna(value_pattern):
                try:
                    match = re.search(str(value_pattern), text_after)
                    if match:
                        return match.group(0)
                except re.error:
                    # If pattern is invalid, just return the text
                    return text_after
            
            # Otherwise return the text after the phrase
            return text_after
            
    elif direction == 'below':
        # Extract text from the line below
        if phrase_line_idx + 1 < len(lines):
            next_line = lines[phrase_line_idx + 1].strip()
            
            # If ignore_after is specified, cut off text at that point
            if ignore_after and not pd.isna(ignore_after):
                ignore_idx = next_line.lower().find(ignore_after.lower())
                if ignore_idx != -1:
                    next_line = next_line[:ignore_idx].strip()
            
            # If we have a value pattern, use it to extract the numeric value
            if value_pattern and not pd.isna(value_pattern):
                try:
                    match = re.search(str(value_pattern), next_line)
                    if match:
                        return match.group(0)
                except re.error:
                    # If pattern is invalid, just return the text
                    return next_line
                    
            return next_line
            
    elif direction == 'table':
        # Handle table format
        if row_search_text and column_search_text:
            # Find the row containing row_search_text
            row_idx = None
            for i, line in enumerate(lines):
                if row_search_text.lower() in line.lower():
                    row_idx = i
                    break
                    
            if row_idx is not None:
                # Find the column containing column_search_text
                header_line = lines[row_idx]
                header_parts = [part.strip() for part in header_line.split() if part.strip()]
                
                col_idx = None
                for i, part in enumerate(header_parts):
                    if column_search_text.lower() in part.lower():
                        col_idx = i
                        break
                        
                if col_idx is not None and row_idx + 1 < len(lines):
                    # Get the value from the next line
                    value_line = lines[row_idx + 1]
                    value_parts = [part.strip() for part in value_line.split() if part.strip()]
                    
                    if col_idx < len(value_parts):
                        value = value_parts[col_idx]
                        
                        # If ignore_after is specified, cut off text at that point
                        if ignore_after and not pd.isna(ignore_after):
                            ignore_idx = value.lower().find(ignore_after.lower())
                            if ignore_idx != -1:
                                value = value[:ignore_idx].strip()
                        
                        # If we have a value pattern, use it to extract the numeric value
                        if value_pattern and not pd.isna(value_pattern):
                            try:
                                match = re.search(str(value_pattern), value)
                                if match:
                                    return match.group(0)
                            except re.error:
                                # If pattern is invalid, just return the value
                                return value
                                
                        return value
                        
    return None

def convert_to_numeric(value, data_type):
    """Convert a value to numeric format based on data type."""
    if value is None:
        return 0 if data_type == 'numeric' else None
        
    # Remove any non-numeric characters except decimal point and minus sign
    if data_type == 'numeric':
        # Remove commas and other non-numeric characters
        value = str(value).replace(',', '')
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            return 0
    return value

def find_value_by_text(page_text, row, data_type):
    """Find a value in the text based on the search parameters."""
    if pd.isna(row['row_search_text']):
        return None
        
    # Get the item order
    item_order = row['item_order']
    
    # Check if the search text exists in the page
    if str(row['row_search_text']) in page_text:
        # Extract the text to the right or below where the string was found
        extracted_text = extract_text_adjacent_to_phrase(
            text=page_text,
            phrase=row['row_search_text'],
            direction=row['search_direction'],
            row_search_text=row['row_search_text'],
            column_search_text=row['column_search_text'],
            item_order=row['item_order'],
            ignore_before=row['ignore_before'],
            value_pattern=row['value_pattern']
        )
        
        if extracted_text:
            if pd.isna(item_order) or item_order == -1:
                # If no item order specified, return the extracted text
                return convert_to_numeric(extracted_text, data_type)
            else:
                # If item order specified, split the text and return the specified item
                parts = extracted_text.split()
                if item_order < len(parts):
                    return convert_to_numeric(parts[item_order], data_type)
                    
    return 0 if data_type == 'numeric' else None

def load_ocr_text(pdf_path):
    """Load OCR text from file."""
    pdf_dir = os.path.dirname(pdf_path)
    parent_dir = os.path.dirname(pdf_dir)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Try handwriting_ocr_output first, then ocr_output
    for ocr_dir in ['handwriting_ocr_output', 'ocr_output']:
        text_file = os.path.join(parent_dir, ocr_dir, f'{pdf_name}.txt')
        if os.path.exists(text_file):
            # if ocr_dir == 'handwriting_ocr_output':
                # print('found handwriting output')
            with open(text_file, 'r') as f:
                text = f.read()
                print(f'reading {text_file}')

                # TODO: remove manual fixes here
                text = text.replace("Maxiumu", "Maximum")
                text = text.replace(",", "")
                text = text.replace("=", "")

                return text
    
    print(f"OCR text file not found for {pdf_name}")
    return None

def find_parameter_value(ocr_text, row, param_types=None):
    """Extract a parameter value from OCR text based on the specified row from parameter_locations."""
    if pd.isna(row['search_direction']):
        return np.nan
        
    data_type = param_types.get(row['parameter_key'], 'text')
    
    try:
        # Get the text to search in
        search_text = ocr_text
        if not pd.isna(row['page_search_text']):
            # If page_search_text is provided, use it to find the starting point
            clean_search = ' '.join(row['page_search_text'].split())
            clean_text = ' '.join(ocr_text.split())
            
            pos = clean_text.find(clean_search)
            if pos == -1:
                return np.nan if data_type == 'text' else 0
                
            # Start searching from after the page_search_text
            search_text = ocr_text[pos + len(row['page_search_text']):]
        
        # If no row_search_text, we can't find the value
        if pd.isna(row['row_search_text']):
            return np.nan if data_type == 'text' else 0
            
        # Search for the value in the appropriate text section
        value = find_value_by_text(page_text=search_text, row=row, data_type=data_type)
        return value
            
    except Exception as e:
        print(f"Error processing parameter {row['parameter_key']}: {str(e)}")
        return np.nan if data_type == 'text' else 0

def identify_manifest_pages(ocr_texts):
    """Identify pages that contain manifests by looking for the manifest header."""
    manifest_starts = []
    manifest_header = "Manure / Process Wastewater Tracking Manifest"
    
    for i, text in enumerate(ocr_texts):
        if manifest_header in text and ("NAME OF OPERATOR" in text.upper() or "OPERATOR INFORMATION" in text.upper()):
            manifest_starts.append(i + 1)
    
    return manifest_starts

def process_manifest_pages(ocr_texts, manifest_params):
    """Process a pair of manifest pages and extract parameters."""
    manifest_data = {}
    
    # Process first page parameters
    for _, row in manifest_params.iterrows():
        for page_num in [0, 1]:
            value = find_parameter_value(ocr_texts[page_num], row, param_types={'text': 'text', 'numeric': 'numeric'})
            if value and not pd.isna(value):
                manifest_data[row['parameter_key']] = value
                break
    
    # Extract the date (always on first page)
    text = ocr_texts[0]
    date_match = re.search(r'Last date hauled:\s*(\d{2}/\d{2}/\d{4})', text)
    if date_match:
        manifest_data['Last Date Hauled'] = datetime.strptime(date_match.group(1), '%m/%d/%Y').date()
    
    return manifest_data

def extract_manifests(pdf_path, manifest_params):
    """Extract all manifests from OCR text."""
    manifests = []
    
    try:
        ocr_texts = load_ocr_text(pdf_path)
        if not ocr_texts:
            return manifests
            
        total_pages = len(ocr_texts)
        manifest_starts = identify_manifest_pages(ocr_texts)
        
        for i, start_page in enumerate(manifest_starts):
            if start_page + 1 <= total_pages:
                manifest_data = process_manifest_pages(ocr_texts[start_page-1:start_page+1], manifest_params)
                if manifest_data:
                    manifest_data['Page Numbers'] = f"{start_page}-{start_page+1}"
                    manifests.append(manifest_data)
        
    except Exception as e:
        print(f"Error processing manifests in {pdf_path}: {e}")
        
    return manifests

def process_pdf(pdf_path, template_params, columns, param_types):
    """Process a single PDF file and extract all parameters from OCR text."""
    result = {col: None for col in columns}
    result['filename'] = os.path.basename(pdf_path)
    
    ocr_text = load_ocr_text(pdf_path)
    if not ocr_text:
        return result
    
    # Process main report parameters
    for _, row in template_params.iterrows():
        if not row['manifest_param']:
            param_key = row['parameter_key']
            value = find_parameter_value(ocr_text, row, param_types=param_types)
            result[param_key] = value
    
    # Process manifests if found
    manifest_params = template_params[template_params['manifest_param']]
    if not manifest_params.empty:
        manifests = extract_manifests(pdf_path, manifest_params)
        if manifests:
            result['manifests'] = manifests
    
    return result

def calculate_annual_milk(df):
    """Calculate annual milk production metrics."""
    
    # Calculate reported milk production
    df['Average Milk Production (kg per cow)'] = df['Average Milk Production (lb per cow per day)'] * LBS_TO_KG
    df['Average Milk Production (L per cow)'] = df['Average Milk Production (kg per cow)'] * KG_TO_L_MILK
    df['Reported Annual Milk Production (L)'] = (
        df['Average Milk Production (L per cow)'] * 
        (df['Average Milk Cows'] + df['Average Dry Cows']) * 
        365
    )
    
    # Calculate estimated milk production using default if not reported
    df['Estimated Milk Production (lb per cow per day)'] = df['Average Milk Production (lb per cow per day)'].fillna(DEFAULT_MILK_PRODUCTION)
    df['Estimated Milk Production (kg per cow)'] = df['Estimated Milk Production (lb per cow per day)'] * LBS_TO_KG
    df['Estimated Milk Production (L per cow)'] = df['Estimated Milk Production (kg per cow)'] * KG_TO_L_MILK
    df['Estimated Annual Milk Production (L)'] = (
        df['Estimated Milk Production (L per cow)'] * 
        (df['Average Milk Cows'] + df['Average Dry Cows']) * 
        365
    )
    
    # Calculate milk production discrepancy
    df['Milk Production Discrepancy (L)'] = abs(
        df['Reported Annual Milk Production (L)'] - 
        df['Estimated Annual Milk Production (L)']
    )

def calculate_all_metrics(df):
    """Calculate all possible metrics, filling with NA where not applicable"""
    
    # Calculate milk production metrics
    calculate_annual_milk(df)
    
    # General Order metrics
    df["Total Herd Size"] = (
        df["Average Milk Cows"] + 
        df["Average Dry Cows"] + 
        df["Average Bred Heifers"] + 
        df["Average Heifers"] + 
        df["Average Calves (4-6 mo.)"] + 
        df["Average Calves (0-3 mo.)"] +
        df["Average Other/Unspecified Head"]
    )

    # Common nutrient calculations for both templates
    nutrient_types = ["N", "P", "K", "Salt"]
    for nutrient in nutrient_types:
        # Total Applied
        df[f"Total Applied {nutrient} (lbs)"] = (
            df[f"Applied {nutrient} Dry Manure (lbs)"] + 
            df[f"Applied Process Wastewater {nutrient} (lbs)"]
        )

        # Total Reported
        if nutrient == "N":
            df[f"Total Reported {nutrient} (lbs)"] = (
                df[f"Total Dry Manure Generated {nutrient} After Ammonia Losses (lbs)"] + 
                df[f"Total Process Wastewater Generated {nutrient} (lbs)"]
            )
        else:
            df[f"Total Reported {nutrient} (lbs)"] = (
                df[f"Total Dry Manure Generated {nutrient} (lbs)"] + 
                df[f"Total Process Wastewater Generated {nutrient} (lbs)"]
            )

        # Unaccounted for
        if nutrient == "N":
            df[f"Unaccounted-for {nutrient} (lbs)"] = (
                df[f"Total Dry Manure Generated {nutrient} After Ammonia Losses (lbs)"] + 
                df[f"Total Process Wastewater Generated {nutrient} (lbs)"] - 
                df[f"Total Applied {nutrient} (lbs)"] - 
                df[f"Total Exports {nutrient} (lbs)"]
            )
        else:
            df[f"Unaccounted-for {nutrient} (lbs)"] = (
                df[f"Total Dry Manure Generated {nutrient} (lbs)"] + 
                df[f"Total Process Wastewater Generated {nutrient} (lbs)"] - 
                df[f"Total Applied {nutrient} (lbs)"] - 
                df[f"Total Exports {nutrient} (lbs)"]
            )

    # Wastewater calculations
    df["Total Process Wastewater Generated (L)"] = df["Total Process Wastewater Generated (gals)"] * 3.78541
    
    # Calculate ratio, handling division by zero
    # First try with reported milk production
    df["Ratio of Wastewater to Milk (L/L)"] = (
        df["Total Process Wastewater Generated (L)"] / 
        df["Reported Annual Milk Production (L)"].replace(0, np.nan)
    )
    
    # For facilities without reported milk production, use estimated values
    mask = df["Ratio of Wastewater to Milk (L/L)"].isna()
    df.loc[mask, "Ratio of Wastewater to Milk (L/L)"] = (
        df.loc[mask, "Total Process Wastewater Generated (L)"] / 
        df.loc[mask, "Estimated Annual Milk Production (L)"].replace(0, np.nan)
    )
    
    # Add a column to track whether the ratio is based on reported or estimated milk production
    df["Milk Production Source"] = "Reported"
    df.loc[mask, "Milk Production Source"] = "Estimated"

    # Calculate manure factor
    df["Calculated Manure Factor"] = df.apply(
        lambda row: (
            row["Total Manure Excreted (tons)"] / 
            (
                row["Average Milk Cows"] + 
                row["Average Dry Cows"] + 
                (row["Average Bred Heifers"] + row["Average Heifers"]) * HEIFER_FACTOR +
                (row["Average Calves (4-6 mo.)"] + row["Average Calves (0-3 mo.)"]) * CALF_FACTOR
            )
        ) if (
            row["Average Milk Cows"] + 
            row["Average Dry Cows"] + 
            (row["Average Bred Heifers"] + row["Average Heifers"]) * HEIFER_FACTOR +
            (row["Average Calves (4-6 mo.)"] + row["Average Calves (0-3 mo.)"]) * CALF_FACTOR
        ) > 0 else np.nan,
        axis=1
    )

    # Calculate percentage deviations for nitrogen estimates
    reported_n = df["Total Dry Manure Generated N After Ammonia Losses (lbs)"]

    # Calculate discrepancies for visualization
    df["Nitrogen Discrepancy"] = df["USDA Nitrogen Estimate (lbs)"] - df["Total Dry Manure Generated N After Ammonia Losses (lbs)"]
    df["Wastewater Ratio Discrepancy"] = df["Wastewater to Milk Ratio"] - df["Ratio of Wastewater to Milk (L/L)"]
    df["Manure Factor Discrepancy"] = df["Calculated Manure Factor"] - BASE_MANURE_FACTOR

    df["USDA Nitrogen % Deviation"] = (
        (df["USDA Nitrogen Estimate (lbs)"] - reported_n) / 
        reported_n.replace(0, np.nan) * 100
    )

    df["UCCE Nitrogen % Deviation"] = (
        (df["UCCE Nitrogen Estimate (lbs)"] - reported_n) / 
        reported_n.replace(0, np.nan) * 100
    )

    # Fill NA values with 0 for all calculated columns
    calculated_columns = [
        "Total Herd Size",
        "Average Milk Production (kg per cow)", "Average Milk Production (L per cow)", "Total Annual Milk Production (L)",
        "Total Applied N (lbs)", "Total Applied P (lbs)", "Total Applied K (lbs)", "Total Applied Salt (lbs)",
        "Total Reported N (lbs)", "Total Reported P (lbs)", "Total Reported K (lbs)", "Total Reported Salt (lbs)",
        "Unaccounted-for N (lbs)", "Unaccounted-for P (lbs)", "Unaccounted-for K (lbs)", "Unaccounted-for Salt (lbs)",
        "Total Process Wastewater Generated (L)", "Ratio of Wastewater to Milk (L/L)",
        "Calculated Manure Factor", "Nitrogen Discrepancy", "Wastewater Ratio Discrepancy", "Manure Factor Discrepancy",
        "USDA Nitrogen % Deviation", "UCCE Nitrogen % Deviation"
    ]
    
    for col in calculated_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)

def convert_to_float_list(text, ignore_before=None):
    """Convert text to a list of float numbers, handling various formats and separators."""
    if not text:
        return []
        
    components = [c for c in text.split() if c]
    float_numbers = []
    
    for component in components:
        if ignore_before and isinstance(component, str) and isinstance(ignore_before, str):
            if ignore_before in component:
                _, component = component.split(ignore_before, 1)
                component = component.strip()
            
        cleaned = ''.join(c for c in component if c.isdigit() or c in '.-')
        
        if not any(c.isdigit() for c in cleaned):
            continue
            
        try:
            float_numbers.append(float(cleaned))
        except ValueError:
            continue
    
    return float_numbers

def main(test_mode=False, process_manifests=True):
    """Main function to process all PDF files and extract data."""
    # Define dtype dictionary for parameter_locations.csv
    dtype_dict = {
        'region': str,
        'template': str,
        'parameter_key': str,
        'page_search_text': str,
        'search_direction': str,
        'row_search_text': str,
        'column_search_text': str,
        'item_order': 'Int64',  # Using Int64 to handle NA values
        'ignore_before': str,
        'value_pattern': str
    }
    
    parameter_locations = pd.read_csv('ca_cafo_compliance/parameter_locations.csv', dtype=dtype_dict)
    if 'manifest_param' not in parameter_locations.columns:
        parameter_locations['manifest_param'] = False
    parameters = pd.read_csv('ca_cafo_compliance/parameters.csv')
    param_types = dict(zip(parameters['parameter_key'], parameters['data_type']))
    
    available_templates = parameter_locations['template'].unique()
    num_cores = 1 if test_mode else max(1, mp.cpu_count() - 3)
    
    for year in YEARS:
        base_data_path = f"data/{year}"
        base_output_path = f"outputs/{year}"
        for region in REGIONS:
            region_data_path = os.path.join(base_data_path, region)
            region_output_path = os.path.join(base_output_path, region)
            if not os.path.exists(region_data_path):
                continue
            
            for county in [d for d in os.listdir(region_data_path) if os.path.isdir(os.path.join(region_data_path, d))]:
                county_data_path = os.path.join(region_data_path, county)
                county_output_path = os.path.join(region_output_path, county)
                
                for template in [d for d in os.listdir(county_data_path) if os.path.isdir(os.path.join(county_data_path, d))]:
                    print(f'processing {template} in {county}')
                    if template not in available_templates:
                        continue
                    
                    folder = os.path.join(county_data_path, template)
                    output_folder = os.path.join(county_output_path, template)
                    name = f"{county.capitalize()}_{year}_{template}"
                    
                    template_params = parameter_locations[parameter_locations['template'].isin([template, 'manifest'])]
                    
                    ocr_folder = os.path.join(folder, 'ocr_output')
                    handwriting_ocr_folder = os.path.join(folder, 'handwriting_ocr_output')
                    
                    if not os.path.exists(ocr_folder) and not os.path.exists(handwriting_ocr_folder):
                        continue
                        
                    pdf_files = []
                    for text_file in glob.glob(os.path.join(ocr_folder, '*.txt')):
                        pdf_name = os.path.basename(text_file).replace('.txt', '.pdf')
                        pdf_path = os.path.join(folder, 'original', pdf_name)
                        if os.path.exists(pdf_path):
                            pdf_files.append(pdf_path)
                    
                    if os.path.exists(handwriting_ocr_folder):
                        for text_file in glob.glob(os.path.join(handwriting_ocr_folder, '*.txt')):
                            pdf_name = os.path.basename(text_file).replace('.txt', '.pdf')
                            pdf_path = os.path.join(folder, 'original', pdf_name)
                            if os.path.exists(pdf_path) and pdf_path not in pdf_files:
                                pdf_files.append(pdf_path)
                    
                    if test_mode:
                        pdf_files = pdf_files[:2]
                        
                    if not pdf_files:
                        continue
                        
                    non_manifest_params = template_params[template_params['manifest_param'] == False]
                    columns = non_manifest_params['parameter_key'].unique().tolist()
                    
                    with mp.Pool(num_cores) as pool:
                        process_pdf_partial = partial(process_pdf, template_params=template_params, 
                                                   columns=columns, param_types=param_types)
                        results = pool.map(process_pdf_partial, pdf_files)
                    
                    df = pd.DataFrame([r for r in results if r is not None])
                    
                    if process_manifests and 'manifests' in df.columns:
                        all_manifests = []
                        for _, row in df.iterrows():
                            if isinstance(row['manifests'], list) and row['manifests']:
                                for manifest in row['manifests']:
                                    manifest['Source File'] = row['filename']
                                all_manifests.extend(row['manifests'])
                                
                        if all_manifests:
                            os.makedirs(output_folder, exist_ok=True)
                            manifest_pickle = os.path.join(output_folder, f"{name}_manifests.pickle")
                            with open(manifest_pickle, 'wb') as f:
                                pickle.dump(all_manifests, f)
                            
                            manifest_df = pd.DataFrame(all_manifests)
                            csv_path = os.path.join(output_folder, f"{name}_manifests.csv")
                            manifest_df.to_csv(csv_path, index=False)
                        
                        df = df.drop('manifests', axis=1)
                    
                    for col in df.columns:
                        if param_types.get(col) == 'numeric':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.rename(columns=dict(zip(parameters['parameter_key'], parameters['parameter_name'])))
                    
                    # Calculate estimates for each row
                    estimates = []
                    for _, row in df.iterrows():
                        # Get animal counts, replacing NaN with 0
                        milk_dry_cows = (row.get('Average Milk Cows', 0) or 0) + (row.get('Average Dry Cows', 0) or 0)
                        heifers = (row.get('Average Bred Heifers', 0) or 0) + (row.get('Average Heifers', 0) or 0)
                        calves = (row.get('Average Calves (4-6 mo.)', 0) or 0) + (row.get('Average Calves (0-3 mo.)', 0) or 0)
                        
                        # Calculate manure generation using base factor from conversion_factors.py
                        estimated_manure = (milk_dry_cows * BASE_MANURE_FACTOR) + \
                                         (heifers * HEIFER_FACTOR * BASE_MANURE_FACTOR) + \
                                         (calves * CALF_FACTOR * BASE_MANURE_FACTOR)
                        
                        # Calculate nitrogen estimates
                        # USDA estimate based on manure generation
                        usda_nitrogen = estimated_manure * MANURE_N_CONTENT
                        
                        # UCCE estimate based on animal units
                        animal_units = milk_dry_cows + (heifers * HEIFER_FACTOR) + (calves * CALF_FACTOR)
                        ucce_nitrogen = animal_units * DAYS_PER_YEAR
                        
                        # Calculate wastewater to milk ratio if data available
                        wastewater_ratio = 0
                        if all(x in row for x in ['Average Milk Production (lb/cow/day)', 'Total Process Wastewater Generated (gals)']):
                            milk_production = row['Average Milk Production (lb/cow/day)']
                            wastewater = row['Total Process Wastewater Generated (gals)']
                            
                            # Convert milk to liters and calculate annual production
                            daily_milk_liters = milk_production * LBS_TO_LITERS * milk_dry_cows
                            annual_milk_liters = daily_milk_liters * DAYS_PER_YEAR
                            
                            # Calculate ratio if milk production is non-zero
                            if annual_milk_liters > 0:
                                wastewater_ratio = wastewater / annual_milk_liters
                        
                        estimates.append({
                            'Estimated Total Manure (tons)': estimated_manure,
                            'USDA Nitrogen Estimate (lbs)': usda_nitrogen,
                            'UCCE Nitrogen Estimate (lbs)': ucce_nitrogen,
                            'Wastewater to Milk Ratio': wastewater_ratio
                        })
                    
                    # Add estimates to dataframe
                    estimates_df = pd.DataFrame(estimates)
                    df = pd.concat([df, estimates_df], axis=1)
                    
                    calculate_all_metrics(df)

                    os.makedirs(output_folder, exist_ok=True)
                    df.to_csv(os.path.join(output_folder, f"{name}.csv"), index=False)

if __name__ == "__main__":
    main(test_mode=False, process_manifests=True)