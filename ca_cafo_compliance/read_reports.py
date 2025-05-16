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
import time
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from geopy.geocoders import Nominatim, ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

read_reports = True
consolidate_data = True

def extract_value_with_pattern(text):
    """Extract the first number (int or float) from the text."""
    if not isinstance(text, str):
        text = str(text)
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        return match.group(0)
    return text

def extract_from_line(line, ignore_before=None, ignore_after=None):
    if not isinstance(line, str):
        line = str(line)
    if ignore_after:
        idx = line.lower().find(str(ignore_after).lower())
        if idx != -1:
            line = line[:idx].strip()
    if ignore_before:
        idx = line.lower().find(str(ignore_before).lower())
        if idx != -1:
            line = line[idx + len(str(ignore_before)) :].strip()
    return extract_value_with_pattern(line)

def find_last_data_row(lines, start_idx, stop_phrases):
    for j in range(start_idx+1, len(lines)):
        line = lines[j]
        if not isinstance(line, str):
            line = str(line)
        if not line.strip() or any(phrase in line.lower() for phrase in stop_phrases):
            return j-1
    return len(lines)-1

def extract_value_from_line(line, item_order=None, ignore_before=None, ignore_after=None):
    """Extract value from a line using item_order, ignore_before, and ignore_after. If none, return full line."""
    if not isinstance(line, str):
        line = str(line)
    if item_order is None and not ignore_before and not ignore_after:
        return line
    # Optionally trim before/after
    if ignore_after:
        idx = line.lower().find(str(ignore_after).lower())
        if idx != -1:
            line = line[:idx].strip()
    if ignore_before:
        idx = line.lower().find(str(ignore_before).lower())
        if idx != -1:
            line = line[idx + len(str(ignore_before)) :].strip()
    # Optionally select item by order
    if item_order is not None and not pd.isna(item_order):
        parts = [p for p in line.split() if p]
        idx = int(item_order)
        if 0 <= idx < len(parts):
            return parts[idx]
        return ''
    return line

def extract_text_adjacent_to_phrase(text, phrase, direction='right', row_search_text=None, column_search_text=None, item_order=None, ignore_before=None, ignore_after=None):
    if not text or not phrase:
        return None
    lines = [str(line).strip() for line in text.split('\n') if str(line).strip()]
    phrase_line_idx = next((i for i, line in enumerate(lines) if isinstance(line, str) and phrase.lower() in line.lower()), None)
    if phrase_line_idx is None:
        return None
    if direction == 'right':
        line = lines[phrase_line_idx]
        phrase_idx = line.lower().find(phrase.lower())
        if phrase_idx != -1:
            text_after = line[phrase_idx + len(phrase):].strip()
            return extract_value_from_line(text_after, item_order, ignore_before, ignore_after)
    elif direction == 'below':
        # Find the next non-blank line after the phrase
        next_line = None
        for j in range(phrase_line_idx + 1, len(lines)):
            if lines[j].strip():
                next_line = lines[j].strip()
                break
        if next_line is not None:
            return extract_value_from_line(next_line, item_order, ignore_before, ignore_after)
    elif direction == 'table':
        if row_search_text and column_search_text:
            row_idx = next((i for i, line in enumerate(lines) if isinstance(line, str) and row_search_text.lower() in line.lower()), None)
            if row_idx is not None:
                header_parts = [part.strip() for part in str(lines[row_idx]).split() if part.strip()]
                col_idx = next((i for i, part in enumerate(header_parts) if column_search_text.lower() in part.lower()), None)
                if col_idx is not None and row_idx + 1 < len(lines):
                    value_parts = [part.strip() for part in str(lines[row_idx + 1]).split() if part.strip()]
                    if col_idx < len(value_parts):
                        return extract_value_from_line(value_parts[col_idx], item_order, ignore_before, ignore_after)
    elif direction == 'above':
        if phrase_line_idx > 0:
            value_line = lines[phrase_line_idx - 1]
            return extract_value_from_line(value_line, item_order, ignore_before, ignore_after)
    return None

def find_value_by_text(page_text, row, data_type):
    if pd.isna(row['row_search_text']):
        return None
    extracted_text = extract_text_adjacent_to_phrase(
        text=page_text,
        phrase=row['row_search_text'],
        direction=row['search_direction'],
        row_search_text=row['row_search_text'],
        column_search_text=row['column_search_text'],
        item_order=row['item_order'],
        ignore_before=row['ignore_before'],
        ignore_after=row['ignore_after'] if 'ignore_after' in row else None
    )
    if extracted_text:
        item_order = row['item_order']
        if pd.isna(item_order) or item_order == -1:
            return convert_to_numeric(extracted_text, data_type)
        else:
            parts = extracted_text.split()
            if item_order < len(parts):
                return convert_to_numeric(parts[item_order], data_type)
    return 0 if data_type == 'numeric' else None

def convert_to_numeric(value, data_type):
    """Convert a value to numeric format based on data type."""
    if value is None:
        return 0 if data_type == 'numeric' else None
        
    # Remove non-numeric characters
    if data_type == 'numeric':
        value = str(value).replace(',', '')
        try:
            return float(value)
        except ValueError:
            return 0
    return value

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
                # print(f'reading {text_file}')

                # TODO: remove manual fixes here
                text = text.replace("Maxiumu", "Maximum")
                text = text.replace("|", "")
                text = text.replace(",", "")
                text = text.replace("=", "")
                text = text.replace(":", "")
                text = text.replace("Ibs", "lbs")
                text = text.replace("/bs", "lbs")
                text = text.replace("FaciIity", "Facility")
                text = text.replace("  ", " ")
                
                # Remove blank lines
                text = '\n'.join([line for line in text.split('\n') if line.strip()])

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

def process_pdf(pdf_path, template_params, columns, param_types):
    """Process a single PDF file and extract all parameters from OCR text."""
    result = {col: None for col in columns}
    result['filename'] = os.path.basename(pdf_path)
    
    ocr_text = load_ocr_text(pdf_path)
    if not ocr_text:
        return result
    
    # Process main report parameters
    for _, row in template_params.iterrows():
        param_key = row['parameter_key']
        value = find_parameter_value(ocr_text, row, param_types=param_types)
        result[param_key] = value
    
    return result

def calculate_annual_milk(df):
    """Calculate annual milk production metrics using parameter_key columns only."""
    # Reported milk production
    df['avg_milk_prod_kg_per_cow'] = safe_calc(
        df, ['avg_milk_lb_per_cow_day'],
        lambda d: d['avg_milk_lb_per_cow_day'] * LBS_TO_KG
    )
    df['avg_milk_prod_l_per_cow'] = safe_calc(
        df, ['avg_milk_lb_per_cow_day'],
        lambda d: d['avg_milk_lb_per_cow_day'] * LBS_TO_KG * KG_PER_L_MILK
    )
    df['reported_annual_milk_production_l'] = safe_calc(
        df, ['avg_milk_lb_per_cow_day', 'avg_milk_cows', 'avg_dry_cows'],
        lambda d: d['avg_milk_lb_per_cow_day'] * LBS_TO_KG * KG_PER_L_MILK * (d['avg_milk_cows'].fillna(0) + d['avg_dry_cows'].fillna(0)) * 365
    )
    # Estimated milk production
    df['estimated_milk_lb_per_cow_day'] = df['avg_milk_lb_per_cow_day'].fillna(DEFAULT_MILK_PRODUCTION) if 'avg_milk_lb_per_cow_day' in df.columns else DEFAULT_MILK_PRODUCTION
    df['estimated_milk_kg_per_cow'] = df['estimated_milk_lb_per_cow_day'] * LBS_TO_KG
    df['estimated_milk_l_per_cow'] = df['estimated_milk_kg_per_cow'] * KG_PER_L_MILK
    df['estimated_annual_milk_production_l'] = safe_calc(
        df, ['estimated_milk_l_per_cow', 'avg_milk_cows', 'avg_dry_cows'],
        lambda d: d['estimated_milk_l_per_cow'] * (d['avg_milk_cows'].fillna(0) + d['avg_dry_cows'].fillna(0)) * 365
    )
    # Discrepancy
    df['milk_production_discrepancy_l'] = safe_calc(
        df, ['reported_annual_milk_production_l', 'estimated_annual_milk_production_l'],
        lambda d: abs(d['reported_annual_milk_production_l'].fillna(0) - d['estimated_annual_milk_production_l'].fillna(0))
    )

def safe_calc(df, keys, func, default=np.nan):
    if all(k in df.columns for k in keys):
        return func(df)
    return default

def calculate_all_metrics(df):
    """Calculate all possible metrics, filling with NA where not applicable, using parameter_key columns."""
    print("\n=== Debug: calculate_all_metrics ===")
    print(f"Input dataframe shape: {df.shape}")
    print(f"Input columns: {df.columns.tolist()}")

    calculate_annual_milk(df)

    # General Order metrics
    herd_keys = [
        "avg_milk_cows", "avg_dry_cows", "avg_bred_heifers",
        "avg_heifers", "avg_calves_4_6_mo", "avg_calves_0_3_mo", "avg_other"
    ]
    df["total_herd_size"] = safe_calc(
        df, herd_keys,
        lambda d: sum(d[k].fillna(0) for k in herd_keys if k in d.columns),
        default=0
    )

    nutrient_types = ["n", "p", "k", "salt"]
    for nutrient in nutrient_types:
        # Total Applied
        dry_key = f"applied_{nutrient}_dry_manure_lbs"
        ww_key = f"applied_ww_{nutrient}_lbs"
        total_applied_key = f"total_applied_{nutrient}_lbs"
        df[total_applied_key] = safe_calc(
            df, [dry_key, ww_key],
            lambda d: d[dry_key].fillna(0) + d[ww_key].fillna(0)
        )

        if nutrient == "n":
            dry_key_reported = "total_manure_gen_n_after_nh3_losses_lbs"
        else:
            dry_key_reported = f"total_manure_gen_{nutrient}_lbs"
        ww_key_reported = f"total_ww_gen_{nutrient}_lbs"
        total_reported_key = f"total_reported_{nutrient}_lbs"
        df[total_reported_key] = safe_calc(
            df, [dry_key_reported, ww_key_reported],
            lambda d: d[dry_key_reported].fillna(0) + d[ww_key_reported].fillna(0)
        )

        # Unaccounted for
        exports_key = f"total_exports_{nutrient}_lbs"
        unaccounted_key = f"unaccounted_for_{nutrient}_lbs"
        df[unaccounted_key] = safe_calc(
            df, [dry_key_reported, ww_key_reported, total_applied_key, exports_key],
            lambda d: d[dry_key_reported].fillna(0) + d[ww_key_reported].fillna(0) - d[total_applied_key].fillna(0) - d[exports_key].fillna(0)
        )

    df["total_ww_gen_liters"] = safe_calc(
        df, ["total_ww_gen_gals"],
        lambda d: d["total_ww_gen_gals"] * 3.78541
    )

    # Ratio calculations
    df["ww_to_reported_milk"] = safe_calc(
        df, ["total_ww_gen_liters", "reported_annual_milk_production_l"],
        lambda d: d["total_ww_gen_liters"] / d["reported_annual_milk_production_l"].replace(0, np.nan)
    )
    df["ww_to_estimated_milk"] = safe_calc(
        df, ["total_ww_gen_liters", "estimated_annual_milk_production_l"],
        lambda d: d["total_ww_gen_liters"] / d["estimated_annual_milk_production_l"].replace(0, np.nan)
    )
    # Remove old ratio_ww_to_milk_l_per_l logic
    # Update any references to wastewater_to_milk_ratio if it refers to estimated ratio
    # (If wastewater_to_milk_ratio is used elsewhere for a different purpose, leave as is)
    # Update discrepancy calculation
    df["wastewater_ratio_discrepancy"] = safe_calc(
        df, ["ww_to_estimated_milk", "ww_to_reported_milk"],
        lambda d: d["ww_to_estimated_milk"] - d["ww_to_reported_milk"]
    )

    manure_keys = [
        "total_manure_excreted_tons", "avg_milk_cows", "avg_dry_cows",
        "avg_bred_heifers", "avg_heifers", "avg_calves_4_6_mo", "avg_calves_0_3_mo"
    ]
    def manure_factor_func(d):
        denom = (
            d["avg_milk_cows"] + d["avg_dry_cows"] +
            (d["avg_bred_heifers"] + d["avg_heifers"]) * HEIFER_FACTOR +
            (d["avg_calves_4_6_mo"] + d["avg_calves_0_3_mo"]) * CALF_FACTOR
        )
        result = d["total_manure_excreted_tons"] / denom
        result[denom <= 0] = np.nan
        return result
    df["calculated_manure_factor"] = safe_calc(df, manure_keys, manure_factor_func)

    # Nitrogen deviations
    n_key = "total_manure_gen_n_after_nh3_losses_lbs"
    usda_key = "usda_nitrogen_estimate_lbs"
    ucce_key = "ucce_nitrogen_estimate_lbs"
    
    print(f"\nChecking nitrogen calculation keys:")
    print(f"n_key ({n_key}) in columns: {n_key in df.columns}")
    print(f"usda_key ({usda_key}) in columns: {usda_key in df.columns}")
    print(f"ucce_key ({ucce_key}) in columns: {ucce_key in df.columns}")
    
    if n_key in df.columns:
        reported_n = df[n_key]
        print(f"reported_n non-null values: {reported_n.notna().sum()}")
        print(f"reported_n sample values: {reported_n.dropna().head().tolist() if reported_n.notna().any() else 'None'}")
        
        if usda_key in df.columns:
            print(f"usda_nitrogen_estimate_lbs non-null values: {df[usda_key].notna().sum()}")
            print(f"usda_nitrogen_estimate_lbs sample values: {df[usda_key].dropna().head().tolist() if df[usda_key].notna().any() else 'None'}")
        
        if ucce_key in df.columns:
            print(f"ucce_nitrogen_estimate_lbs non-null values: {df[ucce_key].notna().sum()}")
            print(f"ucce_nitrogen_estimate_lbs sample values: {df[ucce_key].dropna().head().tolist() if df[ucce_key].notna().any() else 'None'}")
        
        df["nitrogen_discrepancy"] = safe_calc(df, [usda_key, n_key], lambda d: d[usda_key] - reported_n)
        df["usda_nitrogen_pct_deviation"] = safe_calc(df, [usda_key, n_key], lambda d: (d[usda_key] - reported_n) / reported_n.replace(0, np.nan) * 100)
        df["ucce_nitrogen_pct_deviation"] = safe_calc(df, [ucce_key, n_key], lambda d: (d[ucce_key] - reported_n) / reported_n.replace(0, np.nan) * 100)
        
        # Add pretty-named columns for app compatibility
        df["USDA Nitrogen % Deviation"] = df["usda_nitrogen_pct_deviation"]
        df["UCCE Nitrogen % Deviation"] = df["ucce_nitrogen_pct_deviation"]
        
        print(f"\nAfter calculation:")
        print(f"usda_nitrogen_pct_deviation non-null values: {df['usda_nitrogen_pct_deviation'].notna().sum()}")
        print(f"ucce_nitrogen_pct_deviation non-null values: {df['ucce_nitrogen_pct_deviation'].notna().sum()}")
        print(f"USDA Nitrogen % Deviation non-null values: {df['USDA Nitrogen % Deviation'].notna().sum()}")
        print(f"UCCE Nitrogen % Deviation non-null values: {df['UCCE Nitrogen % Deviation'].notna().sum()}")
    else:
        print(f"\nWARNING: {n_key} not found in columns, setting nitrogen deviation columns to NaN")
        df["nitrogen_discrepancy"] = np.nan
        df["usda_nitrogen_pct_deviation"] = np.nan
        df["ucce_nitrogen_pct_deviation"] = np.nan
        df["USDA Nitrogen % Deviation"] = np.nan
        df["UCCE Nitrogen % Deviation"] = np.nan

    # Wastewater ratio discrepancy
    df["manure_factor_discrepancy"] = safe_calc(
        df, ["calculated_manure_factor"],
        lambda d: d["calculated_manure_factor"] - BASE_MANURE_FACTOR
    )

    # Fill NA values with 0 for all calculated columns
    calculated_columns = [
        "total_herd_size",
        "Average Milk Production (kg per cow)", "Average Milk Production (L per cow)", "Total Annual Milk Production (L)",
        "total_applied_n_lbs", "total_applied_p_lbs", "total_applied_k_lbs", "total_applied_salt_lbs",
        "total_reported_n_lbs", "total_reported_p_lbs", "total_reported_k_lbs", "total_reported_salt_lbs",
        "unaccounted_for_n_lbs", "unaccounted_for_p_lbs", "unaccounted_for_k_lbs", "unaccounted_for_salt_lbs",
        "total_ww_gen_liters", "ww_to_reported_milk", "ww_to_estimated_milk",
        "calculated_manure_factor", "nitrogen_discrepancy", "wastewater_ratio_discrepancy", "manure_factor_discrepancy",
        "usda_nitrogen_pct_deviation", "ucce_nitrogen_pct_deviation"
    ]
    for col in calculated_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print("\n=== End Debug: calculate_all_metrics ===\n")
    return df

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

def process_csv(csv_path, template_params, columns, param_types):
    """Process a single CSV file and extract all parameters."""
    result = {col: None for col in columns}
    result['filename'] = os.path.basename(csv_path)
    
    try:
        df = pd.read_csv(csv_path)
        
        # Process each parameter
        for _, row in template_params.iterrows():
            param_key = row['parameter_key']
            if pd.isna(row['column_search_text']):
                continue
                
            # Find the column that matches the search text
            col_name = row['column_search_text']
            if col_name in df.columns:
                # For numeric columns, convert to float and handle any formatting
                if param_types.get(param_key) == 'numeric':
                    value = pd.to_numeric(df[col_name].iloc[0], errors='coerce')
                    if pd.isna(value):
                        value = 0
                else:
                    value = df[col_name].iloc[0]
                result[param_key] = value
                
    except Exception as e:
        print(f"Error processing CSV {csv_path}: {str(e)}")
        
    return result

def main(test_mode=False):
    """Main function to process all PDF files and extract data."""
    if read_reports:
        dtype_dict = {col: str for col in ['region', 'template', 'parameter_key', 'page_search_text', 
                                          'search_direction', 'row_search_text', 'column_search_text',
                                          'ignore_before', 'value_pattern']}
        dtype_dict['item_order'] = 'Int64'  # Using Int64 to handle NA values
        parameter_locations = pd.read_csv('ca_cafo_compliance/parameter_locations.csv', dtype=dtype_dict)

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
                        
                        template_params = parameter_locations[parameter_locations['template'] == template]
                        
                        # Check if this is a CSV template
                        if template == 'r8_csv':
                            # For R8, the CSV files are in data/2023/R8/all_r8/r8_csv/
                            if region == 'R8':
                                # Process both animals and manure CSVs
                                animals_path = os.path.join(base_data_path, 'R8', 'all_r8', 'r8_csv', 'R8_animals.csv')
                                manure_path = os.path.join(base_data_path, 'R8', 'all_r8', 'r8_csv', 'R8_manure.csv')
                                
                                animals_df = pd.read_csv(animals_path)
                                manure_df = pd.read_csv(manure_path)
                                
                                df = pd.merge(animals_df, manure_df, on='Facility Name', how='outer', suffixes=('', '_manure'))
                                
                                results = []
                                for _, row in df.iterrows():
                                    result = {col: None for col in template_params['parameter_key'].unique()}
                                    result['filename'] = 'R8_animals.csv'  # Use animals CSV as the filename
                                    
                                    # Map CSV columns to parameters
                                    for _, param_row in template_params.iterrows():
                                        param_key = param_row['parameter_key']
                                        col_name = param_row['column_search_text']
                                        if col_name in df.columns:
                                            if param_types.get(param_key) == 'numeric':
                                                value = pd.to_numeric(row[col_name], errors='coerce')
                                                if pd.isna(value):
                                                    value = 0
                                            else:
                                                value = row[col_name]
                                            result[param_key] = value
                                        elif param_row['row_search_text'] in df.columns:
                                            # Try using row_search_text as column name
                                            col_name = param_row['row_search_text']
                                            if param_types.get(param_key) == 'numeric':
                                                value = pd.to_numeric(row[col_name], errors='coerce')
                                                if pd.isna(value):
                                                    value = 0
                                            else:
                                                value = row[col_name]
                                            result[param_key] = value
                                    
                                    results.append(result)
                                df = pd.DataFrame(results)
                        else:
                            # Original PDF processing logic
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
                                
                            columns = template_params['parameter_key'].unique().tolist()
                            
                            with mp.Pool(num_cores) as pool:
                                process_pdf_partial = partial(process_pdf, template_params=template_params, 
                                                           columns=columns, param_types=param_types)
                                results = pool.map(process_pdf_partial, pdf_files)
                            
                            df = pd.DataFrame([r for r in results if r is not None])
                        
                        for col in df.columns:
                            if param_types.get(col) == 'numeric':
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Only now, after all calculations, rename columns to pretty names and fill missing
                        key_to_name = dict(zip(parameters['parameter_key'], parameters['parameter_name']))
                        name_to_key = dict(zip(parameters['parameter_name'], parameters['parameter_key']))
                        
                        # Rename columns to pretty names
                        df = df.rename(columns=key_to_name)
                        
                        # Ensure all pretty names are present, fill missing with np.nan
                        for pretty_name in key_to_name.values():
                            if pretty_name not in df.columns:
                                df[pretty_name] = np.nan
                        
                        # Calculate estimates for each row (still using parameter_keys)
                        estimates = []
                        for _, row in df.rename(columns={v: k for k, v in key_to_name.items()}).iterrows():
                            milk_dry_cows = (row.get('avg_milk_cows', 0) or 0) + (row.get('avg_dry_cows', 0) or 0)
                            heifers = (row.get('avg_bred_heifers', 0) or 0) + (row.get('avg_heifers', 0) or 0)
                            calves = (row.get('avg_calves_4_6_mo', 0) or 0) + (row.get('avg_calves_0_3_mo', 0) or 0)
                            estimated_manure = (milk_dry_cows * BASE_MANURE_FACTOR) + \
                                             (heifers * HEIFER_FACTOR * BASE_MANURE_FACTOR) + \
                                             (calves * CALF_FACTOR * BASE_MANURE_FACTOR)
                            usda_nitrogen = estimated_manure * MANURE_N_CONTENT
                            animal_units = milk_dry_cows + (heifers * HEIFER_FACTOR) + (calves * CALF_FACTOR)
                            ucce_nitrogen = animal_units * DAYS_PER_YEAR
                            wastewater_ratio = 0
                            if all(x in row for x in ['avg_milk_lb_per_cow_day', 'total_ww_gen_gals']):
                                milk_production = row['avg_milk_lb_per_cow_day']
                                wastewater = row['total_ww_gen_gals']
                                daily_milk_liters = milk_production * LBS_TO_LITERS * milk_dry_cows
                                annual_milk_liters = daily_milk_liters * DAYS_PER_YEAR
                                if annual_milk_liters > 0:
                                    wastewater_ratio = wastewater / annual_milk_liters
                            estimates.append({
                                'Estimated Total Manure (tons)': estimated_manure,
                                'USDA Nitrogen Estimate (lbs)': usda_nitrogen,
                                'UCCE Nitrogen Estimate (lbs)': ucce_nitrogen,
                                'Wastewater to Milk Ratio': wastewater_ratio
                            })
                        estimates_df = pd.DataFrame(estimates)
                        df = pd.concat([df, estimates_df], axis=1)
                        
                        metrics_df = calculate_all_metrics(df.rename(columns={v: k for k, v in key_to_name.items()}))
                        for col in metrics_df.columns:
                            if col not in df.columns or metrics_df[col].notna().any():
                                df[col] = metrics_df[col]
                        os.makedirs(output_folder, exist_ok=True)
                        # Delete any existing CSVs in the output folder
                        for f in os.listdir(output_folder):
                            if f.endswith('.csv'):
                                os.remove(os.path.join(output_folder, f))
                        df.to_csv(os.path.join(output_folder, f"{name}.csv"), index=False)
    if consolidate_data:
        consolidate_outputs()

def consolidate_outputs():
    """Consolidate all output CSVs into master files and perform geocoding and consultant metrics."""
    GEOCODING_CACHE_FILE = "outputs/geocoding_cache.json"
    R2_COUNTIES = ["Alameda", "Contra Costa", "Marin", "Napa",  "San Francisco", 
                   "San Mateo", "Santa Clara", "Solano", "Sonoma"]
    try:
        with open(GEOCODING_CACHE_FILE, 'r') as f:
            geocoding_cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        geocoding_cache = {}
    # Add R8 data to geocoding cache if not already present
    r8_data_path = "data/2023/R8/all_r8/r8_csv/R8_animals.csv"
    if os.path.exists(r8_data_path):
        print("Adding R8 facility data to geocoding cache...")
        r8_df = pd.read_csv(r8_data_path)
        for _, row in r8_df.iterrows():
            if pd.notna(row['Facility Address']) and pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                address = row['Facility Address']
                if address not in geocoding_cache:
                    geocoding_cache[address] = {
                        'lat': float(row['Latitude']),
                        'lng': float(row['Longitude']),
                        'timestamp': datetime.now().isoformat(),
                        'successful_format': 'R8 direct data',
                        'geocoder': 'R8 CSV',
                        'address': address,
                        'county': row.get('County')  # If county is available in R8 data
                    }
        save_geocoding_cache(geocoding_cache)
        print(f"Added {len(r8_df)} R8 facilities to geocoding cache")
    for year in YEARS:
        base_path = f"outputs/{year}"
        if not os.path.exists(base_path):
            continue
        for region in REGIONS:
            region_path = os.path.join(base_path, region)
            if not os.path.exists(region_path):
                continue
            # Collect and process CSV files
            csv_files = glob.glob(os.path.join(region_path, "**/*.csv"), recursive=True)
            if not csv_files:
                continue
            dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                # Ensure these columns always exist
                for col in ["USDA Nitrogen % Deviation", "UCCE Nitrogen % Deviation", "ww_to_reported_milk", "ww_to_estimated_milk"]:
                    if col not in df.columns:
                        df[col] = np.nan
                # Add metadata columns
                df['Year'] = year
                df['Region'] = region
                df['filename'] = os.path.basename(csv_file)
                # Extract template from path
                path_parts = csv_file.split(os.sep)
                region_idx = path_parts.index(region)
                if region_idx + 2 < len(path_parts):
                    df['Template'] = path_parts[region_idx + 2]
                # Ensure Year is int or str, not float
                if 'Year' in df.columns:
                    df['Year'] = df['Year'].apply(lambda x: str(int(float(x))) if pd.notna(x) else x)
                dfs.append(df)
            if not dfs:
                continue
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.dropna(how='all')
            # Filter out empty rows
            metadata_cols = ['Year', 'Region', 'Template', 'filename']
            data_cols = [col for col in combined_df.columns if col not in metadata_cols]
            combined_df = combined_df[~combined_df.apply(
                lambda row: all(pd.isna(val) or val == 0 for val in row[data_cols]), 
                axis=1
            )]
            combined_df['Consultant'] = combined_df['Template'].map(consultant_mapping).fillna('Unknown')
            herd_cols = [
                'Average Milk Cows', 'Average Dry Cows', 'Average Bred Heifers',
                'Average Heifers', 'Average Calves (4-6 mo.)', 'Average Calves (0-3 mo.)',
            ]
            def calc_herd_size(row):
                vals = [row.get(col, 0) for col in herd_cols if col in row]
                vals = [v for v in vals if pd.notna(v)]
                return sum(vals) if vals else 0
            combined_df['Total Herd Size'] = combined_df.apply(calc_herd_size, axis=1)
            print("Geocoding addresses...")
            address_col = next((col for col in ['Dairy Address', 'Facility Address'] 
                                if col in combined_df.columns), None)
            if address_col:
                combined_df['Latitude'] = None
                combined_df['Longitude'] = None
                combined_df['Street Address'] = None
                combined_df['City'] = None
                combined_df['County'] = None
                combined_df['Zip'] = None
                unique_addresses = combined_df[address_col].dropna().unique()
                new_geocodes = 0
                for address in unique_addresses:
                    # Get the county for this address
                    county = combined_df[combined_df[address_col] == address]['County'].iloc[0] if not combined_df[combined_df[address_col] == address].empty else None
                    lat, lng, geocoded_county = geocode_address(address, geocoding_cache, county=county, try_again=False)
                    if lat is not None:
                        new_geocodes += 1
                    # Parse address components
                    street, city, parsed_county, zip_code = parse_address(address)
                    final_county = geocoded_county or parsed_county
                    mask = combined_df[address_col] == address
                    combined_df.loc[mask, 'Latitude'] = lat
                    combined_df.loc[mask, 'Longitude'] = lng
                    combined_df.loc[mask, 'Street Address'] = street
                    combined_df.loc[mask, 'City'] = city
                    combined_df.loc[mask, 'County'] = final_county
                    combined_df.loc[mask, 'Zip'] = zip_code
                print(f"Geocoding complete: {new_geocodes} addresses geocoded")
            # consultant metrics only for R5 and 2023
            if year == 2023 and region == "R5":
                consultant_metrics = calculate_consultant_metrics(combined_df)
                metrics_file = f"outputs/consolidated/{year}_{region}_consultant_metrics.csv"
                consultant_metrics.to_csv(metrics_file, index=False)
                print(f"Saved consultant metrics to {metrics_file}")
            output_file = f"outputs/consolidated/{year}_{region}_master.csv"
            combined_df.to_csv(output_file, index=False)
            print(f"Saved consolidated data to {output_file}")
            print(f"Total records: {len(combined_df)}")

def save_geocoding_cache(cache):
    """Save geocoded addresses to cache file."""
    os.makedirs(os.path.dirname(GEOCODING_CACHE_FILE), exist_ok=True)
    with open(GEOCODING_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def normalize_address(address):
    if pd.isna(address) or not isinstance(address, str):
        return None
    address = address.replace(": ", "")
    address = address.lower()
    address = re.sub(r'[.,]', '', address)
    replacements = {
        'avenue': 'ave',
        'street': 'st',
        'road': 'rd',
        'boulevard': 'blvd',
        'highway': 'hwy'
    }
    for old, new in replacements.items():
        address = address.replace(old, new)
    address = re.sub(r'\b(ca|california)\b', '', address)
    address = re.sub(r'\b(inc|llc)\b', '', address) 
    return ' '.join(address.split())

def find_cached_address(address, cache):
    if pd.isna(address) or not isinstance(address, str):
        return None
    if address in cache:
        return address
    normalized = normalize_address(address)
    if not normalized:
        return None
        
    # Search through cache keys
    for cached_addr in cache.keys():
        if normalize_address(cached_addr) == normalized:
            return cached_addr
    return None

def geocode_address(address, cache, county=None, try_again=False):
    if pd.isna(address) or not isinstance(address, str):
        return None, None, None
    clean_address = address.replace(": ", "")
    cached_addr = find_cached_address(clean_address, cache)
    if cached_addr:
        cached_result = cache[cached_addr]
        if try_again and (cached_result['lat'] is None or cached_result['lng'] is None):
            print(f"Retrying previously failed address: {clean_address}")
        else:
            return cached_result['lat'], cached_result['lng'], cached_result.get('county')
    address_formats = []
    parts = clean_address.split()
    if len(parts) >= 3:
        street_number = parts[0]
        street_name = ' '.join(parts[1:-2])
        city = parts[-2]
        state_zip = parts[-1]
        formatted_address = f"{street_number} {street_name}, {city}, CA {state_zip}"
        address_formats = [
            formatted_address,
            clean_address,
            f"{clean_address}, California"
        ]
        if county == "all_r2":
            for r2_county in ["Alameda", "Contra Costa", "Marin", "Napa",  "San Francisco", "San Mateo", "Santa Clara", "Solano", "Sonoma"]:
                address_formats.append(f"{formatted_address}, {r2_county} County, CA")
        elif county and county not in ["all_r2", "all_r7"]:
            address_formats.append(f"{formatted_address}, {county} County, CA")
    else:
        address_formats = [
            clean_address,
            f"{clean_address}, California"
        ]
        if county == "all_r2":
            for r2_county in ["Alameda", "Contra Costa", "Marin", "Napa",  "San Francisco", "San Mateo", "Santa Clara", "Solano", "Sonoma"]:
                address_formats.append(f"{clean_address}, {r2_county} County, CA")
        elif county and county not in ["all_r2", "all_r7"]:
            address_formats.append(f"{clean_address}, {county} County, CA")
    geocoders = [
        ArcGIS(user_agent="ca_cafo_compliance"),
        Nominatim(user_agent="ca_cafo_compliance", timeout=30)
    ]
    successful_locations = []
    max_retries = 2
    retry_delay = 3
    for geolocator in geocoders:
        for addr_format in address_formats:
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        time.sleep(retry_delay * (2 ** attempt))
                    else:
                        time.sleep(1)
                    location = geolocator.geocode(addr_format)
                    if location:
                        if location.address and ('California' in location.address or 'CA' in location.address):
                            loc_key = f"{location.latitude:.6f},{location.longitude:.6f}"
                            if not any(f"{loc['lat']:.6f},{loc['lng']:.6f}" == loc_key for loc in successful_locations):
                                county_val = None
                                if isinstance(geolocator, ArcGIS):
                                    try:
                                        reverse = geolocator.reverse(f"{location.latitude}, {location.longitude}")
                                        if reverse and reverse.raw:
                                            address_components = reverse.raw.get('address', {})
                                            county_val = address_components.get('County')
                                    except Exception as e:
                                        print(f"Error getting county from ArcGIS: {e}")
                                successful_locations.append({
                                    'lat': location.latitude,
                                    'lng': location.longitude,
                                    'format': addr_format,
                                    'geocoder': geolocator.__class__.__name__,
                                    'address': location.address,
                                    'county': county_val
                                })
                            if successful_locations:
                                break
                except (GeocoderTimedOut, GeocoderServiceError) as e:
                    print(f"Geocoding error for address format '{addr_format}' (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        print(f"Failed to geocode after {max_retries} attempts")
                    continue
                except Exception as e:
                    print(f"Unexpected error for address format '{addr_format}': {e}")
                    break
            if successful_locations:
                break
        if successful_locations:
            break
    if successful_locations:
        if len(successful_locations) > 1:
            print(f"Multiple unique locations found for address: {clean_address}")
            for loc in successful_locations:
                print(f"  - {loc['address']} ({loc['lat']}, {loc['lng']})")
        best_location = successful_locations[0]
        cache[clean_address] = {
            'lat': best_location['lat'],
            'lng': best_location['lng'],
            'timestamp': datetime.now().isoformat(),
            'successful_format': best_location['format'],
            'geocoder': best_location['geocoder'],
            'address': best_location['address'],
            'county': best_location['county']
        }
        save_geocoding_cache(cache)
        print(f"Successfully geocoded address using format: {best_location['format']} from {best_location['geocoder']}")
        return best_location['lat'], best_location['lng'], best_location['county']
    try:
        search_url = f"https://www.google.com/maps/search/{clean_address.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(search_url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                if tag.get('property') == 'og:latitude':
                    lat = float(tag.get('content'))
                    lng = float(soup.find('meta', property='og:longitude').get('content'))
                    cache[clean_address] = {
                        'lat': lat,
                        'lng': lng,
                        'timestamp': datetime.now().isoformat(),
                        'successful_format': 'Google Maps fallback',
                        'geocoder': 'Google Maps'
                    }
                    save_geocoding_cache(cache)
                    print(f"Successfully geocoded address using Google Maps fallback")
                    return lat, lng, None
    except Exception as e:
        print(f"Google Maps fallback failed: {e}")
    cache[clean_address] = {
        'lat': None,
        'lng': None,
        'error': "All address formats failed",
        'timestamp': datetime.now().isoformat()
    }
    save_geocoding_cache(cache)
    return None, None, None

def parse_address(address):
    if pd.isna(address) or not isinstance(address, str):
        return None, None, None, None
    address = address.replace(": ", "")
    parts = address.split()
    if len(parts) < 3:
        return None, None, None, None
    street_number = parts[0]
    street_name = ' '.join(parts[1:-2])
    city = parts[-2]
    state_zip = parts[-1]
    zip_code = state_zip[-5:] if len(state_zip) >= 5 else None
    county = None
    address_lower = address.lower()
    if 'fresno' in address_lower or 'madera' in address_lower:
        county = 'Fresno/Madera'
    elif 'kern' in address_lower:
        county = 'Kern'
    elif 'kings' in address_lower:
        county = 'Kings'
    elif 'tulare' in address_lower:
        county = 'Tulare'
    elif 'sonoma' in address_lower:
        county = 'Sonoma'
    elif 'marin' in address_lower:
        county = 'Marin'
    elif 'napa' in address_lower:
        county = 'Napa'
    elif 'solano' in address_lower:
        county = 'Solano'
    elif 'contra costa' in address_lower:
        county = 'Contra Costa'
    elif 'alameda' in address_lower:
        county = 'Alameda'
    elif 'san francisco' in address_lower:
        county = 'San Francisco'
    elif 'san mateo' in address_lower:
        county = 'San Mateo'
    elif 'santa clara' in address_lower:
        county = 'Santa Clara'
    return street_number + ' ' + street_name, city, county, zip_code

def calculate_metrics(df):
    """Calculate metrics for each facility."""
    # No need to recalculate metrics that are already in the data
    # Just ensure the columns exist
    required_columns = [
        'ratio_ww_to_milk_l_per_l',
        'usda_nitrogen_pct_deviation',
        'ucce_nitrogen_pct_deviation',
        'calculated_manure_factor'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Required column {col} not found in data")
            df[col] = None
    
    return df

def calculate_consultant_metrics(df):
    """Calculate average under/over-reporting metrics for each consultant."""
    # Group by consultant
    consultant_groups = df.groupby('Consultant')
    
    metrics = []
    for consultant, group in consultant_groups:
        manure_avg = group['Calculated Manure Factor'].mean() if 'Calculated Manure Factor' in group.columns else None
        manure_std = group['Calculated Manure Factor'].std() if 'Calculated Manure Factor' in group.columns else None
        
        wastewater_avg = group['Wastewater to Milk Ratio'].mean() if 'Wastewater to Milk Ratio' in group.columns else None
        wastewater_std = group['Wastewater to Milk Ratio'].std() if 'Wastewater to Milk Ratio' in group.columns else None
        
        nitrogen_usda_avg = group['USDA Nitrogen % Deviation'].mean() if 'USDA Nitrogen % Deviation' in group.columns else None
        nitrogen_usda_std = group['USDA Nitrogen % Deviation'].std() if 'USDA Nitrogen % Deviation' in group.columns else None
        
        nitrogen_ucce_avg = group['UCCE Nitrogen % Deviation'].mean() if 'UCCE Nitrogen % Deviation' in group.columns else None
        nitrogen_ucce_std = group['UCCE Nitrogen % Deviation'].std() if 'UCCE Nitrogen % Deviation' in group.columns else None
        
        metrics.append({
            'Consultant': consultant,
            'Manure Factor Avg': manure_avg,
            'Manure Factor Std': manure_std,
            'Wastewater Ratio Avg': wastewater_avg,
            'Wastewater Ratio Std': wastewater_std,
            'USDA Nitrogen % Dev Avg': nitrogen_usda_avg,
            'USDA Nitrogen % Dev Std': nitrogen_usda_std,
            'UCCE Nitrogen % Dev Avg': nitrogen_ucce_avg,
            'UCCE Nitrogen % Dev Std': nitrogen_ucce_std,
            'Facility Count': len(group)
        })
    
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    main(test_mode=False)