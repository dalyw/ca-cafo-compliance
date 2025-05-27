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
from geopy.geocoders import ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Initialize geolocator at module level
geolocator = ArcGIS(user_agent="ca_cafo_compliance")

read_reports = True
consolidate_data = True

def load_parameters():
    """Load parameters and create mapping dictionaries."""
    parameters = pd.read_csv('ca_cafo_compliance/parameters.csv')
    calculated_metrics = pd.read_csv('ca_cafo_compliance/calculated_metrics.csv')
    return {
        'snake_to_pretty': dict(zip(parameters['parameter_key'], parameters['parameter_name'])),
        'pretty_to_snake': dict(zip(parameters['parameter_name'], parameters['parameter_key'])),
        'data_types': dict(zip(parameters['parameter_key'], parameters['data_type'])),
        'calculated_metrics': dict(zip(calculated_metrics['metric_key'], calculated_metrics['metric_name']))
    }

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

def extract_value_from_line(line, item_order=None, ignore_before=None, ignore_after=None, param_key=None):
    """Extract value from a line using item_order, ignore_before, and ignore_after. If none, return full line."""
    if not isinstance(line, str):
        line = str(line)
    original_line = line  # For debugging
    if item_order is None and not ignore_before and not ignore_after:
        return line
    # Optionally trim before/after
    if ignore_after:
        if ignore_after == 'str':
            # If ignore_after is 'str', trim at first non-numeric character after a numeric sequence
            match = re.match(r'([-+]?\d*\.?\d+)', line.strip())
            if match:
                line = match.group(1)
        else:
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
        # if param_key == 'avg_dry_cows':
            # print(f"[DEBUG extract_value_from_line] param_key: {param_key} | original_line: '{original_line}' | after trim: '{line}' | parts: {parts} | item_order: {item_order}")
            # if 0 <= idx < len(parts):
            #     print(f"[DEBUG extract_value_from_line] Returning value for avg_dry_cows: {parts[idx]}")
            # else:
            #     print(f"[DEBUG extract_value_from_line] item_order {item_order} out of range for line: '{line}' (parts: {parts})")
        if 0 <= idx < len(parts):
            return parts[idx]
        return ''
    return line

def extract_text_adjacent_to_phrase(text, phrase, direction='right', row_search_text=None, column_search_text=None, item_order=None, ignore_before=None, ignore_after=None, param_key=None):
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
            return extract_value_from_line(text_after, item_order, ignore_before, ignore_after, param_key=param_key)
    elif direction == 'below':
        # Find the next non-blank line after the phrase
        next_line = None
        for j in range(phrase_line_idx + 1, len(lines)):
            if lines[j].strip():
                next_line = lines[j].strip()
                break
        if next_line is not None:
            return extract_value_from_line(next_line, item_order, ignore_before, ignore_after, param_key=param_key)
    elif direction == 'table':
        if row_search_text and column_search_text:
            row_idx = next((i for i, line in enumerate(lines) if isinstance(line, str) and row_search_text.lower() in line.lower()), None)
            if row_idx is not None:
                header_parts = [part.strip() for part in str(lines[row_idx]).split() if part.strip()]
                col_idx = next((i for i, part in enumerate(header_parts) if column_search_text.lower() in part.lower()), None)
                if col_idx is not None and row_idx + 1 < len(lines):
                    value_parts = [part.strip() for part in str(lines[row_idx + 1]).split() if part.strip()]
                    if col_idx < len(value_parts):
                        return extract_value_from_line(value_parts[col_idx], item_order, ignore_before, ignore_after, param_key=param_key)
    elif direction == 'above':
        if phrase_line_idx > 0:
            value_line = lines[phrase_line_idx - 1]
            return extract_value_from_line(value_line, item_order, ignore_before, ignore_after, param_key=param_key)
    return None

def find_value_by_text(page_text, row, data_type, param_key=None):
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
        ignore_after=row['ignore_after'] if 'ignore_after' in row else None,
        param_key=param_key
    )
    if extracted_text:
        # If it's a single value, just return it
        if isinstance(extracted_text, str) and len(extracted_text.split()) == 1:
            return convert_to_numeric(extracted_text, data_type)
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
            with open(text_file, 'r') as f:
                text = f.read()
                # Clean up text
                text = text.replace("Maxiumu", "Maximum")
                text = text.replace("|", "")
                text = text.replace(",", "")
                text = text.replace("=", "")
                text = text.replace(":", "")
                text = text.replace("Ibs", "lbs")
                text = text.replace("/bs", "lbs")
                text = text.replace("FaciIity", "Facility")
                text = text.replace("CattIe", "Cattle")
                text = text.replace("  ", " ")
                text = text.replace("___", "")
                text = '\n'.join([line for line in text.split('\n') if line.strip()])
                return text
    
    print(f"OCR text file not found for {pdf_name}")
    return None

def find_parameter_value(ocr_text, row, data_types):
    """Extract a parameter value from OCR text based on the specified row from parameter_locations."""
    if pd.isna(row['search_direction']):
        return np.nan
    data_type = data_types.get(row['parameter_key'], 'text')
    param_key = row['parameter_key']
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
        value = find_value_by_text(page_text=search_text, row=row, data_type=data_type, param_key=param_key)
        return value
    except Exception as e:
        print(f"Error processing parameter {row['parameter_key']}: {str(e)}")
        return np.nan if data_type == 'text' else 0

def process_pdf(pdf_path, template_params, columns, data_types):
    """Process a single PDF file and extract all parameters from OCR text."""
    result = {col: None for col in columns}
    result['filename'] = os.path.basename(pdf_path)
    ocr_text = load_ocr_text(pdf_path)
    if not ocr_text:
        return result
    # Process main report parameters
    for _, row in template_params.iterrows():
        param_key = row['parameter_key']
        value = find_parameter_value(ocr_text, row, data_types)
        # Debug print for avg_dry_cows assignment
        # if param_key == 'avg_dry_cows':
        #     print(f"[DEBUG assign] param_key: {param_key} | value: {value}")
        result[param_key] = value
    # Print the result dict before appending
    # if 'R5' in pdf_path:
    #     print(f"[DEBUG result dict] {result}")
    return result

def process_csv(csv_path, template_params, columns, data_types):
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
                if data_types.get(param_key) == 'numeric':
                    value = pd.to_numeric(df[col_name].iloc[0], errors='coerce')
                    if pd.isna(value):
                        value = 0
                else:
                    value = df[col_name].iloc[0]
                result[param_key] = value
                
    except Exception as e:
        print(f"Error processing CSV {csv_path}: {str(e)}")
        
    return result

def safe_calc(df, keys, func, default=np.nan):
    if all(k in df.columns for k in keys):
        return func(df)
    return default

def calculate_metrics(df):
    """Calculate all metrics for the dataframe."""
    # Load calculated metrics mapping
    params = load_parameters()
    calculated_metrics = params['calculated_metrics']

    # Calculate annual milk production
    df[calculated_metrics['avg_milk_prod_kg_per_cow']] = safe_calc(
        df, ['avg_milk_lb_per_cow_day'],
        lambda d: d['avg_milk_lb_per_cow_day'] * LBS_TO_KG
    )
    df[calculated_metrics['avg_milk_prod_l_per_cow']] = safe_calc(
        df, ['avg_milk_lb_per_cow_day'],
        lambda d: d['avg_milk_lb_per_cow_day'] * LBS_TO_KG * KG_PER_L_MILK
    )
    df[calculated_metrics['reported_annual_milk_production_l']] = safe_calc(
        df, ['avg_milk_lb_per_cow_day', 'avg_milk_cows', 'avg_dry_cows'],
        lambda d: d['avg_milk_lb_per_cow_day'] * LBS_TO_KG * KG_PER_L_MILK * (d['avg_milk_cows'].fillna(0) + d['avg_dry_cows'].fillna(0)) * 365
    )

    # Calculate herd size
    herd_keys = [
        "avg_milk_cows", "avg_dry_cows", "avg_bred_heifers",
        "avg_heifers", "avg_calves_4_6_mo", "avg_calves_0_3_mo", "avg_other"
    ]
    df[calculated_metrics['total_herd_size']] = safe_calc(
        df, herd_keys,
        lambda d: sum(d[k].fillna(0) for k in herd_keys if k in d.columns),
        default=0
    )

    # Calculate nutrient metrics
    nutrient_types = ["n", "p", "k", "salt"]
    for nutrient in nutrient_types:
        # Total Applied
        dry_key = f"applied_{nutrient}_dry_manure_lbs"
        ww_key = f"applied_ww_{nutrient}_lbs"
        total_applied_key = calculated_metrics[f'total_applied_{nutrient}_lbs']
        df[total_applied_key] = safe_calc(
            df, [dry_key, ww_key],
            lambda d: d[dry_key].fillna(0) + d[ww_key].fillna(0)
        )

        if nutrient == "n":
            dry_key_reported = "total_manure_gen_n_after_nh3_losses_lbs"
        else:
            dry_key_reported = f"total_manure_gen_{nutrient}_lbs"
        ww_key_reported = f"total_ww_gen_{nutrient}_lbs"
        total_reported_key = calculated_metrics[f'total_reported_{nutrient}_lbs']
        df[total_reported_key] = safe_calc(
            df, [dry_key_reported, ww_key_reported],
            lambda d: d[dry_key_reported].fillna(0) + d[ww_key_reported].fillna(0)
        )

        # Unaccounted for
        exports_key = f"total_exports_{nutrient}_lbs"
        unaccounted_key = calculated_metrics[f'unaccounted_for_{nutrient}_lbs']
        df[unaccounted_key] = safe_calc(
            df, [dry_key_reported, ww_key_reported, total_applied_key, exports_key],
            lambda d: d[dry_key_reported].fillna(0) + d[ww_key_reported].fillna(0) - d[total_applied_key].fillna(0) - d[exports_key].fillna(0)
        )

    # Calculate wastewater metrics
    total_ww_gen_liters = calculated_metrics['total_ww_gen_liters']
    df[total_ww_gen_liters] = safe_calc(
        df, ["total_ww_gen_gals"],
        lambda d: d["total_ww_gen_gals"] * 3.78541
    )

    wastewater_to_reported = calculated_metrics['wastewater_to_reported']
    reported_annual_milk = calculated_metrics['reported_annual_milk_production_l']
    df[wastewater_to_reported] = safe_calc(
        df, [total_ww_gen_liters, reported_annual_milk],
        lambda d: d[total_ww_gen_liters] / d[reported_annual_milk].replace(0, np.nan)
    )

    # Calculate estimated annual milk production (L)
    est_milk_col = 'estimated_annual_milk_production_l'
    if 'avg_milk_lb_per_cow_day' in df.columns and df['avg_milk_lb_per_cow_day'].notna().any():
        df[est_milk_col] = safe_calc(
            df, ['avg_milk_lb_per_cow_day', 'avg_milk_cows', 'avg_dry_cows'],
            lambda d: d['avg_milk_lb_per_cow_day'] * LBS_TO_KG * KG_PER_L_MILK * (d['avg_milk_cows'].fillna(0) + d['avg_dry_cows'].fillna(0)) * 365
        )
    else:
        df[est_milk_col] = safe_calc(
            df, ['avg_milk_cows', 'avg_dry_cows'],
            lambda d: DEFAULT_MILK_PRODUCTION * (d['avg_milk_cows'].fillna(0) + d['avg_dry_cows'].fillna(0)) * 365 * LBS_TO_KG * KG_PER_L_MILK
        )

    # Calculate estimated wastewater generation (L)
    wastewater_estimated_col = 'wastewater_estimated'
    df[wastewater_estimated_col] = df[est_milk_col] * L_WW_PER_L_MILK_LOW

    # Calculate Wastewater to Estimated Milk Ratio
    wastewater_to_estimated = calculated_metrics['wastewater_to_estimated']
    df[wastewater_to_estimated] = safe_calc(
        df, [total_ww_gen_liters, est_milk_col],
        lambda d: d[total_ww_gen_liters] / d[est_milk_col].replace(0, np.nan)
    )

    # Calculate Wastewater Ratio Discrepancy
    wastewater_ratio_discrepancy = calculated_metrics['wastewater_ratio_discrepancy']
    df[wastewater_ratio_discrepancy] = safe_calc(
        df, [wastewater_to_estimated, wastewater_to_reported],
        lambda d: d[wastewater_to_estimated] - d[wastewater_to_reported]
    )

    # Calculate manure metrics
    manure_keys = [
        "total_manure_excreted_tons", "avg_milk_cows", "avg_dry_cows",
        "avg_bred_heifers", "avg_heifers", "avg_calves_4_6_mo", "avg_calves_0_3_mo"
    ]
    calculated_manure_factor = calculated_metrics['calculated_manure_factor']
    def manure_factor_func(d):
        denom = (
            d["avg_milk_cows"] + d["avg_dry_cows"] +
            (d["avg_bred_heifers"] + d["avg_heifers"]) * HEIFER_FACTOR +
            (d["avg_calves_4_6_mo"] + d["avg_calves_0_3_mo"]) * CALF_FACTOR
        )
        result = d["total_manure_excreted_tons"] / denom
        result[denom <= 0] = np.nan
        return result
    df[calculated_manure_factor] = safe_calc(df, manure_keys, manure_factor_func)
    manure_factor_discrepancy = calculated_metrics['manure_factor_discrepancy']
    df[manure_factor_discrepancy] = safe_calc(
        df, [calculated_manure_factor],
        lambda d: d[calculated_manure_factor] - BASE_MANURE_FACTOR
    )

    # Calculate nitrogen metrics
    n_key = "total_manure_gen_n_after_nh3_losses_lbs"
    usda_key = calculated_metrics['usda_nitrogen_estimate_lbs']
    ucce_key = calculated_metrics['ucce_nitrogen_estimate_lbs']
    # Calculate USDA and UCCE nitrogen estimates
    herd_keys = [
        "avg_milk_cows", "avg_dry_cows",
        "avg_bred_heifers", "avg_heifers",
        "avg_calves_4_6_mo", "avg_calves_0_3_mo"
    ]
    df[usda_key] = safe_calc(
        df, ["total_manure_excreted_tons"],
        lambda d: d["total_manure_excreted_tons"] * MANURE_N_CONTENT * 2000  # tons to lbs
    )
    df[ucce_key] = safe_calc(
        df, herd_keys,
        lambda d: (
            d["avg_milk_cows"].fillna(0) * BASE_MANURE_FACTOR +
            d["avg_dry_cows"].fillna(0) * BASE_MANURE_FACTOR +
            (d["avg_bred_heifers"].fillna(0) + d["avg_heifers"].fillna(0)) * HEIFER_FACTOR * BASE_MANURE_FACTOR +
            (d["avg_calves_4_6_mo"].fillna(0) + d["avg_calves_0_3_mo"].fillna(0)) * CALF_FACTOR * BASE_MANURE_FACTOR
        ) * MANURE_N_CONTENT
    )
    if n_key in df.columns:
        reported_n = df[n_key]
        nitrogen_discrepancy = calculated_metrics['nitrogen_discrepancy']
        usda_pct_dev = calculated_metrics['usda_nitrogen_pct_deviation']
        ucce_pct_dev = calculated_metrics['ucce_nitrogen_pct_deviation']
        df[nitrogen_discrepancy] = safe_calc(df, [usda_key, n_key], lambda d: reported_n - d[usda_key])
        df[usda_pct_dev] = safe_calc(df, [usda_key, n_key], lambda d: (reported_n - d[usda_key]) / d[usda_key].replace(0, np.nan) * 100)
        df[ucce_pct_dev] = safe_calc(df, [ucce_key, n_key], lambda d: (reported_n - d[ucce_key]) / d[ucce_key].replace(0, np.nan) * 100)
    else:
        for col in ['nitrogen_discrepancy', 'usda_nitrogen_pct_deviation', 'ucce_nitrogen_pct_deviation']:
            df[calculated_metrics[col]] = np.nan

    # Fill NA values with 0 for all calculated columns
    for col in calculated_metrics.values():
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df

def main(test_mode=False):
    """Main function to process all PDF files and extract data."""
    if read_reports:
        # Load parameters and create mappings
        params = load_parameters()
        dtype_dict = {col: str for col in ['region', 'template', 'parameter_key', 'page_search_text', 
                                          'search_direction', 'row_search_text', 'column_search_text',
                                          'ignore_before', 'value_pattern']}
        dtype_dict['item_order'] = 'Int64'
        parameter_locations = pd.read_csv('ca_cafo_compliance/parameter_locations.csv', dtype=dtype_dict)
        
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
                # print(f"\n[DEBUG] Processing region: {region} ({year})")
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
                        columns = template_params['parameter_key'].unique().tolist()
                        # Process files based on template type
                        if template == 'r8_csv' and region == 'R8':
                            # Process R8 CSV files
                            animals_path = os.path.join(base_data_path, 'R8', 'all_r8', 'r8_csv', 'R8_animals.csv')
                            manure_path = os.path.join(base_data_path, 'R8', 'all_r8', 'r8_csv', 'R8_manure.csv')
                            animals_df = pd.read_csv(animals_path)
                            manure_df = pd.read_csv(manure_path)
                            # print(f"[DEBUG] R8: avg_dry_cows in animals_df: {animals_df.get('avg_dry_cows', pd.Series()).head()}")
                            # print(f"[DEBUG] R8: avg_dry_cows in manure_df: {manure_df.get('avg_dry_cows', pd.Series()).head()}")
                            df = pd.merge(animals_df, manure_df, on='Facility Name', how='outer', suffixes=('', '_manure'))
                            # print(f"[DEBUG] R8: avg_dry_cows after merge: {df.get('avg_dry_cows', pd.Series()).head()}")
                            results = []
                            for _, row in df.iterrows():
                                result = {col: None for col in columns}
                                result['filename'] = 'R8_animals.csv'
                                for _, param_row in template_params.iterrows():
                                    param_key = param_row['parameter_key']
                                    col_name = param_row['column_search_text']
                                    if col_name in df.columns:
                                        if params['data_types'].get(param_key) == 'numeric':
                                            value = pd.to_numeric(row[col_name], errors='coerce')
                                            if pd.isna(value):
                                                value = 0
                                        else:
                                            value = row[col_name]
                                        result[param_key] = value
                                    elif param_row['row_search_text'] in df.columns:
                                        col_name = param_row['row_search_text']
                                        if params['data_types'].get(param_key) == 'numeric':
                                            value = pd.to_numeric(row[col_name], errors='coerce')
                                            if pd.isna(value):
                                                value = 0
                                        else:
                                            value = row[col_name]
                                        result[param_key] = value
                                results.append(result)
                            df = pd.DataFrame(results)
                            # print(f"[DEBUG DataFrame head]\n{df.head()}")
                        else:
                            # Process PDF files
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
                            with mp.Pool(num_cores) as pool:
                                process_pdf_partial = partial(process_pdf, template_params=template_params, 
                                                           columns=columns, data_types=params['data_types'])
                                results = pool.map(process_pdf_partial, pdf_files)
                            df = pd.DataFrame([r for r in results if r is not None])
                            # print(f"[DEBUG DataFrame head]\n{df.head()}")
                        # Convert numeric columns
                        for col in df.columns:
                            if params['data_types'].get(col) == 'numeric':
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Calculate metrics
                        df = calculate_metrics(df)
                        # Convert to pretty names and ensure all columns exist
                        df = df.rename(columns=params['snake_to_pretty'])
                        for pretty_name in params['snake_to_pretty'].values():
                            if pretty_name not in df.columns:
                                df[pretty_name] = np.nan
                        # Save results
                        os.makedirs(output_folder, exist_ok=True)
                        for f in os.listdir(output_folder):
                            if f.endswith('.csv'):
                                os.remove(os.path.join(output_folder, f))
                        df.to_csv(os.path.join(output_folder, f"{name}.csv"), index=False)
    if consolidate_data:
        consolidate_outputs()

def validate_and_fix_zip(df):
    if 'Zip' in df.columns:
        df['Zip'] = df['Zip'].astype(str)
        df['Zip'] = df['Zip'].str.replace(r'[^\d]', '', regex=True)
        df.loc[df['Zip'] == '', 'Zip'] = np.nan 
    return df

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

    # Load CADD data
    print("Loading CADD data...")
    cadd_facilities = pd.read_csv('data/CADD/CADD_Facility General Information_v1.0.0.csv')
    cadd_herd_size = pd.read_csv('data/CADD/CADD_Facility Herd Size_v1.0.0.csv')
    
    # Rename CADD columns to avoid conflicts
    cadd_facilities = cadd_facilities.rename(columns={
        'Latitude': 'Latitude_cadd',
        'Longitude': 'Longitude_cadd',
        'StreetAddress': 'StreetAddress_cadd',
        'City': 'City_cadd',
        'County': 'County_cadd',
        'ZipCode': 'ZipCode_cadd',
        'RegionalWaterboard': 'RegionalWaterboard_cadd'
    })

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
                for col in ["USDA Nitrogen % Deviation", "UCCE Nitrogen % Deviation", "Wastewater to Reported Milk Ratio", "Wastewater to Estimated Milk Ratio"]:
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
            
            # Validate and fix cow columns
            combined_df = validate_and_fix_zip(combined_df)

            # Initialize location columns if they don't exist
            for col in ['Latitude', 'Longitude', 'Street Address', 'City', 'County', 'Zip']:
                if col not in combined_df.columns:
                    combined_df[col] = np.nan

            # Merge with CADD data
            print(f"\nMerging with CADD data for {year} {region}...")
            
            # 1. Exact name matching
            print("Performing exact name matching...")
            exact_matches = pd.merge(
                combined_df,
                cadd_facilities,
                left_on='Dairy Name',
                right_on='FacilityName',
                how='left',
                suffixes=('', '_cadd')
            )
            
            # 2. Fuzzy matching for remaining unmatched facilities
            print("Performing fuzzy matching...")
            unmatched_mask = exact_matches['FacilityName'].isna()
            unmatched_df = exact_matches[unmatched_mask].copy()
            matched_df = exact_matches[~unmatched_mask].copy()
            
            def find_fuzzy_match(row):
                if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
                    return None
                
                # Calculate distances to all CADD facilities
                distances = []
                for _, cadd_row in cadd_facilities.iterrows():
                    if pd.isna(cadd_row['Latitude_cadd']) or pd.isna(cadd_row['Longitude_cadd']):
                        continue
                    
                    # Calculate distance in meters
                    lat1, lon1 = float(row['Latitude']), float(row['Longitude'])
                    lat2, lon2 = float(cadd_row['Latitude_cadd']), float(cadd_row['Longitude_cadd'])
                    distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111000  # rough conversion to meters
                    
                    # Check if names have at least one word in common
                    name1_words = set(str(row['Dairy Name']).lower().split())
                    name2_words = set(str(cadd_row['FacilityName']).lower().split())
                    common_words = name1_words.intersection(name2_words)
                    
                    if distance <= 100 and len(common_words) > 0:
                        distances.append((distance, cadd_row))
                
                if distances:
                    # Return the closest match
                    return min(distances, key=lambda x: x[0])[1]
                return None
            
            # Apply fuzzy matching
            print("Applying fuzzy matching to unmatched facilities...")
            fuzzy_matches = []
            for _, row in unmatched_df.iterrows():
                match = find_fuzzy_match(row)
                if match is not None:
                    row_dict = row.to_dict()
                    row_dict.update({
                        'Latitude_cadd': match['Latitude_cadd'],
                        'Longitude_cadd': match['Longitude_cadd'],
                        'StreetAddress_cadd': match['StreetAddress_cadd'],
                        'City_cadd': match['City_cadd'],
                        'County_cadd': match['County_cadd'],
                        'ZipCode_cadd': match['ZipCode_cadd'],
                        'RegionalWaterboard_cadd': match['RegionalWaterboard_cadd'],
                        'FacilityName': match['FacilityName']
                    })
                    fuzzy_matches.append(row_dict)
            
            # Combine exact and fuzzy matches
            if fuzzy_matches:
                fuzzy_matches_df = pd.DataFrame(fuzzy_matches)
                final_df = pd.concat([matched_df, fuzzy_matches_df], ignore_index=True)
            else:
                final_df = matched_df
            
            # Merge with CADD herd size data for current year
            print("Merging with CADD herd size data...")
            current_year_herd = cadd_herd_size[cadd_herd_size['Year'] == int(year)].copy()
            if not current_year_herd.empty:
                final_df = pd.merge(
                    final_df,
                    current_year_herd,
                    left_on='FacilityName',
                    right_on='CADDID',
                    how='left',
                    suffixes=('', '_cadd_herd')
                )
            
            # Update geocoding with CADD data where available
            print("Updating geocoding with CADD data...")
            # First ensure all location columns exist
            for col in ['Latitude', 'Longitude', 'Street Address', 'City', 'County', 'Zip']:
                if col not in final_df.columns:
                    final_df[col] = np.nan
            
            # Then update with CADD data
            final_df['Latitude'] = final_df['Latitude_cadd'].fillna(final_df['Latitude'])
            final_df['Longitude'] = final_df['Longitude_cadd'].fillna(final_df['Longitude'])
            final_df['Street Address'] = final_df['StreetAddress_cadd'].fillna(final_df['Street Address'])
            final_df['City'] = final_df['City_cadd'].fillna(final_df['City'])
            final_df['County'] = final_df['County_cadd'].fillna(final_df['County'])
            final_df['Zip'] = final_df['ZipCode_cadd'].fillna(final_df['Zip'])
            
            # Geocode remaining addresses
            print("Geocoding remaining addresses...")
            address_col = next((col for col in ['Dairy Address', 'Facility Address'] 
                                if col in final_df.columns), None)
            if address_col:
                unique_addresses = final_df[address_col].dropna().unique()
                new_geocodes = 0
                for address in unique_addresses:
                    # Get the county for this address
                    county = final_df[final_df[address_col] == address]['County'].iloc[0] if not final_df[final_df[address_col] == address].empty else None
                    lat, lng, geocoded_county = geocode_address(address, geocoding_cache, county=county, try_again=False)
                    if lat is not None:
                        new_geocodes += 1
                    # Parse address components
                    street, city, parsed_county, zip_code = parse_address(address)
                    final_county = geocoded_county or parsed_county
                    mask = final_df[address_col] == address
                    final_df.loc[mask, 'Latitude'] = lat
                    final_df.loc[mask, 'Longitude'] = lng
                    final_df.loc[mask, 'Street Address'] = street
                    final_df.loc[mask, 'City'] = city
                    final_df.loc[mask, 'County'] = final_county
                    final_df.loc[mask, 'Zip'] = zip_code
                print(f"Geocoding complete: {new_geocodes} addresses geocoded")
            
            # consultant metrics only for R5 and 2023
            if year == 2023 and region == "R5":
                consultant_metrics = calculate_consultant_metrics(final_df)
                metrics_file = f"outputs/consolidated/{year}_{region}_consultant_metrics.csv"
                consultant_metrics.to_csv(metrics_file, index=False)
                print(f"Saved consultant metrics to {metrics_file}")
            
            # Remove snake_case columns from final output
            snake_case_cols = [col for col in final_df.columns if '_' in col and col not in ['Dairy_Name', 'Dairy_Address']]
            final_df = final_df.drop(columns=snake_case_cols)

            # Ensure Dairy Name is valid: if missing or <3 chars, use filename (without .pdf)
            if 'Dairy Name' in final_df.columns and 'filename' in final_df.columns:
                def fix_dairy_name(row):
                    name = str(row['Dairy Name']) if pd.notna(row['Dairy Name']) else ''
                    if len(name.strip()) < 3:
                        fname = str(row['filename'])
                        if fname.lower().endswith('.pdf'):
                            fname = fname[:-4]
                        return fname
                    return name
                final_df['Dairy Name'] = final_df.apply(fix_dairy_name, axis=1)

            # Fill missing counties using ZIP code before saving
            final_df = fill_missing_counties_with_zip(final_df)

            # Validate cow columns one final time before saving
            print("\nPerforming final validation of cow columns...")
            final_df = validate_and_fix_zip(final_df)

            output_file = f"outputs/consolidated/{year}_{region}_master.csv"
            final_df.to_csv(output_file, index=False)
            print(f"Saved consolidated data to {output_file}")
            print(f"Total records: {len(final_df)}")

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
    geocoder = ArcGIS(user_agent="ca_cafo_compliance")

    successful_locations = []
    max_retries = 2
    retry_delay = 3
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
    # Clean ZIP code - keep only numeric characters
    zip_code = ''.join(filter(str.isdigit, state_zip[-5:])) if len(state_zip) >= 5 else None
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

def calculate_consultant_metrics(df):
    """Calculate average under/over-reporting metrics for each consultant."""
    # Group by consultant
    consultant_groups = df.groupby('Consultant')
    
    metrics = []
    for consultant, group in consultant_groups:
        manure_avg = group['Calculated Manure Factor'].mean() if 'Calculated Manure Factor' in group.columns else None
        manure_std = group['Calculated Manure Factor'].std() if 'Calculated Manure Factor' in group.columns else None
        
        wastewater_avg = group['Wastewater to Reported Milk Ratio'].mean() if 'Wastewater to Reported Milk Ratio' in group.columns else None
        wastewater_std = group['Wastewater to Reported Milk Ratio'].std() if 'Wastewater to Reported Milk Ratio' in group.columns else None
        
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

def fill_missing_counties_with_zip(df, zip_col='Zip', county_col='County', mapping_file='data/zipcode_to_county.csv'):
    """Fill missing County values using Zip and a ZIP-to-county mapping CSV."""
    # Load ZIP to county mapping
    zip_map = pd.read_csv(mapping_file, dtype={'zip': str, 'county_name': str})
    zip_to_county = dict(zip(zip_map['zip'], zip_map['county_name']))
    # Only fill where county is missing and zip is present
    mask = df[county_col].isna() & df[zip_col].notna()
    df.loc[mask, county_col] = df.loc[mask, zip_col].astype(str).map(zip_to_county)
    return df

if __name__ == "__main__":
    main(test_mode=False)