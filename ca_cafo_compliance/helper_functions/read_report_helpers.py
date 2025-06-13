import pandas as pd
import numpy as np
import os
import re

# Dictionary of conversion factors (cf)
cf_df = pd.read_csv('ca_cafo_compliance/data/conversion_factors.csv')
cf =  {row['NAME']: float(row['VALUE']) for _, row in cf_df.iterrows()}

YEARS = [2023, 2024]

# Read unique regions from county_region.csv
county_region_df = pd.read_csv('ca_cafo_compliance/data/county_region.csv')
REGIONS = sorted(county_region_df['region'].unique().tolist())

# Create consultant mapping from templates.csv
templates_df = pd.read_csv('ca_cafo_compliance/data/templates.csv')
consultant_mapping = dict(zip(templates_df['template_key'], templates_df['template_name']))

def extract_value_from_line(line, item_order=None, ignore_before=None, ignore_after=None, param_key=None):
    """Extract value from a line using item_order, ignore_before, and ignore_after. If none, return full line."""
    if not isinstance(line, str):
        line = str(line)

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

def clean_common_errors(text):
    """Clean up common OCR errors in text while preserving structure."""
    # Common OCR error replacements
    replacements = {
        '|': 'I',
        '0O': 'O',
        '1I': 'I',
        'S5': 'S',
        'Ibs': 'lbs',
        '/bs': 'lbs',
        'Maxiumu': 'Maximum',
        'FaciIity': 'Facility',
        'CattIe': 'Cattle',
        'KjeIdahl': 'Kjeldahl',
        'MiIk': 'Milk'
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove certain characters
    for char in ['|', ',', '=', ':', '___']:
        text = text.replace(char, '')
    
    # Fix number-letter confusions
    text = re.sub(r'(\d)O(\d)', r'\1O\2', text)
    text = re.sub(r'(\d)l(\d)', r'\1l\2', text)
    text = re.sub(r'(\d)I(\d)', r'\1I\2', text)
    text = re.sub(r'([a-zA-Z])0([a-zA-Z])', r'\1O\2', text)
    text = re.sub(r'([a-zA-Z])l([a-zA-Z])', r'\1I\2', text)
    text = re.sub(r'([a-zA-Z])I([a-zA-Z])', r'\1I\2', text)
    
    # Clean up whitespace
    text = text.replace('  ', ' ')
    text = '\n'.join([line for line in text.split('\n') if line.strip()])
    
    return text

def load_ocr_text(pdf_path):
    """Load OCR text from file."""
    pdf_dir = os.path.dirname(pdf_path)
    parent_dir = os.path.dirname(pdf_dir)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Try ai_ocr_output first, then ocr_output
    for ocr_dir in ['ai_ocr_output', 'ocr_output']:
        text_file = os.path.join(parent_dir, ocr_dir, f'{pdf_name}.txt')
        if os.path.exists(text_file):
            with open(text_file, 'r') as f:
                text = f.read()
                return clean_common_errors(text)
    
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
        result[param_key] = value
    return result