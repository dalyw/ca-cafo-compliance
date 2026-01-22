import pandas as pd
import numpy as np
import os
import re

# Dictionary of conversion factors (cf)
cf_df = pd.read_csv("ca_cafo_compliance/data/conversion_factors.csv")
cf = {row["NAME"]: float(row["VALUE"]) for _, row in cf_df.iterrows()}

YEARS = [2023, 2024]

# Read unique regions from county_region.csv
county_region_df = pd.read_csv("ca_cafo_compliance/data/county_region.csv")
REGIONS = sorted(county_region_df["region"].unique().tolist())

# Create consultant mapping from templates.csv
templates_df = pd.read_csv("ca_cafo_compliance/data/templates.csv")
consultant_mapping = dict(
    zip(templates_df["template_key"], templates_df["template_name"])
)

# Known California cities for address parsing
CA_CITIES = {
    'chowchilla', 'fresno', 'kerman', 'madera', 'hanford', 
    'visalia', 'tulare', 'bakersfield', 'merced', 'modesto',
    'stockton', 'lodi', 'manteca', 'tracy', 'riverdale',
    'selma', 'reedley', 'dinuba', 'porterville', 'delano',
    'corcoran', 'lemoore', 'coalinga', 'gustine', 'newman',
    'hilmar', 'turlock', 'atwater', 'livingston', 'dos palos',
    'firebaugh', 'mendota', 'caruthers', 'fowler', 'parlier',
    'sanger', 'clovis', 'kingsburg', 'orange cove', 'exeter',
    'lindsay', 'woodlake', 'farmersville', 'earlimart', 'pixley',
    'tipton', 'terra bella', 'ducor', 'richgrove', 'mcfarland',
    'wasco', 'shafter', 'arvin', 'lamont', 'taft', 'buttonwillow',
    'winton', 'dos palos'
}

# California counties in the dairy regions
CA_COUNTIES = {'madera', 'fresno', 'kings', 'kern', 'tulare', 'merced', 'stanislaus', 'san joaquin'}

# Address label patterns to skip when parsing
ADDRESS_LABEL_PATTERNS = [
    r'^name\s*(of\s+operator)?:?$',
    r'^name\s*(of\s+dairy\s+facility)?:?$',
    r'^(facility\s+)?address:?$',
    r'^number\s+and\s+street$',
    r'^city$',
    r'^county$',
    r'^state$',
    r'^zip\s*code$',
    r'^contact\s+person.*$',
    r'^name$',
    r'^phone\s*number$',
    r'^address$',
    r'^adress$',
    r'^street\s+and\s+nearest.*$',
    r"^assessor'?s?\s+parcel.*$",
    r'^operator\s+information$',
    r'^destination\s+information$',
    r'^composting\s+facility.*$',
    r'^contact\s+information.*$',
    r'^last\s+date\s+hauled.*$',
    r'^instructions$',
    r'^manure\s*/?\s*process.*$',
    r'^\d\).*$',
]

# Labels to skip in table cells
TABLE_CELL_LABELS = {
    'city', 'county', 'state', 'zip code', 'zip', 'name', 'phone number',
    'number and street', 'address', 'adress', 'street and nearest cross street'
}


def parse_marker_table_address(line):
    """
    Parse address from a marker-format table line.
    Example: | 2792 W Mt. Whitney AVE | Riverdale | Fresno | Zip Code |
    Returns: (street, city, state, zipcode)
    """
    if '|' not in line:
        return None, None, None, None
    
    cells = [c.strip() for c in line.split('|') if c.strip()]
    
    street = None
    city = None
    state = None
    zipcode = None
    
    for cell in cells:
        cell_lower = cell.lower()
        
        # Skip labels
        if cell_lower in TABLE_CELL_LABELS:
            continue
        
        # Check for zip code
        if re.match(r'^\d{5}(-\d{4})?$', cell):
            zipcode = cell
            continue
        
        # Check for state
        if cell.upper() in ['CA', 'CALIFORNIA']:
            state = 'CA'
            continue
        
        # Check for street address (starts with number, contains street type)
        if re.match(r'^\d+\s+', cell) and not street:
            if re.search(r'(AVE|Ave|Avenue|ST|St|Street|RD|Rd|Road|DR|Dr|Drive|Blvd|Way|Lane|Ln|HWY|Hwy)', cell, re.IGNORECASE):
                street = cell
                continue
        
        # Remaining cells are likely city (if proper name format)
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', cell) and not city:
            if cell.lower() in CA_CITIES:
                city = cell
    
    return street, city, state, zipcode


def is_address_label(line):
    """Check if a line is an address form label (not a value)."""
    for pattern in ADDRESS_LABEL_PATTERNS:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    return False


def extract_address_from_section(text, section_start, section_end=None):
    """
    Extract address from a section of OCR text.
    Handles both fitz (plain text) and marker (table) formats.
    
    Args:
        text: Full OCR text
        section_start: String marking start of section to search
        section_end: Optional string marking end of section
    
    Returns: Formatted address string or None
    """
    # Find the section
    start_idx = text.find(section_start)
    if start_idx == -1:
        return None
    
    if section_end:
        end_idx = text.find(section_end, start_idx)
        section = text[start_idx:end_idx] if end_idx != -1 else text[start_idx:]
    else:
        section = text[start_idx:start_idx + 1000]
    
    street = None
    city = None
    state = None
    zipcode = None
    county = None
    
    for line in section.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Skip labels
        if is_address_label(line):
            continue
        
        # Handle marker table format
        if '|' in line:
            t_street, t_city, t_state, t_zip = parse_marker_table_address(line)
            if t_street and not street:
                street = t_street
            if t_city and not city:
                city = t_city
            if t_state and not state:
                state = t_state
            if t_zip and not zipcode:
                zipcode = t_zip
            continue
        
        # Plain text format (fitz)
        # Check for zip code
        if re.match(r'^\d{5}(-\d{4})?$', line):
            zipcode = line
            continue
        
        # Check for state
        if line.upper() in ['CA', 'CALIFORNIA']:
            state = 'CA'
            continue
        
        # Check for street address
        street_match = re.match(
            r'^(\d+\s+[A-Za-z0-9\s\./]+(?:AVE|Ave|Avenue|ST|St|Street|RD|Rd|Road|DR|Dr|Drive|Blvd|Boulevard|Way|Lane|Ln|HWY|Hwy|Highway)\.?)(?:\s|$)',
            line, re.IGNORECASE
        )
        if street_match:
            street = street_match.group(1).strip()
            continue
        
        # Check for street with fractional numbers (e.g., "10221 Ave 21 1/2")
        street_alt_match = re.match(
            r'^(\d+\s+(?:Ave|Avenue|Road|Rd|Street|St|Dr|Drive)\s+[\d\s/]+)$',
            line, re.IGNORECASE
        )
        if street_alt_match:
            street = street_alt_match.group(1).strip()
            continue
        
        # Check for city
        if line.lower() in CA_CITIES:
            city = line.title()
            continue
        
        # Check for county
        if line.lower() in CA_COUNTIES:
            county = line.title()
            continue

    if street:
        street = ' '.join(street.split())
    if city:
        city = ' '.join(city.split())
    if state:
        state = state.strip().upper()
        if state == 'CALIFORNIA':
            state = 'CA'
    if zipcode:
        zipcode = zipcode.strip()
    
    # Build address string
    parts = []
    if street:
        parts.append(street)
    if city:
        parts.append(city)
    if state and zipcode:
        parts.append(f"{state} {zipcode}")
    elif state:
        parts.append(state)
    elif zipcode:
        parts.append(zipcode)
    
    return ', '.join(parts) if parts else None


def extract_value_from_line(
    line, item_order=None, ignore_before=None, ignore_after=None, param_key=None
):
    """Extract value from a line using item_order, ignore_before, and ignore_after. 
    If none, return full line."""
    if not isinstance(line, str):
        line = str(line)

    if item_order is None and not ignore_before and not ignore_after:
        return line

    # Optionally trim before/after
    if ignore_after:
        if ignore_after == "str":
            # If ignore_after is 'str', trim at first non-numeric character
            # after a numeric sequence
            match = re.match(r"([-+]?\d*\.?\d+)", line.strip())
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
            result = parts[idx]
        else:
            result = ""
    else:
        result = line

    return result


def extract_text_adjacent_to_phrase(
    text,
    phrase,
    direction="right",
    row_search_text=None,
    column_search_text=None,
    item_order=None,
    ignore_before=None,
    ignore_after=None,
    param_key=None,
):
    if not text or not phrase:
        return None
    
    lines = [str(line).strip() for line in text.split("\n") if str(line).strip()]
    
    # Debug: Print lines if extracting dairy_name for R1
    if param_key == "dairy_name":
        print("DEBUG: Lines being searched for dairy_name:")
        for i, l in enumerate(lines):
            print(f"  {i}: {l}")
    
    phrase_line_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if isinstance(line, str) and phrase.lower() in line.lower()
        ),
        None,
    )
    
    if phrase_line_idx is None:
        return None
    if direction == "right":
        line = lines[phrase_line_idx]
        phrase_idx = line.lower().find(phrase.lower())
        if phrase_idx != -1:
            text_after = line[phrase_idx + len(phrase) :].strip()
            # Debug: Print extracted text_after before ignore_after
            if param_key == "dairy_name":
                print(f"DEBUG: Extracted text after phrase: '{text_after}'")
            return extract_value_from_line(
                text_after, item_order, ignore_before, ignore_after, param_key=param_key
            )
    elif direction == "below":
        # Find the next non-blank line after the phrase
        next_line = None
        for j in range(phrase_line_idx + 1, len(lines)):
            if lines[j].strip():
                next_line = lines[j].strip()
                break
        if next_line is not None:
            return extract_value_from_line(
                next_line, item_order, ignore_before, ignore_after, param_key=param_key
            )
    elif direction == "table":
        if row_search_text and column_search_text:
            row_idx = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if isinstance(line, str) and row_search_text.lower() in line.lower()
                ),
                None,
            )
            if row_idx is not None:
                header_parts = [
                    part.strip() for part in str(lines[row_idx]).split() if part.strip()
                ]
                col_idx = next(
                    (
                        i
                        for i, part in enumerate(header_parts)
                        if column_search_text.lower() in part.lower()
                    ),
                    None,
                )
                if col_idx is not None and row_idx + 1 < len(lines):
                    value_parts = [
                        part.strip()
                        for part in str(lines[row_idx + 1]).split()
                        if part.strip()
                    ]
                    if col_idx < len(value_parts):
                        return extract_value_from_line(
                            value_parts[col_idx],
                            item_order,
                            ignore_before,
                            ignore_after,
                            param_key=param_key,
                        )
    elif direction == "above":
        if phrase_line_idx > 0:
            value_line = lines[phrase_line_idx - 1]
            return extract_value_from_line(
                value_line, item_order, ignore_before, ignore_after, param_key=param_key
            )
    return None


def find_value_by_text(page_text, row, data_type, param_key=None):
    if pd.isna(row["row_search_text"]):
        return None
    extracted_text = extract_text_adjacent_to_phrase(
        text=page_text,
        phrase=row["row_search_text"],
        direction=row["search_direction"],
        row_search_text=row["row_search_text"],
        column_search_text=row.get("column_search_text", pd.NA),
        item_order=row["item_order"],
        ignore_before=row["ignore_before"],
        ignore_after=row["ignore_after"] if "ignore_after" in row else None,
        param_key=param_key,
    )
    if extracted_text:
        # If it's a single value, just return it
        if isinstance(extracted_text, str) and len(extracted_text.split()) == 1:
            return convert_to_numeric(extracted_text, data_type)
        item_order = row["item_order"]
        if pd.isna(item_order) or item_order == -1:
            return convert_to_numeric(extracted_text, data_type)
        else:
            parts = extracted_text.split()
            if item_order < len(parts):
                return convert_to_numeric(parts[item_order], data_type)
    return 0 if data_type == "numeric" else None


def convert_to_numeric(value, data_type):
    """Convert a value to numeric format based on data type."""
    if value is None:
        return 0 if data_type == "numeric" else None

    # Remove non-numeric characters
    if data_type == "numeric":
        value = str(value).replace(",", "")
        try:
            return float(value)
        except ValueError:
            return 0
    return value


def clean_common_errors(text):
    """Clean up common OCR errors in text while preserving structure."""
    # Common OCR error replacements
    replacements = {
        "|": "I",
        "0O": "O",
        "1I": "I",
        "S5": "S",
        "Ibs": "lbs",
        "/bs": "lbs",
        "Maxiumu": "Maximum",
        "FaciIity": "Facility",
        "CattIe": "Cattle",
        "KjeIdahl": "Kjeldahl",
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove certain characters
    for char in ["|", ",", "=", ":", "___"]:
        text = text.replace(char, "")

    # Fix number-letter confusions
    text = re.sub(r"(\d)O(\d)", r"\1O\2", text)
    text = re.sub(r"(\d)l(\d)", r"\1l\2", text)
    text = re.sub(r"(\d)I(\d)", r"\1I\2", text)
    text = re.sub(r"([a-zA-Z])0([a-zA-Z])", r"\1O\2", text)
    text = re.sub(r"([a-zA-Z])I([a-zA-Z])", r"\1I\2", text)

    # Clean up whitespace
    text = text.replace("  ", " ")
    text = "\n".join([line for line in text.split("\n") if line.strip()])

    return text


def get_default_value(param_key, data_types, defaults):
    """Get the default value for a parameter, with type conversion."""
    default = defaults.get(param_key, None)
    dtype = data_types.get(param_key, "text")
    if pd.isna(default) or default == "NA":
        return np.nan if dtype == "numeric" else None
    if dtype == "numeric":
        try:
            return float(default)
        except Exception:
            return np.nan
    return default


def find_parameter_value(ocr_text, row, data_types, defaults):
    """Extract a parameter value from OCR text based on the specified row from parameter_locations."""

    search_direction = row.get("search_direction", pd.NA)
    page_search_text = row.get("page_search_text", pd.NA)
    row_search_text = row.get("row_search_text", pd.NA)
    column_search_text = row.get("column_search_text", 'pd.NA')
    param_key = row["parameter_key"]
    data_type = data_types.get(param_key, "text")

    if pd.isna(search_direction):
        print(param_key)
        return get_default_value(param_key, data_types, defaults)
    data_type = data_types.get(row["parameter_key"], "text")
    param_key = row["parameter_key"]
    
    # try:
    search_text = ocr_text
    if not pd.isna(page_search_text):
        clean_search = " ".join(page_search_text.split())
        clean_text = " ".join(ocr_text.split())
        pos = clean_text.find(clean_search)
        if pos == -1:
            return get_default_value(param_key, data_types, defaults)
        search_text = ocr_text[pos + len(page_search_text):]

    if pd.isna(row_search_text):
        print("NA row_search_text for param:", param_key)
        return get_default_value(param_key, data_types, defaults)

    if pd.isna(column_search_text):
        print("NA column_search_text for param:", param_key)
        return get_default_value(param_key, data_types, defaults)
        
    # Search for the value in the appropriate text section (use search_text, not ocr_text)
    value = find_value_by_text(
        page_text=search_text, row=row, data_type=data_type, param_key=param_key
    )
    
    if value is None or (data_type == "numeric" and (pd.isna(value) or value == 0)):
        # Use default if not found or is zero/NA for numeric
        value = get_default_value(param_key, data_types, defaults)
    
    return value
    # except Exception as e:
    #     print(f"Error processing parameter {row['parameter_key']}: {str(e)}")
    #     return get_default_value(param_key, data_types, defaults)


def process_pdf(pdf_path, template_params, columns, data_types, defaults):
    """Process a single PDF file and extract all parameters from OCR text."""
    result = {col: None for col in columns}
    result["filename"] = os.path.basename(pdf_path)
    ocr_text = None

    pdf_dir = os.path.dirname(pdf_path)
    parent_dir = os.path.dirname(pdf_dir)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for ocr_dir in ["marker_output", "tesseract_output"]:
        text_file = os.path.join(parent_dir, ocr_dir, f"{pdf_name}.txt")
        if os.path.exists(text_file):
            with open(text_file, "r") as f:
                text = f.read()
                ocr_text = clean_common_errors(text)

    if not ocr_text:
        # Use defaults for all parameters if OCR text is missing
        for _, row in template_params.iterrows():
            param_key = row["parameter_key"]
            result[param_key] = get_default_value(param_key, data_types, defaults)
        return result
    # Process main report parameters
    for _, row in template_params.iterrows():
        param_key = row["parameter_key"]
        value = find_parameter_value(ocr_text, row, data_types, defaults)
        result[param_key] = value
    return result