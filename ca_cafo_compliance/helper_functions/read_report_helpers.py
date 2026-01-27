import pandas as pd
import numpy as np
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

GDRIVE_BASE = '/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/ca_cafo_manifests'

def extract_value_from_line(line, item_order=None, ignore_before=None, ignore_after=None):
    """Extract value from a line using item_order, ignore_before, and ignore_after."""
    if not isinstance(line, str):
        line = str(line)

    if item_order is None and not ignore_before and not ignore_after:
        return line.strip()

    # Process ignore_before first (remove everything before and including the marker)
    if ignore_before and ignore_before != "NA":
        if ignore_before == "str":
            # If ignore_before is 'str', extract first numeric sequence
            match = re.search(r"([-+]?\d*\.?\d+)", line.strip())
            if match:
                line = match.group(1)
            else:
                line = line
        elif ignore_before == "num":
            # If ignore_before is 'num', extract first number
            match = re.search(r"([-+]?\d*\.?\d+)", line.strip())
            if match:
                line = match.group(1)
            else:
                line = line
        else:
            # Find the ignore_before marker and take everything after it
            idx = line.lower().find(str(ignore_before).lower())
            if idx != -1:
                line = line[idx + len(str(ignore_before)) :].strip()
            else:
                # If marker not found, return empty (field might be in different format)
                line = line

    # Then process ignore_after (remove everything after and including the marker)
    if ignore_after and ignore_after != "NA":
        if ignore_after == "str":
            # If ignore_after is 'str', trim at first non-numeric character after a numeric sequence
            match = re.match(r"([-+]?\d*\.?\d+)", line.strip())
            if match:
                line = match.group(1)
            else:
                line = ""
        elif ignore_after == "num":
            # If ignore_after is 'num', extract first number only
            match = re.match(r"([-+]?\d*\.?\d+)", line.strip())
            if match:
                line = match.group(1)
            else:
                line = line
        else:
            # Find the ignore_after marker and take everything before it
            idx = line.lower().find(str(ignore_after).lower())
            if idx != -1:
                line = line[:idx].strip()
            # If marker not found, keep the line as is

    # Optionally select item by order
    if item_order is not None and not pd.isna(item_order):
        parts = [p for p in line.split() if p]
        idx = int(item_order)
        if 0 <= idx < len(parts):
            result = parts[idx]
        else:
            result = ""
    else:
        result = line.strip()

    return result


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
        "©": "",
        "Maxiumu": "Maximum",
        "FaciIity": "Facility",
        "CattIe": "Cattle",
        "KjeIdahl": "Kjeldahl",
        "Sroker": "Broker",
        " I Broker": "Broker",
        " I Composting Facility ": "Composting Facility",
        "HauIing": "Hauling",
        "Solide": "Solids",
        "Doing": "Dairy",
        "Tous": "Tons",
        "[X]": "", # for false checkmarks in llmwhisperer
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

    # remove spaces before and after "/"
    text = re.sub(r"\s*/\s*", "/", text)
    
    # Remove underscores from OCR reading of form
    text = re.sub(r"_+", "", text)
    
    # Clean up whitespace with more than 1 space / tab
    text = re.sub(r"[ \t\f\v\r]+", " ", text)
    # Strip leading semicolon + space from lines (e.g. "; E.R. Prins Dairy")
    text = "\n".join([re.sub(r"^\s*;\s*", "", line) for line in text.split("\n")])
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
    """Extract a parameter value from OCR text based on one row from parameter_locations."""
    param_key = row["parameter_key"]
    data_type = data_types.get(param_key, "text")

    if not ocr_text:
        return get_default_value(param_key, data_types, defaults)

    # reduce the search area using page_search_text (case-insensitive)
    page_search_text = row.get("page_search_text", pd.NA)
    search_text = ocr_text
    if not pd.isna(page_search_text) and page_search_text != "NA":
        pos = ocr_text.lower().find(str(page_search_text).lower())
        if pos == -1:
            return get_default_value(param_key, data_types, defaults)
        search_text = ocr_text[pos + len(str(page_search_text)) :]

    row_search_text = row["row_search_text"]
    direction = row["search_direction"]
    item_order = row.get("item_order", pd.NA)
    ignore_before = row.get("ignore_before", None)
    ignore_after = row.get("ignore_after", None)

    if pd.isna(row_search_text) or not str(row_search_text).strip() or pd.isna(direction):
        return get_default_value(param_key, data_types, defaults)

    row_search_text = str(row_search_text)

    # Keep all lines including empty ones for empty line detection
    all_lines = search_text.split("\n")
    lines = [ln.strip() for ln in all_lines]
    non_empty_lines = [ln for ln in lines if ln]

    # Debug which lines are being searched for destination fields
    if param_key in ("destination_name", "destination_address"):
        print(f"[DEBUG] param_key={param_key}")
        print(f"[DEBUG]   page_search_text={page_search_text!r}")
        print(f"[DEBUG]   row_search_text={row_search_text!r}")
        print(f"[DEBUG]   non_empty_lines={len(non_empty_lines)}")

    phrase_line_idx = next(
        (i for i, ln in enumerate(non_empty_lines) if row_search_text.lower() in ln.lower()),
        None,
    )

    extracted_text = None
    selected_line = None
    selected_line_text = None
    if phrase_line_idx is not None:
        d = str(direction).lower()

        if d == "right":
            line = non_empty_lines[phrase_line_idx]
            phrase_idx = line.lower().find(row_search_text.lower())
            if phrase_idx != -1:
                selected_line = phrase_line_idx
                selected_line_text = non_empty_lines[selected_line]
                # "right" means value is to the right of row_search_text. take everything after it
                ignore_before = row_search_text

        elif d == "below":
            # Find the actual line index in the full lines list (including empty lines)
            phrase_actual_idx = None
            for i, ln in enumerate(lines):
                if row_search_text.lower() in ln.lower():
                    phrase_actual_idx = i
                    break
            
            if phrase_actual_idx is not None:
                # Look for next non-empty line, but also check if current line is empty
                found_line_idx = None
                for j in range(phrase_actual_idx + 1, len(lines)):
                    if lines[j]:
                        # Check if this line starts with ignore_before (indicating empty field)
                        if ignore_before and ignore_before != "NA":
                            if lines[j].lower().startswith(str(ignore_before).lower()):
                                # Field is empty, return default
                                return get_default_value(param_key, data_types, defaults)
                        # Use this line for extraction
                        selected_line_text = lines[j]
                        found_line_idx = j
                        break

                if param_key in ("destination_name", "destination_address"):
                    print(f"[DEBUG]   phrase_actual_idx={phrase_actual_idx}")
                    print(f"[DEBUG]   selected_line_idx={found_line_idx}")
                    print(f"[DEBUG]   selected_line_text={selected_line_text!r}")
                
                if selected_line_text:
                    extracted_text = extract_value_from_line(
                        selected_line_text, item_order, ignore_before, ignore_after
                    )
                    
                    # If extracted text is empty, check if next line starts with ignore_before
                    if (not extracted_text or not extracted_text.strip()) and ignore_before and ignore_before != "NA":
                        if found_line_idx is not None and found_line_idx + 1 < len(lines) and lines[found_line_idx + 1]:
                            if lines[found_line_idx + 1].lower().startswith(str(ignore_before).lower()):
                                return get_default_value(param_key, data_types, defaults)

        elif d == "above":
            if phrase_line_idx > 0:
                selected_line = phrase_line_idx - 1
                selected_line_text = non_empty_lines[selected_line]
        
        # Extract text for "right" and "above" directions
        if d in ["right", "above"] and selected_line_text:
            extracted_text = extract_value_from_line(
                        selected_line_text, item_order, ignore_before, ignore_after
                    )
            
            # If extracted text is empty or just whitespace, check if next line starts with ignore_before
            if (not extracted_text or not extracted_text.strip()) and ignore_before and ignore_before != "NA":
                # Find the actual line in full lines list
                actual_line_idx = None
                for i, ln in enumerate(lines):
                    if ln == selected_line_text:
                        actual_line_idx = i
                        break
                
                if actual_line_idx is not None and actual_line_idx + 1 < len(lines):
                    next_line = lines[actual_line_idx + 1]
                    if next_line and next_line.lower().startswith(str(ignore_before).lower()):
                        # Field is empty, return default
                        return get_default_value(param_key, data_types, defaults)

    # Convert + apply item_order
    value = 0 if data_type == "numeric" else None
    if extracted_text and extracted_text.strip():
        if pd.isna(item_order) or item_order == -1:
            value = convert_to_numeric(extracted_text, data_type)
        else:
            try:
                k = int(item_order)
                parts = str(extracted_text).split()
                if 0 <= k < len(parts):
                    value = convert_to_numeric(parts[k], data_type)
            except (TypeError, ValueError):
                value = convert_to_numeric(extracted_text, data_type)

    # get default value if nothing else returned
    if value is None or (data_type == "numeric" and (pd.isna(value) or value == 0)):
        value = get_default_value(param_key, data_types, defaults)

    if type(value) == str:
        value = value.title() # title case for all string outputs
    return value