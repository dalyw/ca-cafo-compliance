import glob
import os
import pandas as pd
import numpy as np
import re

GDRIVE_BASE = "/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/Manure Trucking Network Analysis"
PARAMETERS_DF = pd.read_csv("ca_cafo_compliance/data/parameters.csv")
YEARS = [2023, 2024]
REGIONS = sorted(pd.read_csv("ca_cafo_compliance/data/county_region.csv")["region"].unique())

_KEEP_UPPER = {"LLC", "GPM", "INC", "CA", "DBA", "NA", "N/A"}


def build_parameter_dicts(manifest_only=False):
    """Build parameter mapping dicts from parameters.csv."""
    df = PARAMETERS_DF[PARAMETERS_DF["manifest_type"].isin(["manure", "wastewater", "both"])] if manifest_only else PARAMETERS_DF
    return {
        "key_to_name": dict(zip(df["parameter_key"], df["parameter_name"])),
        "key_to_type": dict(zip(df["parameter_key"], df["data_type"])),
        "key_to_default": dict(zip(df["parameter_key"], df["default"])),
    }


def coerce_columns(df):
    """Coerce columns to their data_type from parameters.csv (in-place)."""
    for dtype, names in PARAMETERS_DF.groupby("data_type")["parameter_name"].apply(set).items():
        cols = [c for c in df.columns if c in names]
        if not cols:
            continue
        if dtype == "numeric":
            df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        elif dtype == "date":
            for c in cols:
                df[c] = pd.to_datetime(df[c], format="mixed", dayfirst=False, errors="coerce")
        elif dtype == "boolean":
            for c in cols:
                df[c] = df[c].astype(str).str.strip().str.upper().map({"TRUE": True, "FALSE": False, "NAN": None})
    return df


def _smart_title(s):
    """Title-case preserving apostrophes and common abbreviations."""
    result = []
    for word in s.split():
        if word.upper().strip(".,;:()") in _KEEP_UPPER:
            result.append(word.upper())
        else:
            titled = word.title()
            titled = re.sub(r"(['\u2019])([A-Z])", lambda m: m.group(1) + m.group(2).lower(), titled)
            result.append(titled)
    return " ".join(result)


def extract_value_from_line(line, item_order=None, ignore_before=None, ignore_after=None):
    """Extract value from line using item_order, ignore_before, and ignore_after."""
    line = str(line)
    if item_order is None and not ignore_before and not ignore_after:
        return line.strip()

    # Process ignore_before
    if ignore_before and ignore_before != "NA":
        if ignore_before in ("str", "num"):
            if m := re.search(r"([-+]?\d*\.?\d+)", line.strip()):
                line = m.group(1)
        else:
            idx = line.lower().find(str(ignore_before).lower())
            if idx != -1:
                line = line[idx + len(str(ignore_before)):].strip()

    # Process ignore_after
    if ignore_after and ignore_after != "NA":
        if ignore_after == "str":
            if m := re.match(r"([-+]?\d*\.?\d+)", line.strip()):
                line = m.group(1)
        else:
            markers = ignore_after if isinstance(ignore_after, list) else [ignore_after]
            idx = len(line)
            line_lower = line.lower()
            for m in markers:
                if not m:
                    continue
                if m == "first_number":
                    if num_m := re.search(r"\(?\d{2,}", line):
                        idx = min(idx, num_m.start())
                else:
                    i = line_lower.find(str(m).lower())
                    if i != -1:
                        idx = min(idx, i)
            line = line[:idx].strip()

    # Select item by order
    if item_order is not None and not pd.isna(item_order):
        parts = line.split()
        idx = int(item_order)
        return parts[idx] if 0 <= idx < len(parts) else ""

    return line.strip()


def convert_to_numeric(value, data_type):
    """Convert value to numeric format based on data type."""
    if value is None:
        return 0 if data_type == "numeric" else None
    if data_type == "numeric":
        try:
            return float(str(value).replace(",", ""))
        except ValueError:
            return value
    return value


def clean_common_errors(text):
    """Clean up common OCR errors while preserving structure."""
    # Case-insensitive replacements
    case_insensitive = {
        "6pm": "GPM", "galions": "Gallons", "galons": "Gallons", "gailons": "Gallons",
        "pek load": "per load", " pek ": " per ", "jons": "tons", "waste water": "Wastewater",
        "havler": "Hauler", "hayler": "Hauler", "haulers calcs": "Hauler's Calculations",
        "haulers calculations": "Hauler's Calculations", "hauler's calculations": "Hauler's Calculations",
        "Arriount": "Amount", " tous": " tons", " tong": "tons",
    }
    for old, new in case_insensitive.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    # Case-sensitive replacements
    case_sensitive = {
        "|": "I", "0O": "O", "1I": "I", "S5": "S", "Ibs": "lbs", "/bs": "lbs", "©": "", "; ": "",
        "Maxiumu": "Maximum", "FaciIity": "Facility", "CattIe": "Cattle", "KjeIdahl": "Kjeldahl",
        "Sroker": "Broker", " I Broker": "Broker", " I Composting Facility ": "Composting Facility",
        "HauIing": "Hauling", "Solide": "Solids", " Doing": " Dairy", "Daing": "Dairy",
        "Dainy": "Dairy", "Daire": "Dairy", "Cubie": "Cubic", "[]": "", "[X]": "", "> ": "", "[ ] ": "",
    }
    for old, new in case_sensitive.items():
        text = text.replace(old, new)

    # Remove certain characters
    for char in ["|", ",", "=", ":", "___"]:
        text = text.replace(char, "")

    # Fix number-letter confusions
    text = re.sub(r"(\d)O(\d)", r"\1O\2", text)
    text = re.sub(r"(\d)[lI](\d)", r"\1l\2", text)
    text = re.sub(r"([a-zA-Z])[0O]([a-zA-Z])", r"\1O\2", text)
    text = re.sub(r"([a-zA-Z])I([a-zA-Z])", r"\1I\2", text)

    # Cleanup spacing and formatting
    text = re.sub(r"\s*[/-]\s*", lambda m: m.group(0).strip(), text)
    text = re.sub(r"_+", "", text)
    text = re.sub(r"[ \t\f\v\r]+", " ", text)
    text = "\n".join([re.sub(r"^\s*;\s*", "", line) for line in text.split("\n") if line.strip()])

    return text


def get_default_value(param_key, data_types, defaults):
    """Get default value for parameter with type conversion."""
    default = defaults.get(param_key)
    dtype = data_types.get(param_key, "text")
    if pd.isna(default) or default == "NA":
        return np.nan if dtype == "numeric" else None
    if dtype == "numeric":
        try:
            return float(default)
        except:
            return np.nan
    return default


def extract_parameters_from_text(text, template, param_locations_df, data_types, defaults):
    """Extract all parameters for a template. Returns dict with parameter_key as keys."""
    result = {}
    for _, row in param_locations_df[param_locations_df["template"] == template].iterrows():
        result[row["parameter_key"]] = find_parameter_value(text, row, data_types, defaults)
    return result


def find_parameter_value(ocr_text, row, data_types, defaults):
    """Extract parameter value from OCR text based on parameter_locations row."""
    param_key = row["parameter_key"]
    data_type = data_types.get(param_key, "text")
    default = lambda: get_default_value(param_key, data_types, defaults)

    if not ocr_text:
        return default()

    # Reduce search area using page_search_text
    search_text = ocr_text
    if pd.notna(page_search := row.get("page_search_text")) and page_search != "NA":
        if (pos := ocr_text.lower().find(str(page_search).lower())) == -1:
            return default()
        search_text = ocr_text[pos + len(str(page_search)):]

    row_search_text = row["row_search_text"]
    direction = str(row.get("search_direction", "")).lower()
    item_order = row.get("item_order", pd.NA)
    ignore_before = row.get("ignore_before")
    ignore_after = row.get("ignore_after")

    if isinstance(ignore_after, str) and "|" in ignore_after:
        ignore_after = [s.strip() for s in ignore_after.split("|") if s.strip()]

    if pd.isna(row_search_text) or not str(row_search_text).strip() or not direction:
        return default()

    search_lower = str(row_search_text).lower()
    lines = [ln.strip() for ln in search_text.split("\n")]
    non_empty = [ln for ln in lines if ln]

    find_line_idx = lambda line_list: next((i for i, ln in enumerate(line_list) if search_lower in ln.lower()), None)
    
    def next_non_empty(start_idx):
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip():
                return j, lines[j]
        return None, None

    if (phrase_idx := find_line_idx(non_empty)) is None:
        return default()

    line = non_empty[phrase_idx]
    actual_idx = find_line_idx(lines)
    extracted_text = None

    if direction == "right":
        idx = line.lower().find(search_lower)
        if idx != -1:
            line = line[idx + len(str(row_search_text)):].strip()
        extracted_text = extract_value_from_line(line, item_order, ignore_before, ignore_after)

    elif direction == "above":
        if phrase_idx > 0:
            extracted_text = extract_value_from_line(non_empty[phrase_idx - 1], item_order, ignore_before, ignore_after)

    elif direction == "below":
        if actual_idx is not None:
            next_idx, next_line = next_non_empty(actual_idx)
            if next_line:
                if ignore_before and ignore_before != "NA" and next_line.lower().startswith(str(ignore_before).lower()):
                    return default()
                extracted_text = extract_value_from_line(next_line, item_order, ignore_before, ignore_after)

    elif direction == "right_below":
        section_starts = ["enter the amount", "process wastewater", "written agreement", "method used"]
        pos = line.lower().find(search_lower)
        right_text = line[pos + len(str(row_search_text)):].strip() if pos != -1 else ""
        
        if right_text:
            right_text_cleaned = extract_value_from_line(right_text, item_order, ignore_before, ignore_after)
            if actual_idx is not None:
                _, next_line = next_non_empty(actual_idx)
                if next_line and not any(next_line.lower().startswith(s) for s in section_starts):
                    next_line_text = extract_value_from_line(next_line, item_order, ignore_before, ignore_after)
                    extracted_text = f"{right_text_cleaned} {next_line_text}"
                else:
                    extracted_text = right_text_cleaned
            else:
                extracted_text = right_text_cleaned
        else:
            if actual_idx is not None:
                _, next_line = next_non_empty(actual_idx)
                extracted_text = extract_value_from_line(next_line, item_order, ignore_before, ignore_after) if next_line else None

    # Convert and apply item_order
    value = None
    if extracted_text and extracted_text.strip():
        if pd.isna(item_order) or item_order == -1:
            value = convert_to_numeric(extracted_text, data_type)
        else:
            try:
                parts = str(extracted_text).split()
                k = int(item_order)
                value = convert_to_numeric(parts[k], data_type) if 0 <= k < len(parts) else None
            except (TypeError, ValueError):
                value = convert_to_numeric(extracted_text, data_type)

    # Return default if no valid value
    if value is None or (data_type == "numeric" and (pd.isna(value) or value == 0)) or value in ("N/A", "NA", "."):
        return default()

    return _smart_title(value) if isinstance(value, str) else value