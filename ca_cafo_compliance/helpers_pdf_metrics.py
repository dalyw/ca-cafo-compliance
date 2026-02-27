import glob
import os
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
TEMPLATE_KEY_TO_NAME = dict(
    zip(templates_df["template_key"], templates_df["template_name"])
)

GDRIVE_BASE = "/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/ca_cafo_manifests"

OCR_METHODS = ["llmwhisperer", "tesseract", "fitz"]
OCR_DIRS = [f"{m}_output" for m in OCR_METHODS]


def find_pdf_files(folder):
    """Find all PDFs that have at least one OCR text output in sibling OCR directories."""
    original_dir = os.path.join(folder, "original")
    seen = set()
    pdf_files = []
    for ocr_dir in OCR_DIRS:
        ocr_path = os.path.join(folder, ocr_dir)
        if not os.path.exists(ocr_path):
            continue
        for text_file in glob.glob(os.path.join(ocr_path, "*.txt")):
            pdf_name = os.path.basename(text_file).replace(".txt", ".pdf")
            pdf_path = os.path.join(original_dir, pdf_name)
            if pdf_path not in seen and os.path.exists(pdf_path):
                seen.add(pdf_path)
                pdf_files.append(pdf_path)
    return pdf_files


def load_ocr_text(pdf_path):
    """Load and clean OCR text for a PDF by checking sibling OCR output directories.

    Tries each OCR directory in priority order and returns the cleaned text
    from the first one found, or empty string if none exist.
    """
    parent_dir = os.path.dirname(os.path.dirname(pdf_path))
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    for ocr_dir in OCR_DIRS:
        text_file = os.path.join(parent_dir, ocr_dir, f"{pdf_stem}.txt")
        if os.path.exists(text_file):
            with open(text_file, "r") as f:
                return clean_common_errors(f.read())
    return ""


# Parameters metadata
_PARAMETERS_DF = pd.read_csv("ca_cafo_compliance/data/parameters.csv")


def build_parameter_dicts(manifest_only=False):
    """Build parameter mapping dicts from parameters.csv.

    Returns dict with keys: 'key_to_name', 'key_to_type', 'key_to_default'.
    If manifest_only=True, filters to manure/wastewater/both parameters.
    """
    df = _PARAMETERS_DF
    if manifest_only:
        df = df[df["manifest_type"].isin(["manure", "wastewater", "both"])]
    return {
        "key_to_name": dict(zip(df["parameter_key"], df["parameter_name"])),
        "key_to_type": dict(zip(df["parameter_key"], df["data_type"])),
        "key_to_default": dict(zip(df["parameter_key"], df["default"])),
    }


def coerce_numeric_columns(df):
    """Coerce columns marked as 'numeric' in parameters.csv to numeric dtype (in-place)."""
    numeric_param_names = set(
        _PARAMETERS_DF.loc[_PARAMETERS_DF["data_type"] == "numeric", "parameter_name"]
    )
    numeric_cols = [c for c in df.columns if c in numeric_param_names]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


_KEEP_UPPER = {"LLC", "GPM", "INC", "CA", "DBA", "NA", "N/A"}


def _smart_title(s):
    """Title-case that preserves apostrophe contractions (Hauler's) and common abbreviations (LLC, GPM)."""
    result = []
    for word in s.split():
        upper = word.upper().strip(".,;:()")
        if upper in _KEEP_UPPER:
            result.append(word.upper())
        else:
            # Use .title() then fix apostrophe/quote mid-word caps: "Hauler'S" -> "Hauler's"
            titled = word.title()
            titled = re.sub(
                r"(['\u2019])([A-Z])", lambda m: m.group(1) + m.group(2).lower(), titled
            )
            result.append(titled)
    return " ".join(result)


def extract_value_from_line(
    line, item_order=None, ignore_before=None, ignore_after=None
):
    """Extract value from a line using item_order, ignore_before, and ignore_after."""
    if not isinstance(line, str):
        line = str(line)

    if item_order is None and not ignore_before and not ignore_after:
        return line.strip()

    # Process ignore_before first (remove everything before and including the phrase)
    if ignore_before and ignore_before != "NA":
        if ignore_before == "str":
            # If ignore_before is 'str', extract first numeric sequence
            match = re.search(r"([-+]?\d*\.?\d+)", line.strip())
            if match:
                line = match.group(1)
        elif ignore_before == "num":
            # If ignore_before is 'num', extract first number
            match = re.search(r"([-+]?\d*\.?\d+)", line.strip())
            if match:
                line = match.group(1)
        else:
            # Find the ignore_before phrase and take everything after it
            idx = line.lower().find(str(ignore_before).lower())
            if idx != -1:
                line = line[idx + len(str(ignore_before)) :].strip()

    # Then process ignore_after (remove everything after and including the marker)
    # ignore_after can be a string, or a list of markers (trim at whichever appears first)
    if ignore_after and ignore_after != "NA":
        if ignore_after == "str":
            # If ignore_after is 'str', trim at first non-numeric character after a numeric sequence
            match = re.match(r"([-+]?\d*\.?\d+)", line.strip())
            if match:
                line = match.group(1)
        else:  # any other search term
            markers = ignore_after if isinstance(ignore_after, list) else [ignore_after]
            idx = len(line)
            line_lower = line.lower()
            for m in markers:
                if not m:
                    continue
                i = line_lower.find(str(m).lower())
                if i != -1 and i < idx:
                    idx = i
            if idx < len(line):
                # trim line at the earliest ignore_after marker
                line = line[:idx].strip()

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

    if data_type == "numeric":
        # Try to coerce to float, but if that fails, keep the original text
        value = str(value).replace(",", "")
        try:
            return float(value)
        except ValueError:
            return value
    return value


def clean_common_errors(text):
    """Clean up common OCR errors in text while preserving structure."""

    # Case-insensitive replacements (pattern -> replacement)
    # These will match regardless of case
    case_insensitive_replacements = {
        "6pm": "GPM",
        "galions": "Gallons",
        "galons": "Gallons",
        "gailons": "Gallons",
        "pek load": "per load",
        " pek ": " per ",
        "jons": "tons",
        "waste water": "Wastewater",
        "havler": "Hauler",
        "hayler": "Hauler",
        "haulers calcs": "Hauler's Calculations",
        "haulers calculations": "Hauler's Calculations",
        "hauler's calculations": "Hauler's Calculations",
        "Arriount": "Amount",
        " tous": " tons",
        " tong": "tons",
    }

    # Case-sensitive replacements
    case_sensitive_replacements = {
        # odd characters
        "|": "I",
        "0O": "O",
        "1I": "I",
        "S5": "S",
        "Ibs": "lbs",
        "/bs": "lbs",
        "©": "",
        "; ": "",  # remove semicolons which sometimes appear leading a word
        # misspellings
        "Maxiumu": "Maximum",
        "FaciIity": "Facility",
        "CattIe": "Cattle",
        "KjeIdahl": "Kjeldahl",
        "Sroker": "Broker",
        " I Broker": "Broker",
        " I Composting Facility ": "Composting Facility",
        "HauIing": "Hauling",
        "Solide": "Solids",
        " Doing": " Dairy",
        "Daing": "Dairy",
        "Dainy": "Dairy",
        "Daire": "Dairy",
        "Cubie": "Cubic",
        "[]": "",  # for false checkmarks in llmwhisperer
        "[X]": "",  # for false checkmarks in llmwhisperer
        "> ": "",  # OCR artifact at beginning of lines
        "[ ] ": "",  # OCR artifact at beginning of lines
    }

    # Apply case-insensitive replacements
    for old, new in case_insensitive_replacements.items():
        # Use regex with re.IGNORECASE for case-insensitive replacement
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    # Apply case-sensitive replacements
    for old, new in case_sensitive_replacements.items():
        text = text.replace(old, new)

    # Remove certain characters
    for char in ["|", ",", "=", ":", "___"]:
        text = text.replace(char, "")

    # Remove empty lines
    text = "\n".join([line for line in text.split("\n") if line.strip()])

    # Fix number-letter confusions
    text = re.sub(r"(\d)O(\d)", r"\1O\2", text)
    text = re.sub(r"(\d)l(\d)", r"\1l\2", text)
    text = re.sub(r"(\d)I(\d)", r"\1I\2", text)
    text = re.sub(r"([a-zA-Z])0([a-zA-Z])", r"\1O\2", text)
    text = re.sub(r"([a-zA-Z])I([a-zA-Z])", r"\1I\2", text)

    # remove spaces before and after "/" and "-"
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)  # helpful for parcel numbers

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


def extract_parameters_from_text(
    text, template, param_locations_df, data_types, defaults
):
    """Extract all parameters for a template. Returns dict with parameter_key as keys."""
    result = {}
    template_params = param_locations_df[param_locations_df["template"] == template]
    for _, row in template_params.iterrows():
        param_key = row["parameter_key"]
        value = find_parameter_value(text, row, data_types, defaults)
        result[param_key] = value
    return result


def find_parameter_value(ocr_text, row, data_types, defaults):
    """Extract a parameter value from OCR text based on one row from parameter_locations."""
    param_key = row["parameter_key"]
    data_type = data_types.get(param_key, "text")

    def default():
        return get_default_value(param_key, data_types, defaults)

    if not ocr_text:
        return default()

    # Reduce search area using page_search_text (case-insensitive)
    page_search_text = row.get("page_search_text", pd.NA)
    search_text = ocr_text
    if not pd.isna(page_search_text) and page_search_text != "NA":
        pos = ocr_text.lower().find(str(page_search_text).lower())
        if pos == -1:
            return default()
        # return lines after the page_search_text
        search_text = ocr_text[pos + len(str(page_search_text)) :]

    row_search_text = row["row_search_text"]
    direction = str(row.get("search_direction", "")).lower()
    item_order = row.get("item_order", pd.NA)
    ignore_before = row.get("ignore_before", None)
    ignore_after = row.get("ignore_after", None)

    # Support pipe-delimited list: "Gallons|(If Amount Reported" -> list
    if isinstance(ignore_after, str) and "|" in ignore_after:
        ignore_after = [s.strip() for s in ignore_after.split("|") if s.strip()]

    if pd.isna(row_search_text) or not str(row_search_text).strip() or not direction:
        return default()

    row_search_text = str(row_search_text)
    search_lower = row_search_text.lower()

    # Parse lines
    lines = [ln.strip() for ln in search_text.split("\n")]
    non_empty = [ln for ln in lines if ln]

    # Helper: find index of first line containing search text
    def find_line_idx(line_list):
        return next(
            (i for i, ln in enumerate(line_list) if search_lower in ln.lower()), None
        )

    # Helper: get next non-empty line after index
    def next_non_empty(start_idx):
        for j in range(start_idx + 1, len(lines)):
            # skip lines with only spaces or tabs
            if lines[j].strip():
                return j, lines[j]
        return None, None

    phrase_idx = find_line_idx(non_empty)
    if phrase_idx is None:
        return default()

    line = non_empty[phrase_idx]
    actual_idx = find_line_idx(lines)
    extracted_text = None

    if direction == "right":
        # ignore_before = row_search_text  # Take everything after search phrase
        # first reduce to text to the right of search phrase on same line
        idx = line.lower().find(str(row_search_text).lower())
        if idx != -1:
            line = line[idx + len(str(row_search_text)) :].strip()
        extracted_text = extract_value_from_line(
            line, item_order, ignore_before, ignore_after
        )

    elif direction == "above":
        if phrase_idx > 0:
            extracted_text = extract_value_from_line(
                non_empty[phrase_idx - 1], item_order, ignore_before, ignore_after
            )

    elif direction == "below":
        if actual_idx is not None:
            next_idx, next_line = next_non_empty(actual_idx)
            if next_line:
                # Check for empty field marker
                if ignore_before and ignore_before != "NA":
                    if next_line.lower().startswith(str(ignore_before).lower()):
                        return default()
                extracted_text = extract_value_from_line(
                    next_line, item_order, ignore_before, ignore_after
                )

    elif direction == "right_below":
        # if "PO" in
        # Section headers that indicate a new field (not spillover)
        section_starts = [
            "enter the amount",
            "process wastewater",
            "written agreement",
            "method used",
        ]

        # Get text to the right of search phrase
        pos = line.lower().find(search_lower)
        right_text = line[pos + len(row_search_text) :].strip() if pos != -1 else ""
        # if "PO" in right_text:
        #     print(f" PO in line {line}")
        if right_text:
            # Apply ignore_before/ignore_after to right_text
            right_text_cleaned = extract_value_from_line(
                right_text, item_order, ignore_before, ignore_after
            )
            # Check for spillover to next line
            if actual_idx is not None:
                _, next_line = next_non_empty(actual_idx)
                if next_line and not any(
                    next_line.lower().startswith(s) for s in section_starts
                ):
                    next_line_text = extract_value_from_line(
                        next_line, item_order, ignore_before, ignore_after
                    )
                    extracted_text = f"{right_text_cleaned} {next_line_text}"
                else:
                    # Next line is a section header or doesn't exist, just use right_text
                    extracted_text = right_text_cleaned
            else:
                extracted_text = right_text_cleaned
        else:
            # Nothing on same line, look below
            if actual_idx is not None:
                _, next_line = next_non_empty(actual_idx)
                extracted_text = next_line
                extracted_text = extract_value_from_line(
                    extracted_text, item_order, ignore_before, ignore_after
                )

    # Convert and apply item_order
    value = None
    if extracted_text and extracted_text.strip():
        if pd.isna(item_order) or item_order == -1:
            value = convert_to_numeric(extracted_text, data_type)
        else:
            try:
                parts = str(extracted_text).split()
                k = int(item_order)
                if 0 <= k < len(parts):
                    value = convert_to_numeric(parts[k], data_type)
            except (TypeError, ValueError):
                value = convert_to_numeric(extracted_text, data_type)

    # Return default if no valid value
    if value is None or (data_type == "numeric" and (pd.isna(value) or value == 0)):
        return default()

    if value == "N/A" or value == "NA" or value == ".":
        return default()

    return _smart_title(value) if isinstance(value, str) else value
