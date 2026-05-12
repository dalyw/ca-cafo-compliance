#!/usr/bin/env python3
import calendar
import os
import re
import glob
import numpy as np
import pandas as pd
import pymupdf as fitz
from dateutil import parser as date_parser
from collections import defaultdict

from helpers import (
    PATH_TO_PDF_DATA,
    build_parameter_dicts,
    coerce_columns,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
YEAR, REGION = "2024", "R5"

LOCATIONS_DF = pd.read_csv(os.path.join(DATA_DIR, "parameter_locations.csv"))
TEMPLATES_DF = pd.read_csv(os.path.join(DATA_DIR, "templates.csv"))
param_dicts = build_parameter_dicts(manifest_only=True)
PARAM_TO_COL = param_dicts["key_to_name"]
PARAM_TYPES = param_dicts["key_to_type"]

# Regex patterns
NUM = r"(\d+(?:\.\d+)?)"
LOAD = (
    "(?:"
    + "|".join(
        f"{t}s?"
        for t in ["load", "loap", "haul", "truckload", "dump\\s*t(?:k|ruck)", "tanker\\s*load"]
    )
    + ")"
)
UNIT = "(" + "|".join(f"{t}s?" for t in ["ton", "gallon", "gal", "yard"]) + ")"
SEP = r"\s*(?:[x×@\-]|at)\s*"
APPROX = r"(?:approx\.?\s*)?"

LOAD_PATTERNS = [
    (re.compile(rf"{NUM}\s*{LOAD}(?:\s+\w+)*?{SEP}{APPROX}{NUM}\s*{UNIT}", re.I), (0, 1, 2)),
    (re.compile(rf"{NUM}\s*{UNIT}{SEP}{NUM}\s*{LOAD}", re.I), (2, 0, 1)),
    (re.compile(rf"{NUM}\s*{LOAD}\s+{NUM}\s*{UNIT}\s+each\b", re.I), (0, 1, 2)),
]

HOURS_RE = re.compile(r"(\d+(?:\s*\d+/\d+)?(?:\.\d+)?)\s*(?:hours?|hrs?)\b", re.I)
GPM_RE = re.compile(r"(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:gpm|gallons per min)\b", re.I)
FRAC_RE = re.compile(r"(\d+)(\d)/(\d+)")
TABLE_ROW_RE = re.compile(r"^(.+?)\s+([\d,]+)\s+(tons?|gallons?|gals?|yards?)\s+(\d+)\s*%", re.I)
PHONE_RE = re.compile(r"\(?\d{3}\)?[\s\-\.]?\d{3,4}[\s\-\.]?\d{4}")

MONTHS = "|".join(calendar.month_name[1:] + calendar.month_abbr[1:]).lower()
DATE_TOKEN_RE = re.compile(
    rf"\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|(?:{MONTHS})(?:\s+\d{{1,2}})?(?:\s*,?\s*\d{{2,4}})?", re.I
)
MONTH_ONLY_RE = re.compile(rf"^\s*({MONTHS})\s*$", re.I)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

PARCEL_RE = re.compile(
    r"(?:\(?\d*\)?\s*[Xx]?\s*)?([\dXx]{2,}\s*[.\-]\s*[\dXx]{2,}(?:\s*[.\-]\s*[\dXx]+)*)",
    re.IGNORECASE,
)

KEEP_UPPER = {"LLC", "GPM", "INC", "CA", "DBA", "NA", "N/A"}

CI_ERRORS = {
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
CS_ERRORS = {
    "|": "I",
    "0O": "O",
    "1I": "I",
    "S5": "S",
    "Ibs": "lbs",
    "/bs": "lbs",
    "©": "",
    "; ": "",
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
    "[]": "",
    "[X]": "",
    "> ": "",
    "[ ] ": "",
}
STRIP_CHARS = ["|", ",", "=", ":", "___"]


def pdf_stem_from_txt_path(txt_path):
    parts = os.path.normpath(txt_path).split(os.sep)
    for folder in ("llmwhisperer_output", "tesseract_output"):
        if folder in parts:
            return parts[parts.index(folder) + 1]


def clean_common_errors(text):
    """Clean up common OCR errors while preserving structure."""
    for old, new in CI_ERRORS.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    for old, new in CS_ERRORS.items():
        text = text.replace(old, new)
    for char in STRIP_CHARS:
        text = text.replace(char, "")
    text = re.sub(r"(\d)O(\d)", r"\1O\2", text)
    text = re.sub(r"(\d)[lI](\d)", r"\1l\2", text)
    text = re.sub(r"([a-zA-Z])[0O]([a-zA-Z])", r"\1O\2", text)
    text = re.sub(r"([a-zA-Z])I([a-zA-Z])", r"\1I\2", text)
    text = re.sub(r"\s*[/-]\s*", lambda m: m.group(0).strip(), text)
    text = re.sub(r"_+", "", text)
    text = re.sub(r"[ \t\f\v\r]+", " ", text)
    text = "\n".join([re.sub(r"^\s*;\s*", "", line) for line in text.split("\n") if line.strip()])
    return text


def identify_manifest_pages(result_text):
    """Identify manifest page ranges and templates from OCR text."""
    matches = list(re.compile(r"=== Page (\d+) ===").finditer(result_text))
    if not matches:
        return []

    pages = {
        int(m.group(1)): result_text[
            m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(result_text)
        ].strip()
        for i, m in enumerate(matches)
    }

    used, manifests = set(), []
    sorted_pages = sorted(pages)

    for idx, p1 in enumerate(sorted_pages):
        t1_upper = pages[p1].upper()
        if (
            p1 in used
            or "REQUIRED ATTACHMENTS" in t1_upper
            or not (
                "MANIFEST" in t1_upper
                and any(k in t1_upper for k in ["TRACKING", "ATTACHMENT"])
                and any(
                    t in t1_upper
                    for t in [
                        "INSTRUCTIONS",
                        "COMPLETE ONE",
                        "WASTE GENERATOR INFORMATION",
                        "ADDRESS OF HAULING",
                    ]
                )
            )
        ):
            continue

        used.add(p1)
        combined, end_pg = pages[p1], p1

        # Add page 2 if applicable
        if (
            idx + 1 < len(sorted_pages)
            and "CERTIFICATION" not in t1_upper
            and "SIGNATURE OF HAULER" not in t1_upper
        ):
            cand = sorted_pages[idx + 1]
            t2_upper = pages[cand].upper()
            if cand not in used and not (
                "REQUIRED ATTACHMENTS" not in t2_upper
                and "MANIFEST" in t2_upper
                and any(t in t2_upper for t in ["INSTRUCTIONS", "COMPLETE ONE"])
            ):
                used.add(cand)
                combined += "\n\n" + pages[cand]
                end_pg = cand

        # Add page 3 if "Page 2 of 3"
        if "PAGE 2 OF 3" in combined.upper() and (p3 := end_pg + 1) in pages and p3 not in used:
            used.add(p3)
            combined += "\n" + pages[p3]
            end_pg = p3

        cleaned = clean_common_errors(combined)
        text_upper = cleaned.upper()
        template = next(
            (
                row["template_key"]
                for _, row in TEMPLATES_DF.iterrows()
                if pd.notna(kw := row["keywords"])
                and all(
                    any(t.strip() in text_upper for t in c.split("|"))
                    for c in str(kw).upper().split("&&")
                )
                and row["page_count"] == (end_pg - p1 + 1)
            ),
            "R5-2007-0035_general_order",
        )

        manifests.append((cleaned, (p1, end_pg), template))

    return manifests


def _parse_hauling_table(manifest_text):
    """Extract hauling event rows from table."""
    lines = manifest_text.split("\n")
    start_idx = next(
        (i + 1 for i, ln in enumerate(lines) if "date" in ln.lower() and "haul" in ln.lower()), None
    )
    if start_idx is None:
        return []

    rows = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("total"):
            break
        if not (m := TABLE_ROW_RE.match(stripped)):
            continue

        date_range, amount, units, moisture = m.groups()
        amount_key = (
            "manure_amount"
            if "ton" in units.lower() or "yard" in units.lower()
            else "wastewater_amount"
        )
        rows.append(
            {
                PARAM_TO_COL["haul_date"]: date_range.strip(),
                PARAM_TO_COL[amount_key]: amount.replace(",", ""),
                PARAM_TO_COL["manure_moisture_percent"]: moisture,
            }
        )
    return rows


def _split_haul_dates(data):
    """Parse haul_date into first and last dates."""
    haul_date = data.get(PARAM_TO_COL["haul_date"])
    if not haul_date or not isinstance(haul_date, str):
        return

    date_parts = DATE_TOKEN_RE.findall(haul_date) or [
        p.strip() for p in re.split(r"[-–—]|\bto\b|,|;|&", haul_date, flags=re.I) if p.strip()
    ]

    parsed = []
    for part in date_parts:
        try:
            dt = date_parser.parse(part, fuzzy=True, dayfirst=False)
            if not YEAR_RE.search(part):
                dt = dt.replace(year=int(YEAR))
            if MONTH_ONLY_RE.match(part):
                dt = dt.replace(day=31 if dt.month == 12 else 1)
            parsed.append(dt)
        except (ValueError, TypeError):
            continue

    if parsed:
        parsed.sort()
        fmt = lambda d: f"{d.month}/{d.day}/{d.year}"
        data[PARAM_TO_COL["haul_date_first"]] = fmt(parsed[0]) if len(parsed) > 1 else None
        data[PARAM_TO_COL["haul_date_last"]] = fmt(parsed[-1])


def strip_phone_number(text):
    if not isinstance(text, str) or not text.strip():
        return text
    matches = list(PHONE_RE.finditer(text))
    if not matches:
        return text
    m = matches[-1]
    before = " ".join(filter(None, [text[: m.start()].strip(), text[m.end() :].strip()])).strip()
    return before or None


def parse_destination_address_and_parcel(value):
    if not isinstance(value, str) or not (s := value.strip()):
        return None, None

    if (
        len(s) >= 3
        and ("-" in s or "." in s)
        and re.fullmatch(r"[\d\s.\-]+", s)
        and re.search(r"\d+\s*[.\-]\s*\d+", s)
    ):
        return None, re.sub(r"\s+", "", s)

    matches = list(PARCEL_RE.finditer(s))
    if not matches:
        return s, None

    m = matches[-1]
    rest = " ".join(filter(None, [s[: m.start()].strip(), s[m.end() :].strip()])).strip()
    parcel = re.sub(r"\s+", "", m.group(1))
    address = rest if (rest and len(rest) >= 5 and re.search(r"[A-Za-z]", rest)) else None
    return address, parcel


def normalize_apn(parcel_number):
    if not isinstance(parcel_number, str) or not parcel_number.strip():
        return None
    s = re.match(r"[\d\s.\-Xx]+", parcel_number.strip())
    if not s:
        return None
    s = re.sub(r"\s+", "", s.group()).replace(".", "-")
    s = re.sub(r"[Xx]", "0", s)
    return s if re.fullmatch(r"[\d\-]+", s) else None


def split_apn_county(parcel_text):
    if not isinstance(parcel_text, str) or not parcel_text.strip():
        return None, None
    m = re.match(r"([\d\s.\-Xx]+)\s+([A-Za-z].*)$", parcel_text.strip())
    return (
        (normalize_apn(m.group(1)), m.group(2).strip()) if m else (normalize_apn(parcel_text), None)
    )


def _smart_title(s):
    """Title-case preserving apostrophes and common abbreviations."""
    result = []
    for word in s.split():
        if word.upper().strip(".,;:()") in KEEP_UPPER:
            result.append(word.upper())
        else:
            titled = word.title()
            titled = re.sub(
                r"(['\u2019])([A-Z])", lambda m: m.group(1) + m.group(2).lower(), titled
            )
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
                line = line[idx + len(str(ignore_before)) :].strip()

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


def find_parameter_value(ocr_text, row, data_types):
    """Extract parameter value from OCR text based on parameter_locations row."""
    param_key = row["parameter_key"]
    data_type = data_types.get(param_key, "text")
    default = np.nan if data_type == "numeric" else None

    if not ocr_text:
        return default

    # Reduce search area using page_search_text
    search_text = ocr_text
    if pd.notna(page_search := row.get("page_search_text")) and page_search != "NA":
        if (pos := ocr_text.lower().find(str(page_search).lower())) == -1:
            return default
        search_text = ocr_text[pos + len(str(page_search)) :]

    row_search_text = row["row_search_text"]
    direction = str(row.get("search_direction", "")).lower()
    item_order = row.get("item_order", pd.NA)
    ignore_before = row.get("ignore_before")
    ignore_after = row.get("ignore_after")

    if isinstance(ignore_after, str) and "|" in ignore_after:
        ignore_after = [s.strip() for s in ignore_after.split("|") if s.strip()]

    if pd.isna(row_search_text) or not str(row_search_text).strip() or not direction:
        return default

    search_lower = str(row_search_text).lower()
    lines = [ln.strip() for ln in search_text.split("\n")]
    non_empty = [ln for ln in lines if ln]

    find_line_idx = lambda line_list: next(
        (i for i, ln in enumerate(line_list) if search_lower in ln.lower()), None
    )

    def next_non_empty(start_idx):
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip():
                return j, lines[j]
        return None, None

    if (phrase_idx := find_line_idx(non_empty)) is None:
        return default

    line = non_empty[phrase_idx]
    actual_idx = find_line_idx(lines)
    extracted_text = None

    if direction == "right":
        idx = line.lower().find(search_lower)
        if idx != -1:
            line = line[idx + len(str(row_search_text)) :].strip()
        extracted_text = extract_value_from_line(line, item_order, ignore_before, ignore_after)

    elif direction == "above":
        if phrase_idx > 0:
            extracted_text = extract_value_from_line(
                non_empty[phrase_idx - 1], item_order, ignore_before, ignore_after
            )

    elif direction == "below":
        if actual_idx is not None:
            next_idx, next_line = next_non_empty(actual_idx)
            if next_line:
                if (
                    ignore_before
                    and ignore_before != "NA"
                    and next_line.lower().startswith(str(ignore_before).lower())
                ):
                    return default
                extracted_text = extract_value_from_line(
                    next_line, item_order, ignore_before, ignore_after
                )

    elif direction == "right_below":
        # Take text right of label; also append next line unless it starts a new section
        section_starts = ["enter the amount", "process wastewater", "written agreement", "method used"]
        pos = line.lower().find(search_lower)
        right_text = line[pos + len(str(row_search_text)) :].strip() if pos != -1 else ""
        extracted_text = (
            extract_value_from_line(right_text, item_order, ignore_before, ignore_after)
            if right_text
            else None
        )
        if actual_idx is not None:
            _, next_line = next_non_empty(actual_idx)
            if next_line and not any(next_line.lower().startswith(s) for s in section_starts):
                next_text = extract_value_from_line(next_line, item_order, ignore_before, ignore_after)
                extracted_text = f"{extracted_text} {next_text}" if extracted_text else next_text

    value = None
    if extracted_text and extracted_text.strip():
        value = convert_to_numeric(extracted_text, data_type)

    # Return default if no valid value
    if (
        value is None
        or (data_type == "numeric" and (pd.isna(value) or value == 0))
        or value in ("N/A", "NA", ".")
    ):
        return default

    return _smart_title(value) if isinstance(value, str) else value


def extract_manifest_fields(manifest_text, template):
    """Extract all manifest fields from text."""
    extracted = {
        row["parameter_key"]: find_parameter_value(manifest_text, row, PARAM_TYPES)
        for _, row in LOCATIONS_DF[LOCATIONS_DF["template"] == template].iterrows()
    }
    data = {PARAM_TO_COL[k]: v for k, v in extracted.items()}
    for k in PARAM_TO_COL:
        if PARAM_TO_COL[k] not in data:
            data[PARAM_TO_COL[k]] = None
    data["Parameter Template"] = template

    # Post-process specific fields
    for param_key, column_name in PARAM_TO_COL.items():
        value = data.get(column_name)
        if value is None:
            continue

        if param_key == "destination_address":
            address_part, parcel_part = parse_destination_address_and_parcel(value)
            if parcel_part and not data.get(PARAM_TO_COL["destination_parcel_number"]):
                data[PARAM_TO_COL["destination_parcel_number"]] = parcel_part
            data[column_name] = address_part or (value if not parcel_part else None)

        elif param_key == "destination_contact_address":
            value = strip_phone_number(value)
            if value and (m := re.search(r"\b\d{2,}", value)):
                value = value[m.start() :].strip()
            data[column_name] = value

        elif param_key == "destination_type":
            vl = str(value).lower()
            if "(as identified" in vl or "above)" in vl:
                data[column_name] = None

    # Parse number of hauls and volume-per-haul out of the free-text method field
    for waste_type, units in [("manure", ["ton", "yard"]), ("wastewater", ["gallon"])]:
        if not (txt := data.get(f"Method Used to Determine Volume of {waste_type.title()}")):
            continue
        txt_str = str(txt)

        # li/ai/ui are group indices for loads, amount, unit within each pattern
        for regex, (li, ai, ui) in LOAD_PATTERNS:
            if m := regex.search(txt_str):
                groups = m.groups()
                data[PARAM_TO_COL[f"{waste_type}_number_hauls"]] = groups[li]
                unit_text = groups[ui].lower()
                for u in units:
                    if u in unit_text:
                        data[PARAM_TO_COL[f"{waste_type}_{u}_per_haul"]] = groups[ai]
                break

        # Wastewater hours and GPM
        if waste_type == "wastewater":
            if m := HOURS_RE.search(txt_str):
                s = m.group(1).strip().replace(" ", "")
                if "/" not in s:
                    hours = s
                elif m := FRAC_RE.match(s):
                    hours = float(m.group(1)) + float(m.group(2)) / float(m.group(3))
                elif m := re.match(r"(\d+)/(\d+)", s):
                    hours = float(m.group(1)) / float(m.group(2))
                data[PARAM_TO_COL["wastewater_hours_pumped"]] = hours
            if m := GPM_RE.search(txt_str):
                data[PARAM_TO_COL["wastewater_pumping_rate"]] = m.group(1)

    # Move solids to moisture if text says "moisture"
    solids_col, moisture_col = (
        PARAM_TO_COL["manure_solids_percent"],
        PARAM_TO_COL["manure_moisture_percent"],
    )
    if data.get(solids_col) and not data.get(moisture_col):
        if re.search(rf"{re.escape(str(data[solids_col]))}\s*%\s*moisture", manifest_text, re.I):
            data[moisture_col] = data.pop(solids_col)
            data[solids_col] = None

    # is_pipeline scans all text; is_trucked infers from the method field but pipeline always wins
    is_pipeline = "pipeline" in manifest_text.lower()
    data[PARAM_TO_COL["is_pipeline"]] = is_pipeline
    ml = (data.get("Method Used to Determine Volume of Wastewater") or "").lower()
    if "pipeline" in ml or is_pipeline:
        data[PARAM_TO_COL["is_trucked"]] = False
    elif any(t in ml for t in ["load", "haul", "tank", "hauler"]) and not any(
        t in ml for t in ["apply", "applied"]
    ):
        data[PARAM_TO_COL["is_trucked"]] = True
    else:
        data[PARAM_TO_COL["is_trucked"]] = "UNSURE"

    return data


def extract_manifests_from_txt(txt_path):
    """Extract all manifests from a single OCR text file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        result_text = f.read()

    manifest_pages = identify_manifest_pages(result_text)
    if not manifest_pages:
        return []

    output_dir = os.path.dirname(txt_path)
    parts = os.path.normpath(txt_path).split(os.sep)
    idx = parts.index("Manure Trucking Network Analysis")
    data, region, county, template = parts[idx + 1 : idx + 5]
    pdf_stem = parts[-2]
    original_pdf = os.path.join(
        PATH_TO_PDF_DATA, region, county, template, "original", f"{pdf_stem}.pdf"
    )

    manifests, all_manifests_doc = [], fitz.open()  # all_manifests_doc collects pages from every manifest in this PDF

    with fitz.open(original_pdf) as doc:
        for i, (manifest_text, (start_pg, end_pg), manifest_template) in enumerate(
            manifest_pages, start=1
        ):
            data = extract_manifest_fields(manifest_text, manifest_template)
            metadata = {"Source PDF": pdf_stem, "Start Page": start_pg, "End Page": end_pg}

            # Multi-row table or single manifest
            table_rows = (
                _parse_hauling_table(manifest_text)
                if manifest_template == "R5-2007-0035_one_page_2"
                else None
            )
            if table_rows:
                entries = [
                    {**data, **row, **metadata, "Manifest Number": f"{i}{chr(97 + j)}"}
                    for j, row in enumerate(table_rows)
                ]
            else:
                entries = [{**data, **metadata, "Manifest Number": i}]

            for entry in entries:
                _split_haul_dates(entry)
                manifests.append(entry)

            # Save individual manifest files
            with open(os.path.join(output_dir, f"manifest_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(manifest_text)
            with fitz.open() as new_doc:
                for p in range(start_pg - 1, end_pg):
                    if 0 <= p < len(doc):
                        new_doc.insert_pdf(doc, from_page=p, to_page=p)
                        all_manifests_doc.insert_pdf(doc, from_page=p, to_page=p)
                new_doc.save(os.path.join(output_dir, f"manifest_{i}.pdf"))

    all_manifests_doc.save(os.path.join(output_dir, "all_manifests.pdf"))
    all_manifests_doc.close()

    print(f"{len(manifest_pages)} manifests of {template} in {pdf_stem}")
    return manifests


def main():
    """Extract all manifests and save to CSV."""
    stems = {}
    for ocr_method in ["llmwhisperer", "tesseract"]:
        files = [
            p
            for p in glob.glob(
                f"{PATH_TO_PDF_DATA}/{REGION}/**/{ocr_method}_output/**/*.txt", recursive=True
            )
            if not os.path.basename(p).startswith("manifest_")
        ]
        out = {}
        for p in sorted(files):
            if (stem := pdf_stem_from_txt_path(p)) in out:
                raise ValueError(
                    f"Duplicate txt for {stem} in {ocr_method}_output: {out[stem]} and {p}"
                )
            out[stem] = p
        stems[ocr_method] = out

    all_manifests = []
    for stem in sorted(set(stems["tesseract"]) | set(stems["llmwhisperer"])):
        chosen = stems["llmwhisperer"].get(stem) or stems["tesseract"].get(stem)
        all_manifests.extend(extract_manifests_from_txt(chosen))

    df = pd.DataFrame(all_manifests)
    print(df.head())

    out_csv = "output_data/all_manifests_as_written_automatic.csv"
    coerce_columns(df)

    # Clean contact addresses and parcel numbers
    contact_col = PARAM_TO_COL["destination_contact_address"]
    df[contact_col] = df[contact_col].apply(
        lambda x: strip_phone_number(x) if isinstance(x, str) else x
    )

    parcel_col, county_col = (
        PARAM_TO_COL["destination_parcel_number"],
        PARAM_TO_COL["destination_county"],
    )
    df[[parcel_col, county_col]] = pd.DataFrame(
        df[parcel_col]
        .map(lambda x: split_apn_county(x) if isinstance(x, str) else (None, None))
        .tolist(),
        index=df.index,
    )

    # Categorize manifest type
    manure_col, wastewater_col = PARAM_TO_COL["manure_amount"], PARAM_TO_COL["wastewater_amount"]
    has_manure, has_wastewater = df[manure_col].notna(), df[wastewater_col].notna()
    df["Manifest Type"] = "unknown"
    df.loc[has_manure & has_wastewater, "Manifest Type"] = "both"
    df.loc[has_manure & ~has_wastewater, "Manifest Type"] = "manure"
    df.loc[~has_manure & has_wastewater, "Manifest Type"] = "wastewater"

    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")


if __name__ == "__main__":
    main()
