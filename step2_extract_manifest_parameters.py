#!/usr/bin/env python3
import calendar
import os
import re
import glob
import pandas as pd
import pymupdf as fitz
from dateutil import parser as date_parser
from collections import defaultdict

from helpers_pdf_metrics import (
    GDRIVE_BASE, build_parameter_dicts, clean_common_errors, coerce_columns,
    extract_parameters_from_text,
)
from helpers_geocoding import (
    normalize_apn
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
YEAR, REGION = "2024", "R5"

LOCATIONS_DF = pd.read_csv(os.path.join(DATA_DIR, "parameter_locations.csv"))
TEMPLATES_DF = pd.read_csv(os.path.join(DATA_DIR, "templates.csv"))
_manifest_dicts = build_parameter_dicts(manifest_only=True)
PARAM_TO_COL = _manifest_dicts["key_to_name"]
PARAM_TYPES = _manifest_dicts["key_to_type"]
PARAM_DEFAULTS = _manifest_dicts["key_to_default"]

# Regex patterns
_NUM = r"(\d+(?:\.\d+)?)"
_LOAD = "(?:" + "|".join(f"{t}s?" for t in ["load", "loap", "haul", "truckload", "dump\\s*t(?:k|ruck)", "tanker\\s*load"]) + ")"
_UNIT = "(" + "|".join(f"{t}s?" for t in ["ton", "gallon", "gal", "yard"]) + ")"
_SEP = r"\s*(?:[x×@\-]|at)\s*"
_APPROX = r"(?:approx\.?\s*)?"

_LOAD_PATTERNS = [
    (re.compile(rf"{_NUM}\s*{_LOAD}(?:\s+\w+)*?{_SEP}{_APPROX}{_NUM}\s*{_UNIT}", re.I), (0, 1, 2)),
    (re.compile(rf"{_NUM}\s*{_UNIT}{_SEP}{_NUM}\s*{_LOAD}", re.I), (2, 0, 1)),
    (re.compile(rf"{_NUM}\s*{_LOAD}\s+{_NUM}\s*{_UNIT}\s+each\b", re.I), (0, 1, 2)),
]

_HOURS_RE = re.compile(r"(\d+(?:\s*\d+/\d+)?(?:\.\d+)?)\s*(?:hours?|hrs?)\b", re.I)
_GPM_RE = re.compile(r"(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:gpm|gallons per min)\b", re.I)
_FRAC_RE = re.compile(r"(\d+)(\d)/(\d+)")
_TABLE_ROW_RE = re.compile(r"^(.+?)\s+([\d,]+)\s+(tons?|gallons?|gals?|yards?)\s+(\d+)\s*%", re.I)

_MONTHS = "|".join(calendar.month_name[1:] + calendar.month_abbr[1:]).lower()
_DATE_TOKEN_RE = re.compile(rf"\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|(?:{_MONTHS})(?:\s+\d{{1,2}})?(?:\s*,?\s*\d{{2,4}})?", re.I)
_MONTH_ONLY_RE = re.compile(rf"^\s*({_MONTHS})\s*$", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def pdf_stem_from_txt_path(txt_path):
    parts = os.path.normpath(txt_path).split(os.sep)
    for folder in ("llmwhisperer_output", "tesseract_output"):
        if folder in parts:
            return parts[parts.index(folder) + 1]


def identify_manifest_pages(result_text):
    """Identify manifest page ranges and templates from OCR text."""
    matches = list(re.compile(r"=== Page (\d+) ===").finditer(result_text))
    if not matches:
        return [], [], [], []

    pages = {
        int(m.group(1)): result_text[m.end():matches[i+1].start() if i+1 < len(matches) else len(result_text)].strip()
        for i, m in enumerate(matches)
    }

    used, manifest_num, nums, blocks, ranges, templates = set(), 0, [], [], [], []
    sorted_pages = sorted(pages)

    for idx, p1 in enumerate(sorted_pages):
        t1_upper = pages[p1].upper()
        if (p1 in used or "REQUIRED ATTACHMENTS" in t1_upper or 
            not ("MANIFEST" in t1_upper and 
                 any(k in t1_upper for k in ["TRACKING", "ATTACHMENT"]) and
                 any(t in t1_upper for t in ["INSTRUCTIONS", "COMPLETE ONE", "WASTE GENERATOR INFORMATION", "ADDRESS OF HAULING"]))):
            continue

        manifest_num += 1
        used.add(p1)
        combined, end_pg = pages[p1], p1

        # Add page 2 if applicable
        if (idx + 1 < len(sorted_pages) and "CERTIFICATION" not in t1_upper and "SIGNATURE OF HAULER" not in t1_upper):
            cand = sorted_pages[idx + 1]
            t2_upper = pages[cand].upper()
            if (cand not in used and not ("REQUIRED ATTACHMENTS" not in t2_upper and "MANIFEST" in t2_upper and 
                any(t in t2_upper for t in ["INSTRUCTIONS", "COMPLETE ONE"]))):
                used.add(cand)
                combined += "\n\n" + pages[cand]
                end_pg = cand

        # Add page 3 if "Page 2 of 3"
        if "PAGE 2 OF 3" in combined.upper() and (p3 := end_pg + 1) in pages and p3 not in used:
            used.add(p3)
            combined += "\n" + pages[p3]
            end_pg = p3

        # Match template
        text_upper = clean_common_errors(combined).upper()
        template = next(
            (row["template_key"] for _, row in TEMPLATES_DF.iterrows()
             if pd.notna(kw := row["keywords"]) and
             all(any(t.strip() in text_upper for t in c.split("|")) for c in str(kw).upper().split("&&")) and
             row["page_count"] == (end_pg - p1 + 1)),
            "R5-2007-0035_general_order"
        )

        nums.append(manifest_num)
        blocks.append(combined)
        ranges.append((p1, end_pg))
        templates.append(template)

    return nums, blocks, ranges, templates


def _parse_hauling_table(manifest_text):
    """Extract hauling event rows from table."""
    lines = manifest_text.split("\n")
    start_idx = next((i + 1 for i, ln in enumerate(lines) if "date" in ln.lower() and "haul" in ln.lower()), None)
    if start_idx is None:
        return []

    rows = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("total"):
            break
        if not (m := _TABLE_ROW_RE.match(stripped)):
            continue

        date_range, amount, units, moisture = m.groups()
        amount_key = "manure_amount" if "ton" in units.lower() or "yard" in units.lower() else "wastewater_amount"
        rows.append({
            PARAM_TO_COL["haul_date"]: date_range.strip(),
            PARAM_TO_COL[amount_key]: amount.replace(",", ""),
            PARAM_TO_COL["manure_moisture_percent"]: moisture,
        })
    return rows


def _split_haul_dates(data):
    """Parse haul_date into first and last dates."""
    haul_date = data.get(PARAM_TO_COL["haul_date"])
    if not haul_date or not isinstance(haul_date, str):
        return

    date_parts = _DATE_TOKEN_RE.findall(haul_date) or [
        p.strip() for p in re.split(r"[-–—]|\bto\b|,|;|&", haul_date, flags=re.I) if p.strip()
    ]

    parsed = []
    for part in date_parts:
        try:
            dt = date_parser.parse(part, fuzzy=True, dayfirst=False)
            if not _YEAR_RE.search(part):
                dt = dt.replace(year=int(YEAR))
            if _MONTH_ONLY_RE.match(part):
                dt = dt.replace(day=31 if dt.month == 12 else 1)
            parsed.append(dt)
        except (ValueError, TypeError):
            continue

    if parsed:
        parsed.sort()
        fmt = lambda d: f"{d.month}/{d.day}/{d.year}"
        data[PARAM_TO_COL["haul_date_first"]] = fmt(parsed[0]) if len(parsed) > 1 else None
        data[PARAM_TO_COL["haul_date_last"]] = fmt(parsed[-1])



def strip_trailing_pattern(text, regex):
    if not isinstance(text, str) or not text.strip():
        return text, None
    matches = list(regex.finditer(text))
    if not matches:
        return text, None
    m = matches[-1]
    before = " ".join(filter(None, [text[: m.start()].strip(), text[m.end() :].strip()])).strip()
    return (before or None), m.group(0)



def strip_phone_number(text):
    _PHONE_RE = re.compile(r"\(?\d{3}\)?[\s\-\.]?\d{3,4}[\s\-\.]?\d{4}")
    cleaned, _ = strip_trailing_pattern(text, _PHONE_RE)
    return cleaned


def looks_like_parcel_number(text):
    if not isinstance(text, str) or not (s := text.strip()) or len(s) < 3:
        return False
    return (
        (("-" in s) or ("." in s))
        and bool(re.fullmatch(r"[\d\s.\-]+", s))
        and bool(re.search(r"\d+\s*[.\-]\s*\d+", s))
    )


def parse_destination_address_and_parcel(value):
    if not isinstance(value, str) or not (s := value.strip()):
        return None, None

    if looks_like_parcel_number(s):
        return None, re.sub(r"\s+", "", s)

    _PARCEL_RE = re.compile(
        r"(?:\(?\d*\)?\s*[Xx]?\s*)?([\dXx]{2,}\s*[.\-]\s*[\dXx]{2,}(?:\s*[.\-]\s*[\dXx]+)*)",
        re.IGNORECASE,
    )

    matches = list(_PARCEL_RE.finditer(s))
    if not matches:
        return s, None

    m = matches[-1]
    rest = " ".join(filter(None, [s[: m.start()].strip(), s[m.end() :].strip()])).strip()
    parcel = re.sub(r"\s+", "", m.group(1))
    address = rest if (rest and len(rest) >= 5 and re.search(r"[A-Za-z]", rest)) else None
    return address, parcel


def split_apn_county(parcel_text):
    if not isinstance(parcel_text, str) or not parcel_text.strip():
        return None, None
    m = re.match(r"([\d\s.\-Xx]+)\s+([A-Za-z].*)$", parcel_text.strip())
    return (normalize_apn(m.group(1)), m.group(2).strip()) if m else (normalize_apn(parcel_text), None)


def extract_manifest_fields(manifest_text, template):
    """Extract all manifest fields from text."""
    extracted = extract_parameters_from_text(manifest_text, template, LOCATIONS_DF, PARAM_TYPES, PARAM_DEFAULTS)
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
                value = value[m.start():].strip()
            data[column_name] = value

        elif param_key == "destination_type":
            vl = str(value).lower()
            if "(as identified" in vl or "above)" in vl:
                data[column_name] = None

    # Extract loads/hauls from method text
    for wt, units in [("manure", ["ton", "yard"]), ("wastewater", ["gallon"])]:
        if not (txt := data.get(f"Method Used to Determine Volume of {wt.title()}")):
            continue
        txt_str = str(txt)

        for regex, (li, ai, ui) in _LOAD_PATTERNS:
            if m := regex.search(txt_str):
                g = m.groups()
                data[PARAM_TO_COL[f"{wt}_number_hauls"]] = g[li]
                unit_text = g[ui].lower()
                for u in units:
                    if u in unit_text:
                        data[PARAM_TO_COL[f"{wt}_{u}_per_haul"]] = g[ai]
                break

        # Wastewater hours and GPM
        if wt == "wastewater":
            if m := _HOURS_RE.search(txt_str):
                s = m.group(1).strip().replace(" ", "")
                if "/" not in s:
                    hours = s
                elif m := _FRAC_RE.match(s):
                    hours = float(m.group(1)) + float(m.group(2)) / float(m.group(3))
                elif m := re.match(r"(\d+)/(\d+)", s):
                    hours = float(m.group(1)) / float(m.group(2))
                data[PARAM_TO_COL["wastewater_hours_pumped"]] = hours
            if m := _GPM_RE.search(txt_str):
                data[PARAM_TO_COL["wastewater_pumping_rate"]] = m.group(1)

    # Move solids to moisture if text says "moisture"
    solids_col, moisture_col = PARAM_TO_COL["manure_solids_percent"], PARAM_TO_COL["manure_moisture_percent"]
    if data.get(solids_col) and not data.get(moisture_col):
        if re.search(rf"{re.escape(str(data[solids_col]))}\s*%\s*moisture", manifest_text, re.I):
            data[moisture_col] = data.pop(solids_col)
            data[solids_col] = None

    # Pipeline and trucked flags
    is_pipeline = "pipeline" in manifest_text.lower()
    data[PARAM_TO_COL["is_pipeline"]] = is_pipeline
    method_text = data.get(f"Method Used to Determine Volume of Wastewater", "") or ""
    if method_text:
        ml = method_text.lower()
        if "pipeline" in ml:
            data[PARAM_TO_COL["is_trucked"]] = False
        elif any(t in ml for t in ["load", "haul", "tank", "hauler"]) and not any(t in ml for t in ["apply", "applied"]):
            data[PARAM_TO_COL["is_trucked"]] = True
        else:
            data[PARAM_TO_COL["is_trucked"]] = "UNSURE"
    else:
        data[PARAM_TO_COL["is_trucked"]] = "UNSURE"
    # is_pipeline always takes precedence
    if is_pipeline:
        data[PARAM_TO_COL["is_trucked"]] = False

    return data


def extract_manifests_from_txt(txt_path):
    """Extract all manifests from a single OCR text file."""
    pdf_stem = pdf_stem_from_txt_path(txt_path)
    
    with open(txt_path, "r", encoding="utf-8") as f:
        result_text = f.read()

    nums, blocks, ranges, templates = identify_manifest_pages(result_text)
    if not nums:
        return []

    output_dir = os.path.dirname(txt_path)
    parts = os.path.normpath(txt_path).split(os.sep)
    idx = parts.index("Manure Trucking Network Analysis")
    year, region, county, template = parts[idx+1:idx+5]
    pdf_stem = parts[-2]
    original_pdf = os.path.join(GDRIVE_BASE, year, region, county, template, "original", f"{pdf_stem}.pdf")

    manifests, all_manifests_doc = [], fitz.open()

    for i, (block_text, (start_pg, end_pg)) in enumerate(zip(blocks, ranges), start=1):
        manifest_text = clean_common_errors(block_text)
        manifest_template = templates[i - 1]
        data = extract_manifest_fields(manifest_text, manifest_template)
        metadata = {"Source PDF": pdf_stem, "Start Page": start_pg, "End Page": end_pg}

        # Multi-row table or single manifest
        table_rows = _parse_hauling_table(manifest_text) if manifest_template == "R5-2007-0035_one_page_2" else None
        if table_rows:
            entries = [{**data, **row, **metadata, "Manifest Number": f"{i}{chr(97 + j)}"} for j, row in enumerate(table_rows)]
        else:
            entries = [{**data, **metadata, "Manifest Number": i}]

        for entry in entries:
            _split_haul_dates(entry)
            manifests.append(entry)

        # Save individual manifest files
        with open(os.path.join(output_dir, f"manifest_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(manifest_text)
        with fitz.open(original_pdf) as doc, fitz.open() as new_doc:
            for p in range(start_pg - 1, end_pg):
                if 0 <= p < len(doc):
                    new_doc.insert_pdf(doc, from_page=p, to_page=p)
                    all_manifests_doc.insert_pdf(doc, from_page=p, to_page=p)
            new_doc.save(os.path.join(output_dir, f"manifest_{i}.pdf"))

    if len(all_manifests_doc) > 0:
        all_manifests_doc.save(os.path.join(output_dir, "all_manifests.pdf"))
        all_manifests_doc.close()

    print(f"{len(nums)} manifests of {template} in {pdf_stem}")
    return manifests


def main():
    """Extract all manifests and save to CSV."""
    stems = {}
    for ocr_method in ["llmwhisperer", "tesseract"]:
        files = [p for p in glob.glob(f"{GDRIVE_BASE}/data/{REGION}/**/{ocr_method}_output/**/*.txt", recursive=True)
                 if not os.path.basename(p).startswith("manifest_")]
        out = {}
        for p in sorted(files):
            if (stem := pdf_stem_from_txt_path(p)) in out:
                raise ValueError(f"Duplicate txt for {stem} in {ocr_method}_output: {out[stem]} and {p}")
            out[stem] = p
        stems[ocr_method] = out

    all_manifests = []
    for stem in sorted(set(stems["tesseract"]) | set(stems["llmwhisperer"])):
        chosen = stems["llmwhisperer"].get(stem) or stems["tesseract"].get(stem)
        all_manifests.extend(extract_manifests_from_txt(chosen))

    df = pd.DataFrame(all_manifests)
    print(df.head())
    
    out_csv = "ca_cafo_compliance/output_data/all_manifests_as_written_automatic.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    coerce_columns(df)

    # Clean contact addresses and parcel numbers
    contact_col = PARAM_TO_COL["destination_contact_address"]
    if contact_col in df.columns:
        df[contact_col] = df[contact_col].apply(lambda x: strip_phone_number(x) if isinstance(x, str) else x)
    
    parcel_col, county_col = PARAM_TO_COL["destination_parcel_number"], PARAM_TO_COL["destination_county"]
    if parcel_col in df.columns:
        split = df[parcel_col].apply(lambda x: split_apn_county(x) if isinstance(x, str) else (None, None))
        df[parcel_col] = split.apply(lambda x: x[0])
        df[county_col] = split.apply(lambda x: x[1])

    # Categorize manifest type
    manure_col, wastewater_col = PARAM_TO_COL["manure_amount"], PARAM_TO_COL["wastewater_amount"]
    has_manure, has_wastewater = df[manure_col].notna(), df[wastewater_col].notna()
    df["Manifest Type"] = "unknown"
    df.loc[has_manure & has_wastewater, "Manifest Type"] = "both"
    df.loc[has_manure & ~has_wastewater, "Manifest Type"] = "manure"
    df.loc[~has_manure & has_wastewater, "Manifest Type"] = "wastewater"

    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")


def identify_files_to_delete():
    """Generate delete lists for cleanup."""
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
    os.makedirs(OUT_DIR, exist_ok=True)

    g = lambda pat: [p for p in glob.glob(os.path.join(GDRIVE_BASE, pat), recursive=True) if "all_manifests" not in p]

    # One-page manifests
    by_pages = defaultdict(list)
    for p in g("**/manifest_*.pdf"):
        with fitz.open(p) as doc:
            by_pages[len(doc)].append(p)

    one_page = sorted({f for folder in {os.path.dirname(p) for p in by_pages[1]}
                       for ext in ("txt", "pdf") for f in glob.glob(os.path.join(folder, f"manifest_*.{ext}"))})

    # Engine-specific manifests
    engine_delete_lists = {
        engine: sorted([p for p in g(f"**/{engine}_output/**/manifest_*") if os.path.isfile(p)])
        for engine in ["fitz", "tesseract", "llmwhisperer"]
    }

    # Empty subdirectories
    def empty_subdirs(folder):
        dirs = sorted({d for d in g(f"**/{folder}/**/") if os.path.isdir(d)}, key=lambda p: p.count(os.sep), reverse=True)
        return [d for d in dirs if os.path.isdir(d) and len(os.listdir(d)) == 0]

    empty_subfolders = sum([empty_subdirs(f) for f in ["llmwhisperer_output", "fitz_output", "tesseract_output"]], [])

    # Write delete lists
    delete_lists = {
        "one_page.txt": one_page,
        **{f"delete_list_all_{k}.txt": v for k, v in engine_delete_lists.items()},
        "delete_list_empty_subfolders.txt": empty_subfolders,
    }

    for fname, items in delete_lists.items():
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(items) + "\n")


if __name__ == "__main__":
    main()
    identify_files_to_delete()