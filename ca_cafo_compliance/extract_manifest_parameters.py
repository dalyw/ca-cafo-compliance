#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import pymupdf as fitz

from helper_functions.read_report_helpers import clean_common_errors, find_parameter_value
from helper_functions.geocoding_helpers import (
    geocode_address,
    geocode_parcel,
    load_geocoding_cache,
    parse_destination_address_and_parcel,
)

GDRIVE_BASE = "/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/ca_cafo_manifests"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

YEAR = "2024"
REGION = "R5"

# use handwriting OCR instead of tesseract if handwriting OCR produced manifests for that PDF
OCR_PRIORITY_FOLDERS = ["llmwhisperer_output", "tesseract_output"]

cache = load_geocoding_cache()

PARAMS_DF = pd.read_csv(os.path.join(DATA_DIR, "parameters.csv"))
LOCATIONS_DF = pd.read_csv(os.path.join(DATA_DIR, "parameter_locations.csv"))
TEMPLATES_DF = pd.read_csv(os.path.join(DATA_DIR, "templates.csv"))
manifest_params = PARAMS_DF[PARAMS_DF["manifest_type"].isin(["manure", "wastewater", "both"])]
manifest_param_cols = manifest_params.set_index("parameter_key")["parameter_name"].tolist()
manure_param_cols = manifest_params[manifest_params["manifest_type"].isin(["manure", "both"])]["parameter_name"].tolist()
wastewater_param_cols = manifest_params[manifest_params["manifest_type"].isin(["wastewater", "both"])]["parameter_name"].tolist()
PARAM_TO_COL = manifest_params.set_index("parameter_key")["parameter_name"].to_dict()

BASE_MANIFEST_COLUMNS = (
    list(PARAM_TO_COL.values())
    + ["Source PDF", "Manifest Number"]
)
MANIFEST_COLUMNS = BASE_MANIFEST_COLUMNS + ["Parameter Template"]


PARAM_TYPES = dict(zip(PARAMS_DF["parameter_key"], PARAMS_DF["data_type"]))
PARAM_DEFAULTS = dict(zip(PARAMS_DF["parameter_key"], PARAMS_DF["default"]))

# Parse "X Loads @ Y Tons Per Load" / "X Loads @ Y Gallons Per Load" from method text
_NUMBER_HAULS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:Loads?|Hauls?|Truckloads?)\b",
    re.IGNORECASE,
)

def _parse_number(s):
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def pdf_stem_from_txt_path(txt_path):
    parts = os.path.normpath(txt_path).split(os.sep)
    for ocr_folder in ("llmwhisperer_output", "tesseract_output", "fitz_output"):
        if ocr_folder in parts:
            i = parts.index(ocr_folder)
            # /<ocr_folder>/<pdf_stem>/<pdf_stem>.txt
            if i + 1 < len(parts):
                return parts[i + 1]
    # fallback TODO: remove
    return os.path.splitext(os.path.basename(txt_path))[0]


def read_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def identify_manifest_pages(result_text):
    """
    Identify manifest page groups
    - page1 has instructions/info
    - if "certification" is on page 1, then don't include page 2
    - if not, include page 2
    - if page 2 has "Page 2 of 3", then also include page 3
    """

    def has_manifest_header(text: str) -> bool:
        t = text.upper()
        return ("MANIFEST" in t) and any(k in t for k in ["TRACKING", "ATTACHMENT"])

    def has_any(text: str, terms: list[str]) -> bool:
        t = text.upper()
        return any(term in t for term in terms)

    matches = list(re.compile(r"=== Page (\d+) ===").finditer(result_text))
    if not matches:
        print("No page markers found")
        return [], [], [], []

    # page_num -> page_text
    pages: dict[int, str] = {}
    for i, m in enumerate(matches):
        pnum = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(result_text)
        pages[pnum] = result_text[start:end].strip()

    sorted_pages = sorted(pages)
    # print(sorted_pages)
    used: set[int] = set()

    manifest_nums: list[int] = []
    manifest_blocks: list[str] = []
    manifest_ranges: list[tuple[int, int]] = []
    manifest_templates: list[str] = []
    manifest_num = 0

    for idx, p1 in enumerate(sorted_pages):
        if p1 in used:
            # print(f" Page {p1} already used")
            continue

        t1 = pages[p1]
        if "REQUIRED ATTACHMENTS" in t1.upper():
            # print(f" Removing required attachment page {p1}")
            continue

        # Page 1: instructions/information + manifest header
        if not (has_manifest_header(t1) and has_any(t1, ["INSTRUCTIONS", "Complete one"])):
            # print(f" Page {p1} doesn't contain instructions/information")
            continue

        # Page 2: prefer immediate next page unless it contains "INSTRUCTIONS" or "Complete one"
        p2 = None
        if idx + 1 < len(sorted_pages):
            cand = sorted_pages[idx + 1]
            if cand not in used:
                # If the next page is the start of a new manifest
                # (has header + instructions), don't merge it as page 2.
                t2 = pages[cand]
                cand_is_manifest_start = (
                    ("REQUIRED ATTACHMENTS" not in t2.upper())
                    and has_manifest_header(t2)
                    and has_any(t2, ["INSTRUCTIONS", "Complete one"]) # , "INFORMATION" on page 2 also...
                )
                if "black diamongd" in t2.lower():
                    print(f"black diamond {manifest_num} {cand_is_manifest_start}")
                if not cand_is_manifest_start:
                    p2 = cand

        manifest_num += 1
        manifest_nums.append(manifest_num)

        used.add(p1)
        combined = t1
        start_page, end_page = p1, p1

        if p2 is not None:
            if not "CERTIFICATION" in pages[p1].upper(): # if manifest didn't end on page 1 already
                used.add(p2)
                combined += "\n\n" + pages[p2]
                end_page = p2
            # or if p2 is identical to page 1

        # If manifest says "Page 2 of 3", include the 3rd page as well
        if "PAGE 2 OF 3" in combined.upper():
            p3 = end_page + 1
            if p3 in pages and p3 not in used:
                used.add(p3)
                combined += "\n" + pages[p3]
                end_page = p3

        manifest_blocks.append(combined)
        manifest_ranges.append((start_page, end_page))

        manifest_text = clean_common_errors(combined)
        
        # Template matching: try each template's keywords until one matches
        template = None
        text_upper = manifest_text.upper()
        for _, row in TEMPLATES_DF.iterrows():
            if (kw := row["keywords"]) and not pd.isna(kw):
                if all(any(term.strip() in text_upper for term in clause.split("|")) 
                       for clause in str(kw).upper().split("&&")):
                    if row["page_count"] != (end_page - start_page + 1):
                        continue
                    template = row["template_key"]
                    break
        
        manifest_templates.append(template)

    return manifest_nums, manifest_blocks, manifest_ranges, manifest_templates


def _is_empty(val) -> bool:
    return val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == ""

def _geocode_if_valid(addr, geocode_fn):
    """Geocode address and return result if both coordinates are non-null."""
    if not addr:
        return None
    geocoded = geocode_fn(addr, cache)
    return geocoded if geocoded and geocoded[0] is not None and geocoded[1] is not None else None


def extract_manifest_fields(manifest_text, template):
    data = {col: None for col in MANIFEST_COLUMNS}
    origin_dairy_address_geocoded = None
    destination_address_geocoded = None

    for param_key, column_name in PARAM_TO_COL.items():
        m = LOCATIONS_DF[
            (LOCATIONS_DF["template"] == template) &
            (LOCATIONS_DF["parameter_key"] == param_key)
        ]
        if m.empty:
            # print(f"  {param_key} not found in {template}")
            continue

        row = m.iloc[0]
        value = find_parameter_value(manifest_text, row, PARAM_TYPES, PARAM_DEFAULTS)

        # Save cleaned version to param_key_std when that column exists (e.g. destination_type_std)
        std_key = param_key + "_std"
        if std_key in PARAM_TO_COL:
            data[PARAM_TO_COL[std_key]] = apply_parameter_standardization(value, param_key)

        # Parse destination into address and/or parcel; store both when present; geocode by parcel when parcel present
        if param_key == "destination_address" and value:
            address_part, parcel_part = parse_destination_address_and_parcel(value)
            if parcel_part:
                data[PARAM_TO_COL["destination_parcel_number"]] = parcel_part
                destination_address_geocoded = _geocode_if_valid(parcel_part, geocode_parcel)
            value = address_part if address_part else (value if not parcel_part else None)
            if not destination_address_geocoded and address_part:
                destination_address_geocoded = _geocode_if_valid(address_part, geocode_address)
        elif param_key == "origin_dairy_address" and value:
            origin_dairy_address_geocoded = _geocode_if_valid(value, geocode_address)
        # elif param_key in ["manure_amount", "wastewater_amount"]:
        #     value = str(value).replace("N/A", "").replace("NA", "").replace("nan", "") if value else value
        
        data[column_name] = value

    data[PARAM_TO_COL["origin_dairy_address_geocoded"]] = origin_dairy_address_geocoded
    data[PARAM_TO_COL["destination_address_geocoded"]] = destination_address_geocoded
    data["Parameter Template"] = template

    for waste_type, units in [
        ("Manure", ["ton", "yard"]),
        ("Wastewater", ["gallon"]),
    ]:
        if (txt := data.get(f"Method Used to Determine Volume of {waste_type}")) and "per" in str(txt).lower():
            txt_lower = str(txt).lower()
            if (m := _NUMBER_HAULS_RE.search(txt_lower)) and (n := _parse_number(m.group(1))) is not None:
                data[PARAM_TO_COL[f"{waste_type.lower()}_number_hauls"]] = str(int(n) if n == int(n) else n)
            for unit_key in units:
                if (m := re.search(rf"(\d+(?:,\d+)?)\s*{unit_key}", txt_lower)) and (v := _parse_number(m.group(1))) is not None:
                    data[PARAM_TO_COL[f"{waste_type.lower()}_{unit_key}_per_haul"]] = v

    # Extract hours and GPM for wastewater pumping (hours * GPM pattern)
    if (ww_method_txt := data.get("Method Used to Determine Volume of Wastewater")):
        ww_method_str = str(ww_method_txt)
        # Look for hours: "X hours", "X hrs", "X hr", including fractions like "11 1/2"
        # Match patterns like: "12 hours", "11 1/2 hours", "111/2 hours", "8.5 hrs"
        if (m := re.search(r"(\d+(?:\s*\d+/\d+)?(?:\.\d+)?)\s*(?:hours?|hrs?)\b", ww_method_str, re.IGNORECASE)):
            hour_text = m.group(1).strip()
            # Handle fractions: "11 1/2" or "111/2"
            if "/" in hour_text:
                hour_text = hour_text.replace(" ", "")  # Remove spaces: "11 1/2" -> "111/2"
                if (frac_m := re.match(r"(\d+)(\d)/(\d+)", hour_text)):  # "111/2"
                    whole, num, denom = frac_m.groups()
                    hours = float(whole) + float(num) / float(denom)
                elif (frac_m := re.match(r"(\d+)/(\d+)", hour_text)):  # Just "1/2"
                    hours = float(frac_m.group(1)) / float(frac_m.group(2))
                else:
                    hours = _parse_number(hour_text)
            else:
                hours = _parse_number(hour_text)
            if hours is not None:
                data[PARAM_TO_COL["wastewater_hours_pumped"]] = hours
        # Look for GPM: "X GPM" or "X GALLONS PER MIN", case-insensitive
        if (m := re.search(r"(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:gpm|gallons per min)\b", ww_method_str, re.IGNORECASE)):
            if (gpm := _parse_number(m.group(1))) is not None:
                data[PARAM_TO_COL["wastewater_pumping_rate"]] = gpm

    # add is_pipeline column depending on whether "pipeline" is in manifest text
    data[PARAM_TO_COL["is_pipeline"]] = "pipeline" in manifest_text.lower()
    return data


def extract_manifests_from_txt(txt_path, *, backup_txt_path = None):
    pdf_stem = pdf_stem_from_txt_path(txt_path)
    result_text = read_txt(txt_path)

    nums, blocks, ranges, templates = identify_manifest_pages(result_text)
    if not nums:
        print(f"No manifests found in {txt_path}")
        return []

    output_dir = os.path.dirname(txt_path)

    parts = os.path.normpath(txt_path).split(os.sep)
    i = parts.index("ca_cafo_manifests")
    year, region, county, template = parts[i + 1:i + 5]
    pdf_stem = parts[-2]
    original_pdf = os.path.join(GDRIVE_BASE, year, region, county, template, "original", f"{pdf_stem}.pdf")

    manifests: list[dict] = []
    all_manifests_doc = None
    all_manifests_doc = fitz.open()
    
    for i, (block_text, (start_page, end_page)) in enumerate(zip(blocks, ranges), start=1):
        manifest_text = clean_common_errors(block_text)
        manifest_template = templates[i-1]
        data = extract_manifest_fields(manifest_text, manifest_template)
        data["Source PDF"] = pdf_stem
        data["Manifest Number"] = i
        manifests.append(data)
        templates.append(template)

        # Save individual manifest txt
        with open(os.path.join(output_dir, f"manifest_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(manifest_text)

        # Save manifest pdf (2-page slice)
        with fitz.open(original_pdf) as doc, fitz.open() as new_doc:
            for p in range(start_page - 1, end_page):
                if 0 <= p < len(doc):
                    new_doc.insert_pdf(doc, from_page=p, to_page=p)
                    if all_manifests_doc is not None:
                        all_manifests_doc.insert_pdf(doc, from_page=p, to_page=p)
            new_doc.save(os.path.join(output_dir, f"manifest_{i}.pdf"))
    
    if all_manifests_doc is not None and len(all_manifests_doc) > 0:
        all_manifests_path = os.path.join(output_dir, "all_manifests.pdf")
        all_manifests_doc.save(all_manifests_path)
        all_manifests_doc.close()


    print(f"{len(nums)} manifests of {template} in {pdf_stem}")
    return manifests


def needs_handwiting_ocr(txt_path: str) -> bool:
    text = read_txt(txt_path)
    t = text.upper()
    return ("R5-2013" in t) or ("2013-0122" in t) or ("CUBIC YARDS" in t)
    # TODO: merge with template identification

def apply_parameter_standardization(parameter, param_key):
    if param_key == "destination_type" and isinstance(parameter, str):
        # Reject OCR spillover from next line (e.g. "Farmer or Other (as identified")
        if "(as identified" in parameter.lower() or "above)" in parameter.lower():
            return None
        # Strip any prefix before "Farmer"
        if "Farmer" in parameter:
            parameter = parameter[parameter.find("Farmer"):]
        if "compost" in parameter.lower():
            return "Composting Facility"
    # TODO: expand
    return parameter

def main():
    all_manifests: list[dict] = []
    stems = {}
    for ocr_method in ["tesseract", "llmwhisperer", "fitz"]:
        folder_name = f"{ocr_method}_output"
        files = [p for p in glob.glob(f"{GDRIVE_BASE}/{YEAR}/{REGION}/**/{folder_name}/**/*.txt", recursive=True)
                if not os.path.basename(p).startswith("manifest_")]
        out = {}
        for p in sorted(files):
            if (stem := pdf_stem_from_txt_path(p)) in out:
                raise ValueError(f"Duplicate txt for pdf_stem={stem} in {folder_name}: {out[stem]} and {p}")
            out[stem] = p
        stems[ocr_method] = out

    all_stems = sorted(set(stems["tesseract"]) | set(stems["llmwhisperer"]) | set(stems["fitz"]))
    if not all_stems:
        print("No OCR txt files found")
        return

    # Decide per stem whether handwriting OCR is required (scan any available OCR for the signal)
    handwriting_ocr_required: dict[str, bool] = {}
    for stem in all_stems:
        scan_path = stems["llmwhisperer"].get(stem) or stems["tesseract"].get(stem) or stems["fitz"].get(stem)
        handwriting_ocr_required[stem] = needs_handwiting_ocr(scan_path) if scan_path else False

    # Extract using the selected OCR source
    for stem in all_stems:
        chosen = (stems["llmwhisperer"].get(stem) or stems["tesseract"].get(stem) or stems["fitz"].get(stem)) if handwriting_ocr_required[stem] else (stems["tesseract"].get(stem) or stems["llmwhisperer"].get(stem) or stems["fitz"].get(stem))
        if not chosen:
            continue
        backup_txt = stems["tesseract"].get(stem) if stem in stems["tesseract"] and chosen != stems["tesseract"].get(stem) and "tesseract" not in chosen else None
        all_manifests.extend(extract_manifests_from_txt(chosen, backup_txt_path=backup_txt))

    if not all_manifests:
        print("No manifests found")
        return

    print(f"\nExtracted {len(all_manifests)} total manifests")

    df = pd.DataFrame(all_manifests)
    out_csv = "ca_cafo_compliance/outputs/extracted_manifests.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")

    for col in MANIFEST_COLUMNS:
        print(f"  {col}: {df[col].notna().sum()}/{len(df)} extracted")

    n_total = len(df)
    manure_col = PARAM_TO_COL["manure_amount"]
    wastewater_col = PARAM_TO_COL["wastewater_amount"]
    n_manure = df[manure_col].notnull().sum()
    n_wastewater = df[wastewater_col].notnull().sum()

    summary_df = pd.DataFrame([{
        "n_total": n_total,
        "n_manure": n_manure,
        "frac_manure": n_manure / n_total if n_total else 0,
        "n_wastewater": n_wastewater,
        "frac_wastewater": n_wastewater / n_total if n_total else 0
    }])
    summary_df.to_csv("ca_cafo_compliance/outputs/manifest_extraction_summary.csv", index=False)

    manure_cols = [manure_col]
    wastewater_cols = [wastewater_col]
    has_manure = df[manure_cols].notna().any(axis=1)
    has_wastewater = df[wastewater_cols].notna().any(axis=1)

    print(f"\n  Manure manifests: {int(has_manure.sum())}")
    print(f"  Wastewater manifests: {int(has_wastewater.sum())}")
    print(f"  Both: {int((has_manure & has_wastewater).sum())}")
    print(f"  Unknown: {int((~has_manure & ~has_wastewater).sum())}")

    # Only count as geocoded if quantity is valid numeric (exclude N/A, text, etc.)
    to_numeric = lambda col: pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").notna()
    manure_qty_numeric = to_numeric(manure_col)
    wastewater_qty_numeric = to_numeric(wastewater_col)

    has_geocoded = df[[PARAM_TO_COL["origin_dairy_address_geocoded"], PARAM_TO_COL["destination_address_geocoded"]]].notna().all(axis=1)
    df_manure_geocoded = df[has_manure & has_geocoded & manure_qty_numeric]
    df_manure_need_validation = df[has_manure & ~(has_geocoded & manure_qty_numeric)]
    df_wastewater_geocoded = df[has_wastewater & has_geocoded & wastewater_qty_numeric]
    df_wastewater_need_validation = df[has_wastewater & ~(has_geocoded & wastewater_qty_numeric)]
    df_unknown = df[~has_manure & ~has_wastewater]

    to_save = [
        ("extracted_manure_manifests_geocoded", df_manure_geocoded, manure_param_cols),
        ("extracted_manure_manifests_need_validation", df_manure_need_validation, manure_param_cols),
        ("extracted_wastewater_manifests_geocoded", df_wastewater_geocoded, wastewater_param_cols),
        ("extracted_wastewater_manifests_need_validation", df_wastewater_need_validation, wastewater_param_cols),
        ("extracted_unknown_manifests", df_unknown, manifest_param_cols),
    ]
    for name, sub_df, cols in to_save:
        if sub_df.empty:
            continue
        sub_df = sub_df[cols + ["Source PDF", "Manifest Number", "Parameter Template"]]
        local_csv = os.path.join("ca_cafo_compliance", "outputs", f"{name}.csv")
        gdrive_csv = os.path.join(GDRIVE_BASE, f"{name}.csv")
        sub_df.to_csv(local_csv, index=False)
        sub_df.to_csv(gdrive_csv, index=False)
        print(f"{name} length: {len(sub_df)}")

    to_numeric = lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    dest_type_col = PARAM_TO_COL["destination_type_std"]
    print("\nManure-only summary by destination type:")
    print(df[has_manure].groupby(dest_type_col)[manure_col].apply(lambda x: to_numeric(x).sum()))
    print("\nWastewater-only summary by destination type:")
    print(df[has_wastewater].groupby(dest_type_col)[wastewater_col].apply(lambda x: to_numeric(x).sum()))

    # print breakdown of templates from all manifests
    print("\nTemplates breakdown:")
    print(df["Parameter Template"].value_counts())

if __name__ == "__main__":
    main()