#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import pymupdf as fitz

from helper_functions.read_report_helpers import clean_common_errors, find_parameter_value
from helper_functions.geocoding_helpers import geocode_address, load_geocoding_cache

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

manifest_params = PARAMS_DF[(PARAMS_DF["type"] == "manifest") & (PARAMS_DF["source"] == "pdf")]
PARAM_TO_COLUMN = manifest_params.set_index("parameter_key")["parameter_name"].to_dict()
MANIFEST_COLUMNS = list(PARAM_TO_COLUMN.values()) + ["Source PDF", "Manifest Number"]

# Add extra columns for full and geocoded addresses
GEOCODED_ADDRESS_COLS = ["Origin Dairy Address (Geocoded)", "Destination Address (Geocoded)"]
MANIFEST_COLUMNS = list(PARAM_TO_COLUMN.values()) + ["Source PDF", "Manifest Number"] + GEOCODED_ADDRESS_COLS

PARAM_TYPES = dict(zip(PARAMS_DF["parameter_key"], PARAMS_DF["data_type"]))
PARAM_DEFAULTS = dict(zip(PARAMS_DF["parameter_key"], PARAMS_DF["default"]))

PAGE_SPLIT_RE = re.compile(r"=== Page (\d+) ===")


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


def original_pdf_path_from_txt_path(txt_path):
    """
    Reconstruct original PDF path from the output folder structure:
      .../ca_cafo_manifests/<year>/<region>/<county>/<template>/<ocr_folder>/<pdf_stem>/<pdf_stem>.txt
    """
    parts = os.path.normpath(txt_path).split(os.sep)
    try:
        i = parts.index("ca_cafo_manifests")
        year, region, county, template = parts[i + 1:i + 5]
        pdf_stem = parts[-2]
        return os.path.join(GDRIVE_BASE, year, region, county, template, "original", f"{pdf_stem}.pdf")
    except (ValueError, IndexError):
        return None


def detect_ocr_format(txt_path):
    if "tesseract_output" in txt_path:
        return "tesseract"
    if "llmwhisperer_output" in txt_path:
        return "llmwhisperer"
    if "marker_output" in txt_path:
        return "marker"
    return "fitz"


def read_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def identify_manifest_pages(result_text):
    """Identify 2-page manifest pairs: page1 has instructions/info; page2 has method used or is the next page."""

    def has_manifest_header(text: str) -> bool:
        t = text.upper()
        return ("MANIFEST" in t) and any(k in t for k in ["TRACKING", "ATTACHMENT"])

    def has_any(text: str, terms: list[str]) -> bool:
        t = text.upper()
        return any(term in t for term in terms)

    matches = list(PAGE_SPLIT_RE.finditer(result_text))
    if not matches:
        return [], [], []

    # page_num -> page_text
    pages: dict[int, str] = {}
    for i, m in enumerate(matches):
        pnum = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(result_text)
        pages[pnum] = result_text[start:end].strip()

    sorted_pages = sorted(pages)
    used: set[int] = set()

    manifest_nums: list[int] = []
    manifest_blocks: list[str] = []
    manifest_ranges: list[tuple[int, int]] = []
    manifest_num = 0

    for idx, p1 in enumerate(sorted_pages):
        if p1 in used:
            continue

        t1 = pages[p1]
        if "REQUIRED ATTACHMENTS" in t1.upper():
            continue

        # Page 1: instructions/information + manifest header
        if not (has_manifest_header(t1) and has_any(t1, ["INSTRUCTIONS", "INFORMATION"])):
            continue

        # Page 2: prefer immediate next page if it contains METHOD USED; otherwise use immediate next page anyway
        p2 = None
        if idx + 1 < len(sorted_pages):
            cand = sorted_pages[idx + 1]
            if cand not in used:
                p2 = cand

        if p2 is not None and not has_any(pages[p2], ["METHOD USED"]):
            # If it's not the method-used page, still keep it because you said:
            # "second page ... or will be immediately following first page"
            pass

        manifest_num += 1
        manifest_nums.append(manifest_num)

        used.add(p1)
        combined = t1
        start_page, end_page = p1, p1

        if p2 is not None:
            used.add(p2)
            combined += "\n\n" + pages[p2]
            end_page = p2

        manifest_blocks.append(combined)
        manifest_ranges.append((start_page, end_page))

    return manifest_nums, manifest_blocks, manifest_ranges


def _template_for_manifest_block(manifest_text: str, ocr_format: str) -> str:
    """Return the parameter_locations template name for this manifest block."""
    u = manifest_text.upper()
    is_2013 = ("R5-2013" in u) or ("13-0122" in u)
    is_2007 = ("R5-2007" in u) or ("07-0035" in u)
    likely_handwriting = ("YARDS" in u) or ("CIRCLE" in u) or ("REPORTED IN" in u)
    has_page_2 = "D-2" in u
    if likely_handwriting or has_page_2:
        if is_2013:
            return f"manifest_R5-2013-0122_{ocr_format}"
        if is_2007:
            kind = "manifest_handwritten" if has_page_2 else "manifest_onepage"
            return f"{kind}_R5-2007-0035_{ocr_format}"
    if is_2013:
        return f"manifest_R5-2013-0122_{ocr_format}"
    return f"manifest_digital_R5-2007-0035_{ocr_format}"


def _is_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    return str(val).strip() == ""


def _apply_tesseract_backup_for_manure(manifests: list[dict], backup_txt_path: str) -> None:
    """When primary OCR missed manure amount or solids, try filling from tesseract output."""
    backup_text = read_txt(backup_txt_path)
    _, backup_blocks, _ = identify_manifest_pages(backup_text)
    if not backup_blocks or len(backup_blocks) != len(manifests):
        return
    manure_col = "Total Manure Amount (tons)"
    solids_col = "Manure Solids (%)"
    for i, m in enumerate(manifests):
        need_manure = _is_empty(m.get(manure_col))
        need_solids = _is_empty(m.get(solids_col))
        if not need_manure and not need_solids:
            continue
        block = clean_common_errors(backup_blocks[i])
        template = _template_for_manifest_block(block, "tesseract")
        backup_data = extract_manifest_fields(block, template)
        if need_manure and not _is_empty(backup_data.get(manure_col)):
            m[manure_col] = backup_data[manure_col]
        if need_solids and not _is_empty(backup_data.get(solids_col)):
            m[solids_col] = backup_data[solids_col]


def extract_manifest_fields(manifest_text, template):
    data = {col: None for col in MANIFEST_COLUMNS}
    origin_dairy_address_geocoded = None
    destination_address_geocoded = None

    for param_key, column_name in PARAM_TO_COLUMN.items():
        m = LOCATIONS_DF[
            (LOCATIONS_DF["template"] == template) &
            (LOCATIONS_DF["parameter_key"] == param_key)
        ]
        if m.empty:
            # print(f"  {param_key} not found in {template}")
            continue

        row = m.iloc[0]
        value = find_parameter_value(manifest_text, row, PARAM_TYPES, PARAM_DEFAULTS)
        value = apply_parameter_standardization(value, param_key)
        # Save full text and geocoded for origin and destination addresses
        if param_key == "origin_dairy_address":
            if value:
                geocoded = geocode_address(value, cache)
                # geocoded = "placeholder" # TODO: switch back to geocoding
                if geocoded:
                    origin_dairy_address_geocoded = geocoded
        if param_key == "destination_address":
            if value:
                geocoded = geocode_address(value, cache)
                # geocoded = "placeholder" # TODO: switch back to geocoding
                if geocoded:
                    destination_address_geocoded = geocoded

        # For the main columns, use full text
        data[column_name] = value

    data["Origin Dairy Address (Geocoded)"] = origin_dairy_address_geocoded
    data["Destination Address (Geocoded)"] = destination_address_geocoded

    return data


def extract_manifests_from_txt(txt_path, *, allow_write: bool = True, backup_txt_path: str | None = None):
    pdf_stem = pdf_stem_from_txt_path(txt_path)
    result_text = read_txt(txt_path)

    nums, blocks, ranges = identify_manifest_pages(result_text)
    if not nums:
        return []

    output_dir = os.path.dirname(txt_path)
    original_pdf = original_pdf_path_from_txt_path(txt_path)
    ocr_format = detect_ocr_format(txt_path)

    manifests: list[dict] = []
    for i, (block_text, (start_page, end_page)) in enumerate(zip(blocks, ranges), start=1):
        manifest_text = clean_common_errors(block_text)
        template = _template_for_manifest_block(manifest_text, ocr_format)

        data = extract_manifest_fields(manifest_text, template)
        data["Source PDF"] = pdf_stem
        data["Manifest Number"] = i
        manifests.append(data)

        # Save individual manifest txt
        if allow_write:
            with open(os.path.join(output_dir, f"manifest_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(manifest_text)

        # Save manifest pdf (2-page slice)
        if allow_write and original_pdf and os.path.exists(original_pdf):
            out_pdf = os.path.join(output_dir, f"manifest_{i}.pdf")
            with fitz.open(original_pdf) as doc, fitz.open() as new_doc:
                for p in range(start_page - 1, end_page):  # end_page is inclusive in your logic
                    if 0 <= p < len(doc):
                        new_doc.insert_pdf(doc, from_page=p, to_page=p)
                new_doc.save(out_pdf)

    if backup_txt_path and ocr_format != "tesseract" and os.path.isfile(backup_txt_path):
        _apply_tesseract_backup_for_manure(manifests, backup_txt_path)

    print(f"{len(nums)} manifests of {template} in {pdf_stem} ({detect_ocr_format(txt_path)})")
    return manifests


def index_ocr_txt_files(folder_name: str) -> dict[str, str]:
    """
    Returns: {pdf_stem: txt_path} for a given OCR folder.
    Assumes at most one non-manifest txt per pdf stem (your guarantee).
    """
    pattern = f"{GDRIVE_BASE}/{YEAR}/{REGION}/**/{folder_name}/**/*.txt"
    files = [
        p for p in glob.glob(pattern, recursive=True)
        if not os.path.basename(p).startswith("manifest_")
    ]
    out: dict[str, str] = {}
    for p in sorted(files):
        stem = pdf_stem_from_txt_path(p)
        if stem in out:
            raise ValueError(f"Duplicate txt for pdf_stem={stem} in {folder_name}: {out[stem]} and {p}")
        out[stem] = p
    return out


def needs_handwiting_ocr(txt_path: str) -> bool:
    text = read_txt(txt_path)
    t = text.upper()
    return ("R5-2013" in t) or ("2013-0122 in t") or ("CUBIC YARDS" in t)
    # TODO: merge with template identification

def apply_parameter_standardization(parameter, param_key):
    if parameter is None or not isinstance(parameter, str):
        return parameter
    if param_key == "destination_address":
        # look for XXX-XXX-XXX pattern and apply to destination_parcel_number
        # match = re.search(r'\d{3}-\d{3}-\d{3}', parameter)
        # if match:
        #     return match.group(0)
        return parameter # TODO: implement parcel extraction
    if param_key == "destination_type":
        # replace Compost, Compost Facility, and Composting with Composting Facility
        if type(parameter) == str:
            if "compost" in parameter.lower():
                return "Composting Facility"
        return parameter

    return parameter

def main():
    all_manifests: list[dict] = []

    tesseract = index_ocr_txt_files("tesseract_output")
    marker = index_ocr_txt_files("marker_output")
    fitz = index_ocr_txt_files("fitz_output")
    llmwhisperer = index_ocr_txt_files("llmwhisperer_output")

    all_stems = sorted(set(tesseract) | set(llmwhisperer) | set(fitz))
    if not all_stems:
        print("No OCR txt files found")
        return

    # Decide per stem whether marker is required (scan any available OCR for the signal)
    handwriting_ocr_required: dict[str, bool] = {}
    for stem in all_stems:
        scan_path = llmwhisperer.get(stem) or tesseract.get(stem) or fitz.get(stem)
        handwriting_ocr_required[stem] = needs_handwiting_ocr(scan_path) if scan_path else False

    # Extract using the selected OCR source
    for stem in all_stems:
        if handwriting_ocr_required[stem]:
            chosen = llmwhisperer.get(stem)
            if not chosen:
                print(f"[WARN] {stem}: llmwhisperer_output missing; falling back to tesseract")
                chosen = tesseract.get(stem) or fitz.get(stem)
        else:
            # Prefer tesseract for R5-2007-0035 (and everything else unless handwriting OCR-required)
            chosen = tesseract.get(stem)
            if not chosen:
                chosen = llmwhisperer.get(stem) or fitz.get(stem)

        if not chosen:
            continue

        backup_txt = tesseract.get(stem) if (stem in tesseract and chosen != tesseract.get(stem) and detect_ocr_format(chosen) != "tesseract") else None
        manifests = extract_manifests_from_txt(chosen, allow_write=True, backup_txt_path=backup_txt)
        all_manifests.extend(manifests)

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
    n_manure = df["Total Manure Amount (tons)"].notnull().sum()
    n_wastewater = df["Total Process Wastewater Exports (Gallons)"].notnull().sum()

    summary_df = pd.DataFrame([{
        "n_total": n_total,
        "n_manure": n_manure,
        "frac_manure": n_manure / n_total if n_total else 0,
        "n_wastewater": n_wastewater,
        "frac_wastewater": n_wastewater / n_total if n_total else 0
    }])
    summary_df.to_csv("ca_cafo_compliance/outputs/manifest_extraction_summary.csv", index=False)

    manure_cols = ["Total Manure Amount (tons)"]
    wastewater_cols = ["Total Process Wastewater Exports (Gallons)"]
    has_manure = df[manure_cols].notna().any(axis=1)
    has_wastewater = df[wastewater_cols].notna().any(axis=1)

    print(f"\n  Manure manifests: {int(has_manure.sum())}")
    print(f"  Wastewater manifests: {int(has_wastewater.sum())}")
    print(f"  Both: {int((has_manure & has_wastewater).sum())}")
    print(f"  Unknown: {int((~has_manure & ~has_wastewater).sum())}")

    df_manure = df[has_manure]
    df_wastewater = df[has_wastewater]

    if not df_manure.empty:
        manure_csv = "ca_cafo_compliance/outputs/extracted_manure_manifests.csv"
        gdrive_manure_csv = os.path.join(GDRIVE_BASE, "extracted_manure_manifests.csv")
        df_manure.to_csv(manure_csv, index=False)
        df_manure.to_csv(gdrive_manure_csv, index=False)
        print(f"\nSaved {len(df_manure)} manure manifests")

    if not df_wastewater.empty:
        wastewater_csv = "ca_cafo_compliance/outputs/extracted_wastewater_manifests.csv"
        gdrive_wastewater_csv = os.path.join(GDRIVE_BASE, "extracted_wastewater_manifests.csv")
        df_wastewater.to_csv(wastewater_csv, index=False)
        df_wastewater.to_csv(gdrive_wastewater_csv, index=False)
        print(f"Saved {len(df_wastewater)} wastewater manifests")

    to_numeric = lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    print("\nManure-only summary by destination type:")
    print(df[has_manure].groupby("Destination Type")["Total Manure Amount (tons)"].apply(lambda x: to_numeric(x).sum()))
    print("\nWastewater-only summary by destination type:")
    print(df[has_wastewater].groupby("Destination Type")["Total Process Wastewater Exports (Gallons)"].apply(lambda x: to_numeric(x).sum()))

if __name__ == "__main__":
    main()