#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import pymupdf as fitz
from dateutil import parser as date_parser
from datetime import (
    datetime,
)  # TODO: make use of datetime in date parsing instead of re
from collections import defaultdict

from helpers_pdf_metrics import clean_common_errors, extract_parameters_from_text
from helpers_geocoding import parse_destination_address_and_parcel
from helpers_pdf_metrics import GDRIVE_BASE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

YEAR = "2024"
REGION = "R5"

PARAMS_DF = pd.read_csv(os.path.join(DATA_DIR, "parameters.csv"))
LOCATIONS_DF = pd.read_csv(os.path.join(DATA_DIR, "parameter_locations.csv"))
TEMPLATES_DF = pd.read_csv(os.path.join(DATA_DIR, "templates.csv"))
manifest_params = PARAMS_DF[
    PARAMS_DF["manifest_type"].isin(["manure", "wastewater", "both"])
]
PARAM_TO_COL = manifest_params.set_index("parameter_key")["parameter_name"].to_dict()
PARAM_TYPES = dict(zip(manifest_params["parameter_key"], manifest_params["data_type"]))
PARAM_DEFAULTS = dict(zip(manifest_params["parameter_key"], manifest_params["default"]))

# Load / haul pattern parsing
_NUM = r"(\d+(?:\.\d+)?)"  # capture a number
_LOAD_TERMS = [
    "load",
    "loap",
    "haul",
    "truckload",
    "dump\s*t(?:k|ruck)",
    "tanker\s*load",
]
_UNIT_TERMS = ["ton", "gallon", "gal", "yard"]
_LOAD = "(?:" + "|".join(t + "s?" for t in _LOAD_TERMS) + ")"
_UNIT = "(" + "|".join(t + "s?" for t in _UNIT_TERMS) + ")"  # capturing
_SEP = r"\s*(?:[x×@\-]|at)\s*"
_APPROX = r"(?:approx\.?\s*)?"

_LOAD_PATTERNS = [
    # "418.5 Loads X 24 Tons" / "150 Loads At 9500 Gals/Load"
    # "100 Loads @ 10 Ton Per Load" / "552 Dump Tks @ Approx. 6.25 Ton Ave"
    (
        re.compile(
            rf"{_NUM}\s*{_LOAD}(?:\s+\w+)*?{_SEP}{_APPROX}{_NUM}\s*{_UNIT}", re.I
        ),
        (0, 1, 2),
    ),
    # "24 Tons X 56 Loads" (units before loads)
    (re.compile(rf"{_NUM}\s*{_UNIT}{_SEP}{_NUM}\s*{_LOAD}", re.I), (2, 0, 1)),
    # "130 Loads 12 Tons Each" (no separator sign)
    (re.compile(rf"{_NUM}\s*{_LOAD}\s+{_NUM}\s*{_UNIT}\s+each\b", re.I), (0, 1, 2)),
]

# Wastewater-specific patterns
_HOURS_RE = re.compile(r"(\d+(?:\s*\d+/\d+)?(?:\.\d+)?)\s*(?:hours?|hrs?)\b", re.I)
_GPM_RE = re.compile(r"(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:gpm|gallons per min)\b", re.I)
_FRAC_RE = re.compile(r"(\d+)(\d)/(\d+)")  # "111/2" -> whole=11, num=1, denom=2


def pdf_stem_from_txt_path(txt_path):
    parts = os.path.normpath(txt_path).split(os.sep)
    for ocr_folder in ("llmwhisperer_output", "tesseract_output"):
        if ocr_folder in parts:
            i = parts.index(ocr_folder)
            # /<ocr_folder>/<pdf_stem>/<pdf_stem>.txt
            if i + 1 < len(parts):
                return parts[i + 1]


def identify_manifest_pages(result_text):
    """
    - P1 has instructions/info
    - if "certification" or "signature of hauler" is on page 1, then don't include page 2
    - if page 2 has "Page 2 of 3", then also include page 3
    """
    matches = list(re.compile(r"=== Page (\d+) ===").finditer(result_text))
    if not matches:
        print("No page markers found")
        return [], [], [], []

    # Extract pages
    pages = {
        int(m.group(1)): result_text[
            m.end() : (
                matches[i + 1].start() if i + 1 < len(matches) else len(result_text)
            )
        ].strip()
        for i, m in enumerate(matches)
    }

    used, manifest_num, nums, blocks, ranges, templates = set(), 0, [], [], [], []
    sorted_pages = sorted(pages)

    for idx, p1 in enumerate(sorted_pages):
        t1_upper = pages[p1].upper()
        # Skip if used, attachment, or not manifest start (has header + instructions)
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

        manifest_num += 1
        used.add(p1)
        combined, end_pg = pages[p1], p1

        # Add page 2 if: exists, unused, not manifest start, and p1 has no certification
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
        if (
            "PAGE 2 OF 3" in combined.upper()
            and (p3 := end_pg + 1) in pages
            and p3 not in used
        ):
            used.add(p3)
            combined += "\n" + pages[p3]
            end_pg = p3

        # Template matching
        text_upper = clean_common_errors(combined).upper()
        template = next(
            (
                row["template_key"]
                for _, row in TEMPLATES_DF.iterrows()
                if (kw := row["keywords"])
                and not pd.isna(kw)
                and all(
                    any(t.strip() in text_upper for t in c.split("|"))
                    for c in str(kw).upper().split("&&")
                )
                and row["page_count"] == (end_pg - p1 + 1)
            ),
            "R5-2007-0035_general_order",
        )  # backup to R5-2007

        nums.append(manifest_num)
        blocks.append(combined)
        ranges.append((p1, end_pg))
        templates.append(template)

    return nums, blocks, ranges, templates


_TABLE_ROW_RE = re.compile(
    r"^(.+?)\s+"  # group 1: date range
    r"([\d,]+)\s+"  # group 2: amount
    r"(tons?|gallons?|gals?|yards?)\s+"  # group 3: units
    r"(\d+)\s*%",  # group 4: moisture %
    re.I,
)


def _parse_hauling_table(manifest_text):
    """Extract hauling event rows from table for templates with hauling tables."""
    lines = manifest_text.split("\n")
    start_idx = next(
        (
            i + 1
            for i, ln in enumerate(lines)
            if "date" in ln.lower() and "haul" in ln.lower()
        ),
        None,
    )
    if start_idx is None:
        return []

    rows = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("total"):
            break
        if not (m := _TABLE_ROW_RE.match(stripped)):
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


_MONTHS = (
    r"january|february|march|april|may|june|july|august|september"
    r"|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
)
_DATE_TOKEN_RE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4}"
    rf"|(?:{_MONTHS})(?:\s+\d{{1,2}})?(?:\s*,?\s*\d{{2,4}})?",
    re.I,
)
_MONTH_ONLY_RE = re.compile(rf"^\s*({_MONTHS})\s*$", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _split_haul_dates(data):
    """Parse haul_date into haul_date_first and haul_date_last in-place."""
    haul_date = data.get(PARAM_TO_COL["haul_date"])
    if not haul_date or not isinstance(haul_date, str):
        return

    date_parts = _DATE_TOKEN_RE.findall(haul_date)
    if not date_parts:
        date_parts = [
            p.strip()
            for p in re.split(r"[-–—]|\bto\b|,|;|&", haul_date, flags=re.I)
            if p.strip()
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

    if not parsed:
        return
    parsed.sort()
    fmt = lambda d: f"{d.month}/{d.day}/{d.year}"
    data[PARAM_TO_COL["haul_date_first"]] = fmt(parsed[0]) if len(parsed) > 1 else None
    data[PARAM_TO_COL["haul_date_last"]] = fmt(parsed[-1])


def extract_manifest_fields(manifest_text, template):
    # Extract all parameters using shared function (returns param_key -> value)
    extracted = extract_parameters_from_text(
        manifest_text, template, LOCATIONS_DF, PARAM_TYPES, PARAM_DEFAULTS
    )

    # Map to column names and initialize result. Initialize blank column if no values extracted
    data = {PARAM_TO_COL[k]: v for k, v in extracted.items()}
    # initialize any missing columns to None
    for k in PARAM_TO_COL:
        col = PARAM_TO_COL[k]
        if col not in data:
            data[col] = None
    data["Parameter Template"] = template

    # Post-process specific parameters with special handling
    for param_key, column_name in PARAM_TO_COL.items():
        value = data.get(column_name)
        if value is None:
            continue

        # Standardize destination_type -> destination_type_std
        std_key = param_key + "_std"
        if std_key in PARAM_TO_COL:
            std_val = value
            if param_key == "destination_type" and isinstance(value, str):
                vl = value.lower()
                if "(as identified" in vl or "above)" in vl:
                    std_val = None
                elif "compost" in vl:
                    std_val = "Composting Facility"
                elif "Farmer" in value:
                    std_val = value[value.find("Farmer") :]
                    std_val = re.sub(r"^Farmer[\s.—-]+$", "Farmer", std_val)
            data[PARAM_TO_COL[std_key]] = std_val

        # Parse destination into address and/or parcel; store both when present
        if param_key == "destination_address":
            address_part, parcel_part = parse_destination_address_and_parcel(value)
            # Only set parcel from address parsing if not already explicitly extracted
            if parcel_part and not data.get(PARAM_TO_COL["destination_parcel_number"]):
                data[PARAM_TO_COL["destination_parcel_number"]] = parcel_part
            value = (
                address_part if address_part else (value if not parcel_part else None)
            )
            data[column_name] = value

    for wt, units in [("manure", ["ton", "yard"]), ("wastewater", ["gallon"])]:
        if not (txt := data.get(f"Method Used to Determine Volume of {wt.title()}")):
            continue
        txt_str = str(txt)

        # Extract loads/hauls and per-load amounts from method text
        for regex, (li, ai, ui) in _LOAD_PATTERNS:
            if m := regex.search(txt_str):
                g = m.groups()
                data[PARAM_TO_COL[f"{wt}_number_hauls"]] = g[li]
                unit_text = g[ui].lower()
                for u in units:
                    if u in unit_text:
                        data[PARAM_TO_COL[f"{wt}_{u}_per_haul"]] = g[ai]
                break

        # Wastewater-only: hours pumped and GPM
        if wt == "wastewater":
            if m := _HOURS_RE.search(txt_str):
                # Parse hour text like '11', '11.5', or '11 1/2'
                s = m.group(1)
                hours = s.strip().replace(" ", "")
                if "/" not in s:
                    hours = s
                elif m := _FRAC_RE.match(s):
                    hours = float(m.group(1)) + float(m.group(2)) / float(m.group(3))
                elif m := re.match(r"(\d+)/(\d+)", s):
                    hours = float(m.group(1)) / float(m.group(2))
                data[PARAM_TO_COL["wastewater_hours_pumped"]] = hours
            if m := _GPM_RE.search(txt_str):
                data[PARAM_TO_COL["wastewater_pumping_rate"]] = m.group(1)

    # add is_pipeline column depending on whether "pipeline" is in manifest text
    data[PARAM_TO_COL["is_pipeline"]] = "pipeline" in manifest_text.lower()
    return data


def extract_manifests_from_txt(txt_path):
    pdf_stem = pdf_stem_from_txt_path(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        result_text = f.read()

    nums, blocks, ranges, templates = identify_manifest_pages(result_text)
    if not nums:
        # print(f"No manifests found in {txt_path}")
        return []

    output_dir = os.path.dirname(txt_path)

    parts = os.path.normpath(txt_path).split(os.sep)
    idx = parts.index("ca_cafo_manifests")
    year, region, county, template = parts[idx + 1 : idx + 5]
    pdf_stem = parts[-2]
    original_pdf = os.path.join(
        GDRIVE_BASE, year, region, county, template, "original", f"{pdf_stem}.pdf"
    )

    manifests: list[dict] = []
    all_manifests_doc = fitz.open()

    for i, (block_text, (start_pg, end_pg)) in enumerate(zip(blocks, ranges), start=1):
        manifest_text = clean_common_errors(block_text)
        manifest_template = templates[i - 1]

        data = extract_manifest_fields(manifest_text, manifest_template)
        metadata = {
            "Source PDF": pdf_stem,
            "Start Page": start_pg,
            "End Page": end_pg,
        }

        # Multi-row table template: one manifest entry per hauling row
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

        # Save manifest txt + pdf slice
        with open(
            os.path.join(output_dir, f"manifest_{i}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(manifest_text)
        with fitz.open(original_pdf) as doc, fitz.open() as new_doc:
            for p in range(start_pg - 1, end_pg):
                if 0 <= p < len(doc):
                    new_doc.insert_pdf(doc, from_page=p, to_page=p)
                    all_manifests_doc.insert_pdf(doc, from_page=p, to_page=p)
            new_doc.save(os.path.join(output_dir, f"manifest_{i}.pdf"))

    if all_manifests_doc is not None and len(all_manifests_doc) > 0:
        all_manifests_path = os.path.join(output_dir, "all_manifests.pdf")
        all_manifests_doc.save(all_manifests_path)
        all_manifests_doc.close()

    print(f"{len(nums)} manifests of {template} in {pdf_stem}")
    return manifests


def main():
    all_manifests: list[dict] = []
    stems = {}
    for ocr_method in ["llmwhisperer", "tesseract"]:
        folder_name = f"{ocr_method}_output"
        files = [
            p
            for p in glob.glob(
                f"{GDRIVE_BASE}/{YEAR}/{REGION}/**/{folder_name}/**/*.txt",
                recursive=True,
            )
            if not os.path.basename(p).startswith("manifest_")
        ]
        out = {}
        for p in sorted(files):
            if (stem := pdf_stem_from_txt_path(p)) in out:
                raise ValueError(
                    f"Duplicate txt for pdf_stem={stem} in {folder_name}: {out[stem]} and {p}"
                )
            out[stem] = p
        stems[ocr_method] = out

    all_stems = sorted(set(stems["tesseract"]) | set(stems["llmwhisperer"]))
    for stem in all_stems:  # all PDFs
        # prioritize llmwhisperer if it was run, then tesseract for simpler PDFs
        chosen = stems["llmwhisperer"].get(stem) or stems["tesseract"].get(stem)
        all_manifests.extend(extract_manifests_from_txt(chosen))

    print(all_manifests[:2])  # print first 2 for sanity check
    df = pd.DataFrame(all_manifests)
    print(df.head())
    out_csv = "ca_cafo_compliance/outputs/2024_manifests_raw.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Coerce all numeric columns
    numeric_cols = [PARAM_TO_COL[k] for k, t in PARAM_TYPES.items() if t == "numeric"]
    print(df.columns)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    n_total = len(df)
    manure_col = PARAM_TO_COL["manure_amount"]
    wastewater_col = PARAM_TO_COL["wastewater_amount"]
    n_manure = df[manure_col].notnull().sum()
    n_wastewater = df[wastewater_col].notnull().sum()

    summary_df = pd.DataFrame(
        [
            {
                "n_total": n_total,
                "n_manure": n_manure,
                "frac_manure": n_manure / n_total if n_total else 0,
                "n_wastewater": n_wastewater,
                "frac_wastewater": n_wastewater / n_total if n_total else 0,
            }
        ]
    )
    summary_df.to_csv(
        "ca_cafo_compliance/outputs/2024_manifest_summary.csv", index=False
    )

    has_manure = df[manure_col].notna()
    has_wastewater = df[wastewater_col].notna()

    df["Manifest Type"] = "unknown"
    df.loc[has_manure & has_wastewater, "Manifest Type"] = "both"
    df.loc[has_manure & ~has_wastewater, "Manifest Type"] = "manure"
    df.loc[~has_manure & has_wastewater, "Manifest Type"] = "wastewater"

    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")


def identify_files_to_delete():
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Helpers
    def g(pattern):
        return [
            p
            for p in glob.glob(os.path.join(GDRIVE_BASE, pattern), recursive=True)
            if "all_manifests" not in p
        ]

    # One-page manifests
    manifest_pdfs = g("**/manifest_*.pdf")
    by_pages = defaultdict(list)
    for p in manifest_pdfs:
        with fitz.open(p) as doc:
            n = len(doc)
            by_pages[n].append(p)

    folders_1pg = {os.path.dirname(p) for p in by_pages[1]}
    one_page = sorted(
        {
            f
            for folder in folders_1pg
            for ext in ("txt", "pdf")
            for f in glob.glob(os.path.join(folder, f"manifest_*.{ext}"))
        }
    )

    # All manifests for each OCR approach
    engine_patterns = {
        "fitz": "**/fitz_output/**/manifest_*",
        "tesseract": "**/tesseract_output/**/manifest_*",
        "llmwhisperer": "**/llmwhisperer_output/**/manifest_*",
    }
    engine_delete_lists = {
        k: sorted([p for p in g(pat) if os.path.isfile(p)])
        for k, pat in engine_patterns.items()
    }

    # Empty subdirectories under an output_type folder (fitz/tesseract)
    def empty_subdirs(output_folder):
        dirs = [d for d in g(f"**/{output_folder}/**/") if os.path.isdir(d)]
        dirs = sorted(set(dirs), key=lambda p: p.count(os.sep), reverse=True)
        return [d for d in dirs if os.path.isdir(d) and len(os.listdir(d)) == 0]

    empty_subfolders = sum(
        [
            empty_subdirs(f)
            for f in ["llmwhisperer_output", "fitz_output", "tesseract_output"]
        ],
        [],
    )

    # Write delete lists
    delete_lists = {
        "one_page.txt": one_page,
        "delete_list_all_fitz.txt": engine_delete_lists["fitz"],
        "delete_list_all_tesseract.txt": engine_delete_lists["tesseract"],
        "delete_list_all_llmwhisperer.txt": engine_delete_lists["llmwhisperer"],
        "delete_list_empty_subfolders.txt": empty_subfolders,
    }

    for fname, items in delete_lists.items():
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(items) + "\n")

    # Deletion commands (keeping for reference)
    #   while IFS= read -r f; do rm -f "$f"; done < ca_cafo_compliance/outputs/delete_list_all_fitz.txt
    #   while IFS= read -r f; do rm -f "$f"; done < ca_cafo_compliance/outputs/delete_list_all_tesseract.txt
    #   while IFS= read -r f; do rm -f "$f"; done < ca_cafo_compliance/outputs/delete_list_all_llmwhisperer.txt
    #   while IFS= read -r d; do rmdir "$d" 2>/dev/null; done < ca_cafo_compliance/outputs/delete_list_empty_subfolders.txt


if __name__ == "__main__":
    main()
    identify_files_to_delete()
