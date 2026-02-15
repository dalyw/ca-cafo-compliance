#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import pymupdf as fitz
from dateutil import parser as date_parser
from datetime import datetime
from collections import defaultdict

from ca_cafo_compliance.helpers_pdf_metrics import (
    clean_common_errors,
    extract_parameters_from_text,
)
from ca_cafo_compliance.helpers_geocoding import (
    parse_destination_address_and_parcel,
)

from ca_cafo_compliance.helpers_pdf_metrics import GDRIVE_BASE


def to_numeric(s):
    """Convert a Series to numeric, stripping commas first."""
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

YEAR = "2024"
REGION = "R5"

# use handwriting OCR instead of tesseract if handwriting OCR produced manifests for that PDF
OCR_PRIORITY_FOLDERS = ["llmwhisperer_output", "tesseract_output"]


PARAMS_DF = pd.read_csv(os.path.join(DATA_DIR, "parameters.csv"))
LOCATIONS_DF = pd.read_csv(os.path.join(DATA_DIR, "parameter_locations.csv"))
TEMPLATES_DF = pd.read_csv(os.path.join(DATA_DIR, "templates.csv"))
manifest_params = PARAMS_DF[
    PARAMS_DF["manifest_type"].isin(["manure", "wastewater", "both"])
]
manifest_param_cols = manifest_params.set_index("parameter_key")[
    "parameter_name"
].tolist()
manure_param_cols = manifest_params[
    manifest_params["manifest_type"].isin(["manure", "both"])
]["parameter_name"].tolist()
wastewater_param_cols = manifest_params[
    manifest_params["manifest_type"].isin(["wastewater", "both"])
]["parameter_name"].tolist()
PARAM_TO_COL = manifest_params.set_index("parameter_key")["parameter_name"].to_dict()

BASE_MANIFEST_COLUMNS = list(PARAM_TO_COL.values()) + [
    "Source PDF",
    "County",
    "Consultant",
    "Manifest Number",
    "Start Page",
    "End Page",
    "Manifest Type",
]
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
    for ocr_folder in ("llmwhisperer_output", "tesseract_output"):
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
        combined, end_page = pages[p1], p1

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
                end_page = cand

        # Add page 3 if "Page 2 of 3"
        if (
            "PAGE 2 OF 3" in combined.upper()
            and (p3 := end_page + 1) in pages
            and p3 not in used
        ):
            used.add(p3)
            combined += "\n" + pages[p3]
            end_page = p3

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
                and row["page_count"] == (end_page - p1 + 1)
            ),
            "R5-2007-0035_general_order",
        )  # backup to R5-2007

        nums.append(manifest_num)
        blocks.append(combined)
        ranges.append((p1, end_page))
        templates.append(template)

    return nums, blocks, ranges, templates


def _parse_hauling_table(manifest_text):
    """Extract hauling event rows from table for R5-2007-0035_one_page_2 template."""
    lines = manifest_text.split("\n")

    # Find start of table (look for "Dates Hauled" header)
    start_idx = None
    for i, ln in enumerate(lines):
        if "dates hauled" in ln.lower() or "date hauled" in ln.lower():
            start_idx = i + 1
            break

    if start_idx is None:
        return []

    rows = []
    for line in lines[start_idx:]:
        # Stop at Total row
        if "total" in line.lower():
            break

        # Skip empty lines
        if not line.strip():
            continue

        # Parse the line - split by whitespace and work backwards
        # Structure: [date range...] [amount] [units] [moisture%] [%]
        tokens = line.split()
        if len(tokens) < 4:
            continue

        # Remove trailing % symbol if present
        if tokens[-1] == "%":
            tokens = tokens[:-1]

        # From right to left: moisture_pct, units, amount, date_range (rest)
        moisture_pct = tokens[-1] if len(tokens) >= 4 else None
        units = tokens[-2] if len(tokens) >= 3 else None
        amount = tokens[-3].replace(",", "") if len(tokens) >= 2 else None
        date_range = " ".join(tokens[:-3]) if len(tokens) > 3 else None

        rows.append(
            {
                "haul_date": date_range,
                "amount": amount,
                "units": units,
                "moisture_pct": moisture_pct,
            }
        )

    return rows


def extract_manifest_fields(manifest_text, template):
    # Extract all parameters using shared function (returns param_key -> value)
    extracted = extract_parameters_from_text(
        manifest_text, template, LOCATIONS_DF, PARAM_TYPES, PARAM_DEFAULTS
    )

    # Map to column names and initialize result
    data = {PARAM_TO_COL[k]: v for k, v in extracted.items() if k in PARAM_TO_COL}
    data["Parameter Template"] = template

    # Post-process specific parameters with special handling
    for param_key, column_name in PARAM_TO_COL.items():
        value = data.get(column_name)
        if value is None:
            continue

        # Save cleaned version to param_key_std when that column exists (e.g. destination_type_std)
        std_key = param_key + "_std"
        if std_key in PARAM_TO_COL:
            data[PARAM_TO_COL[std_key]] = apply_parameter_standardization(
                value, param_key
            )

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

    # Extract first and last haul dates from haul_date field
    haul_date = data.get(PARAM_TO_COL.get("haul_date"))
    if haul_date and isinstance(haul_date, str):
        # Split on dash, "to", ",", ";"
        date_parts = re.split(r"[-–—]|\bto\b|,|;", haul_date)
        date_parts = [p.strip() for p in date_parts if p.strip()]

        parsed_dates = []
        if date_parts:
            for part in date_parts:
                try:  # using dateutil.parser
                    dt = date_parser.parse(part, fuzzy=True)
                    parsed_dates.append(dt)
                except (ValueError, TypeError):
                    continue

        if parsed_dates:
            parsed_dates.sort()
            # Format as m/d/Y (only set first if multiple dates)
            first_date = (
                f"{parsed_dates[0].month}/{parsed_dates[0].day}/{parsed_dates[0].year}"
                if len(parsed_dates) > 1
                else None
            )
            last_date = f"{parsed_dates[-1].month}/{parsed_dates[-1].day}/{parsed_dates[-1].year}"
            # Default to 2024 if year is missing
            if parsed_dates and all(dt.year < 1900 for dt in parsed_dates):
                parsed_dates = [dt.replace(year=2024) for dt in parsed_dates]
                first_date = (
                    f"{parsed_dates[0].month}/{parsed_dates[0].day}/{parsed_dates[0].year}"
                    if len(parsed_dates) > 1
                    else None
                )
                last_date = f"{parsed_dates[-1].month}/{parsed_dates[-1].day}/{parsed_dates[-1].year}"
            data[PARAM_TO_COL["haul_date_first"]] = first_date
            data[PARAM_TO_COL["haul_date_last"]] = last_date

    for waste_type, units in [
        ("Manure", ["ton", "yard"]),
        ("Wastewater", ["gallon"]),
    ]:
        if not (txt := data.get(f"Method Used to Determine Volume of {waste_type}")):
            continue
        txt_str = str(txt)
        txt_lower = txt_str.lower()

        # Extract loads/hauls and per-load amounts (pattern: "X loads @ Y tons per load")
        if "per" in txt_lower:
            if (m := _NUMBER_HAULS_RE.search(txt_lower)) and (
                n := _parse_number(m.group(1))
            ) is not None:
                data[PARAM_TO_COL[f"{waste_type.lower()}_number_hauls"]] = str(
                    int(n) if n == int(n) else n
                )
            for unit_key in units:
                if (m := re.search(rf"(\d+(?:,\d+)?)\s*{unit_key}", txt_lower)) and (
                    v := _parse_number(m.group(1))
                ) is not None:
                    data[PARAM_TO_COL[f"{waste_type.lower()}_{unit_key}_per_haul"]] = v

        # Extract hours and GPM for wastewater (pattern: "X hours @ Y GPM")
        if waste_type == "Wastewater":
            # Hours: "X hours", "X hrs", including fractions like "11 1/2"
            if m := re.search(
                r"(\d+(?:\s*\d+/\d+)?(?:\.\d+)?)\s*(?:hours?|hrs?)\b",
                txt_str,
                re.IGNORECASE,
            ):
                hour_text = m.group(1).strip().replace(" ", "")  # "11 1/2" -> "111/2"
                if "/" in hour_text:
                    if frac_m := re.match(r"(\d+)(\d)/(\d+)", hour_text):  # "111/2"
                        whole, num, denom = frac_m.groups()
                        hours = float(whole) + float(num) / float(denom)
                    elif frac_m := re.match(r"(\d+)/(\d+)", hour_text):  # "1/2"
                        hours = float(frac_m.group(1)) / float(frac_m.group(2))
                    else:
                        hours = _parse_number(hour_text)
                else:
                    hours = _parse_number(hour_text)
                if hours is not None:
                    data[PARAM_TO_COL["wastewater_hours_pumped"]] = hours
            # GPM: "X GPM" or "X gallons per min"
            if m := re.search(
                r"(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:gpm|gallons per min)\b",
                txt_str,
                re.IGNORECASE,
            ):
                if (gpm := _parse_number(m.group(1))) is not None:
                    data[PARAM_TO_COL["wastewater_pumping_rate"]] = gpm

    # add is_pipeline column depending on whether "pipeline" is in manifest text
    data[PARAM_TO_COL["is_pipeline"]] = "pipeline" in manifest_text.lower()
    return data


def extract_manifests_from_txt(txt_path, *, backup_txt_path=None):
    pdf_stem = pdf_stem_from_txt_path(txt_path)
    result_text = read_txt(txt_path)

    nums, blocks, ranges, templates = identify_manifest_pages(result_text)
    if not nums:
        print(f"No manifests found in {txt_path}")
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

    for i, (block_text, (start_page, end_page)) in enumerate(
        zip(blocks, ranges), start=1
    ):
        manifest_text = clean_common_errors(block_text)
        manifest_template = templates[i - 1]

        # Handle multi-row table template (R5-2007-0035_one_page_2)
        if manifest_template == "R5-2007-0035_one_page_2":
            table_rows = _parse_hauling_table(manifest_text)

            if table_rows:
                # Extract base data once (all header/metadata fields)
                base_data = extract_manifest_fields(manifest_text, manifest_template)
                base_data["Source PDF"] = pdf_stem

                # Create one manifest entry per table row
                for j, row in enumerate(table_rows):
                    row_data = base_data.copy()

                    # Override hauling-specific fields from table
                    row_data[PARAM_TO_COL["haul_date"]] = row["haul_date"]

                    # Determine waste type based on units
                    if row["units"] and (
                        "ton" in row["units"].lower() or "yard" in row["units"].lower()
                    ):
                        row_data[PARAM_TO_COL["manure_amount"]] = row["amount"]
                        if "manure_solids_percent" in PARAM_TO_COL:
                            row_data[PARAM_TO_COL["manure_solids_percent"]] = row[
                                "moisture_pct"
                            ]
                    else:
                        row_data[PARAM_TO_COL["wastewater_amount"]] = row["amount"]

                    # Use letter suffix for manifest number (1a, 1b, 1c, etc.)
                    row_data["County"] = county
                    row_data["Consultant"] = template
                    row_data["Manifest Number"] = f"{i}{chr(97 + j)}"
                    row_data["Start Page"] = start_page
                    row_data["End Page"] = end_page
                    manifests.append(row_data)

                # Save files once for all rows (using base manifest number)
                with open(
                    os.path.join(output_dir, f"manifest_{i}.txt"), "w", encoding="utf-8"
                ) as f:
                    f.write(manifest_text)

                with fitz.open(original_pdf) as doc, fitz.open() as new_doc:
                    for p in range(start_page - 1, end_page):
                        if 0 <= p < len(doc):
                            new_doc.insert_pdf(doc, from_page=p, to_page=p)
                            if all_manifests_doc is not None:
                                all_manifests_doc.insert_pdf(
                                    doc, from_page=p, to_page=p
                                )
                    new_doc.save(os.path.join(output_dir, f"manifest_{i}.pdf"))

            continue  # Skip normal processing

        # Normal single-manifest processing
        data = extract_manifest_fields(manifest_text, manifest_template)
        data["Source PDF"] = pdf_stem
        data["County"] = county
        data["Consultant"] = template
        data["Manifest Number"] = i
        data["Start Page"] = start_page
        data["End Page"] = end_page
        manifests.append(data)

        # Save individual manifest txt
        with open(
            os.path.join(output_dir, f"manifest_{i}.txt"), "w", encoding="utf-8"
        ) as f:
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
            parameter = parameter[parameter.find("Farmer") :]
        # Remove trailing punctuation from Farmer (period, dash, etc.)
        if parameter.startswith("Farmer"):
            parameter = re.sub(r"^Farmer[\s.—-]+$", "Farmer", parameter)
        if "compost" in parameter.lower():
            return "Composting Facility"
    # TODO: expand
    return parameter


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
    if not all_stems:
        print("No OCR txt files found")
        return

    # Extract using the selected OCR source
    for stem in all_stems:
        # Always prioritize llmwhisperer if it exists, fallback to tesseract
        chosen = stems["llmwhisperer"].get(stem) or stems["tesseract"].get(stem)
        if not chosen:
            continue
        backup_txt = (
            stems["tesseract"].get(stem)
            if stem in stems["tesseract"]
            and chosen != stems["tesseract"].get(stem)
            and "tesseract" not in chosen
            else None
        )
        all_manifests.extend(
            extract_manifests_from_txt(chosen, backup_txt_path=backup_txt)
        )

    print(f"\nExtracted {len(all_manifests)} total manifests")

    df = pd.DataFrame(all_manifests)
    out_csv = "ca_cafo_compliance/outputs/extracted_manifests.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    for col in MANIFEST_COLUMNS:
        if col in df.columns:
            print(f"  {col}: {df[col].notna().sum()}/{len(df)} extracted")

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
        "ca_cafo_compliance/outputs/manifest_extraction_summary.csv", index=False
    )

    manure_cols = [manure_col]
    wastewater_cols = [wastewater_col]
    has_manure = df[manure_cols].notna().any(axis=1)
    has_wastewater = df[wastewater_cols].notna().any(axis=1)

    df["Manifest Type"] = "unknown"
    df.loc[has_manure & has_wastewater, "Manifest Type"] = "both"
    df.loc[has_manure & ~has_wastewater, "Manifest Type"] = "manure"
    df.loc[~has_manure & has_wastewater, "Manifest Type"] = "wastewater"

    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")

    print(f"\n  Manure manifests: {int(has_manure.sum())}")
    print(f"  Wastewater manifests: {int(has_wastewater.sum())}")
    print(f"  Both: {int((has_manure & has_wastewater).sum())}")
    print(f"  Unknown: {int((~has_manure & ~has_wastewater).sum())}")

    # Only count as geocoded if quantity is valid numeric (exclude N/A, text, etc.)

    dest_type_col = PARAM_TO_COL["destination_type_std"]
    print("\nManure-only summary by destination type:")
    print(
        df[has_manure]
        .groupby(dest_type_col)[manure_col]
        .apply(lambda x: to_numeric(x).sum())
    )
    print("\nWastewater-only summary by destination type:")
    print(
        df[has_wastewater]
        .groupby(dest_type_col)[wastewater_col]
        .apply(lambda x: to_numeric(x).sum())
    )

    # print breakdown of templates from all manifests
    print("\nTemplates breakdown:")
    print(df["Parameter Template"].value_counts())


def identify_files_to_delete():
    CUTOFF_TIME = datetime(2026, 1, 17, 14, 0, 0)

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

    # OCR outputs before time cutoff
    ocr_patterns = (
        "**/fitz_output/**/*.txt",
        "**/fitz_output/**/*.json",
        "**/tesseract_output/**/*.txt",
        "**/tesseract_output/**/*.json",
        "**/llmwhisperer_output/**/*.txt",
        "**/llmwhisperer_output/**/*.json",
    )
    ocr_files = [
        p
        for pat in ocr_patterns
        for p in g(pat)
        if not os.path.basename(p).startswith("manifest_")
    ]
    ocr_before_cutoff = sorted(
        [
            p
            for p in ocr_files
            if datetime.fromtimestamp(os.path.getmtime(p)) < CUTOFF_TIME
        ]
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

    # Optional: expand to include parent dirs that become empty
    # to_delete = set(empty_subfolders)
    # for d in sorted(empty_subfolders, key=lambda p: p.count(os.sep), reverse=True):
    #     cur = os.path.dirname(d)
    #     while cur and os.path.isdir(cur):
    #         if os.path.basename(cur) == "llmwhisperer_output":
    #             break
    #         try:
    #             remaining = [c for n in os.listdir(cur)
    #                         if (c := os.path.join(cur, n)) not in to_delete]
    #             if not remaining:
    #                 to_delete.add(cur)
    #                 cur = os.path.dirname(cur)
    #             else:
    #                 break
    #         except FileNotFoundError:
    #             break
    # empty_subfolders = sorted(to_delete, key=lambda p: p.count(os.sep), reverse=True)

    # Write delete lists
    delete_lists = {
        "delete_list_1_page.txt": one_page,
        "delete_list_ocr_before_cutoff.txt": ocr_before_cutoff,
        "delete_list_all_fitz.txt": engine_delete_lists["fitz"],
        "delete_list_all_tesseract.txt": engine_delete_lists["tesseract"],
        "delete_list_all_llmwhisperer.txt": engine_delete_lists["llmwhisperer"],
        "delete_list_empty_subfolders.txt": empty_subfolders,
    }

    for fname, items in delete_lists.items():
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(items) + "\n")

    # Deletion commands:
    #   while IFS= read -r f; do rm -f "$f"; done < ca_cafo_compliance/outputs/delete_list_all_fitz.txt
    #   while IFS= read -r f; do rm -f "$f"; done < ca_cafo_compliance/outputs/delete_list_all_tesseract.txt
    #   while IFS= read -r f; do rm -f "$f"; done < ca_cafo_compliance/outputs/delete_list_all_llmwhisperer.txt
    #   while IFS= read -r d; do rmdir "$d" 2>/dev/null; done < ca_cafo_compliance/outputs/delete_list_empty_subfolders.txt


if __name__ == "__main__":
    main()
    identify_files_to_delete()
