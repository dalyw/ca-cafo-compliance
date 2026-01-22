#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import pymupdf as fitz
from helper_functions.read_report_helpers import (
    identify_manifest_pages,
    extract_address_from_section,
    format_address,
    parse_marker_table_address,
)

# Load manifest parameter configuration files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load manifest parameters
MANIFEST_PARAMS_DF = pd.read_csv(os.path.join(DATA_DIR, "manifest_parameters.csv"))
MANIFEST_PARAM_TYPES = dict(zip(MANIFEST_PARAMS_DF["parameter_key"], MANIFEST_PARAMS_DF["data_type"]))
MANIFEST_PARAM_DEFAULTS = dict(zip(MANIFEST_PARAMS_DF["parameter_key"], MANIFEST_PARAMS_DF["default"]))

# Load manifest parameter locations (extraction rules)
MANIFEST_LOCATIONS_DF = pd.read_csv(os.path.join(DATA_DIR, "manifest_parameter_locations.csv"))
PARAM_TO_COLUMN = MANIFEST_PARAMS_DF.set_index("parameter_key")["parameter_name"].to_dict()
MANIFEST_COLUMNS = list(PARAM_TO_COLUMN.values()) + ["Source PDF", "Manifest Number"]

def get_section_text(text, section_start, section_end=None):
    """
    Extract text between section markers.
    
    Args:
        text: Full manifest text
        section_start: Start marker string (case insensitive)
        section_end: Optional end marker string (case insensitive)
    
    Returns: Section text or full text if markers not found
    """
    text_lower = text.lower()
    start_idx = text_lower.find(section_start.lower())
    
    if start_idx == -1:
        return text
    
    if section_end:
        end_idx = text_lower.find(section_end.lower(), start_idx)
        if end_idx != -1:
            return text[start_idx:end_idx]
    
    return text[start_idx:]


def extract_value_from_row(text, row, ocr_format):
    """
    Extract a parameter value from text using the location configuration.
    
    Args:
        text: The manifest text (may be section-limited)
        row: A row from manifest_parameter_locations with extraction config
        ocr_format: 'fitz' or 'marker'
    
    Returns: Extracted value or None
    """
    # Get section text if section markers are defined
    section_start = row.get('section_start', '')
    section_end = row.get('section_end', '')
    
    if pd.notna(section_start) and section_start:
        section_text = get_section_text(text, section_start, section_end if pd.notna(section_end) else None)
    else:
        section_text = text
    
    # Get extraction parameters
    row_search_text = row.get('row_search_text', '')
    search_direction = row.get('search_direction', 'right')
    ignore_before = row.get('ignore_before', '')
    ignore_after = row.get('ignore_after', '')
    
    if pd.isna(row_search_text) or not row_search_text or row_search_text == 'NA':
        return None

    lines = section_text.split('\n')
    
    # Find the line containing the search text
    for i, line in enumerate(lines):
        if row_search_text.lower() in line.lower():
            value = None
            
            if search_direction == 'right':
                if ocr_format == 'marker' and '|' in line:
                    # Marker table format: find value in next cell
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    for j, cell in enumerate(cells):
                        if row_search_text.lower() in cell.lower():
                            # Check same cell after the label
                            idx = cell.lower().find(row_search_text.lower())
                            after = cell[idx + len(row_search_text):].strip()
                            after = after.lstrip(':').strip()
                            if after and after.lower() not in ['', 'na', 'none']:
                                value = after
                                break
                            # Try next cell
                            if j + 1 < len(cells):
                                next_cell = cells[j + 1]
                                if next_cell.lower() not in ['city', 'county', 'state', 'zip code', 'name', 'phone number']:
                                    value = next_cell
                            break
                else:
                    # Plain text format: value after label on same line
                    idx = line.lower().find(row_search_text.lower())
                    after = line[idx + len(row_search_text):].strip()
                    after = after.lstrip(':').strip()
                    if after:
                        value = after
            
            elif search_direction == 'below':
                # Value on next non-empty line
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not next_line.startswith('|--'):
                        # For marker format, extract from table cell
                        if ocr_format == 'marker' and '|' in next_line:
                            cells = [c.strip() for c in next_line.split('|') if c.strip()]
                            if cells:
                                value = cells[0]  # First non-empty cell
                        else:
                            value = next_line
                        break
            
            elif search_direction == 'left':
                # Value before the label on same line
                idx = line.lower().find(row_search_text.lower())
                before = line[:idx].strip()
                if before:
                    value = before
            
            # Apply ignore_before and ignore_after
            if value:
                if pd.notna(ignore_before) and ignore_before:
                    idx = value.find(str(ignore_before))
                    if idx != -1:
                        value = value[idx + len(str(ignore_before)):].strip()
                
                if pd.notna(ignore_after) and ignore_after:
                    idx = value.lower().find(str(ignore_after).lower())
                    if idx != -1:
                        value = value[:idx].strip()
            
            return value if value else None
    
    return None


def extract_structured_address(text, section_markers):
    """
    Extract address from manifest text by trying multiple section markers.
    Wrapper around shared extract_address_from_section helper.
    """
    for start_marker, end_marker in section_markers:
        address = extract_address_from_section(text, start_marker, end_marker)
        if address:
            return address
    return None



def process_manifest_with_params(manifest_text, manifest_num, source_pdf, text_file_path=None):
    """
    Extract manifest data using parameter-based extraction framework.
    
    Uses manifest_parameter_locations.csv to define extraction rules for each field,
    supporting both fitz (plain text) and marker (markdown table) OCR formats.
    
    This is the primary extraction method that consolidates extraction logic
    and allows easier maintenance/tuning of extraction patterns.
    
    Args:
        manifest_text: The full manifest text (may span multiple pages)
        manifest_num: The manifest number (1-indexed) within the PDF
        source_pdf: Path to the source PDF file
        text_file_path: Path to the OCR text file (used to determine OCR format from folder)
    
    Returns:
        Dictionary with extracted manifest data matching MANIFEST_COLUMNS
    """
    data = {col: None for col in MANIFEST_COLUMNS}
    data["Source PDF"] = source_pdf
    data["Manifest Number"] = manifest_num
    
    # Detect OCR format from folder path and template type from text content
    ocr_format = 'marker' if 'marker' in text_file_path else 'fitz'

    text_upper = manifest_text.upper()
    
    # ATTACHMENT D is a distinct R5 form variant
    template = 'manifest_attachment_d' if 'ATTACHMENT D' in text_upper else 'manifest_r5'
    
    # Extract each field using parameter locations

    for param_key, column_name in PARAM_TO_COLUMN.items():
        matching = MANIFEST_LOCATIONS_DF[
            (MANIFEST_LOCATIONS_DF['template'] == template) &
            (MANIFEST_LOCATIONS_DF['ocr_format'] == ocr_format) &
            (MANIFEST_LOCATIONS_DF['parameter_key'] == param_key)
            # (MANIFEST_LOCATIONS_DF['template'].astype(str).str.strip() == template.strip()) &
            # (MANIFEST_LOCATIONS_DF['ocr_format'].astype(str).str.strip() == ocr_format.strip()) &
            # (MANIFEST_LOCATIONS_DF['parameter_key'].astype(str).str.strip() == param_key.strip())    
        ]
        # Optionally, print debug info if still empty
        if matching.empty:
            print(f"No match for: template='{template}', ocr_format='{ocr_format}', param_key='{param_key}'")
            print("Unique templates:", MANIFEST_LOCATIONS_DF['template'].unique())
            print("Unique ocr_formats:", MANIFEST_LOCATIONS_DF['ocr_format'].unique())
            print("Unique parameter_keys:", MANIFEST_LOCATIONS_DF['parameter_key'].unique())
            data[column_name] = None
            continue
        row = matching.iloc[0]
        value = extract_value_from_row(manifest_text, row, ocr_format)
        # Special handling for addresses - use dedicated extraction if parameter-based fails
        if value is None and param_key.endswith('_address'):
            if param_key == 'origin_dairy_address':
                value = extract_structured_address(manifest_text, [
                    ("Facility Address:", "Contact Person"),
                    ("OPERATOR INFORMATION", "PROCESS WASTEWATER HAULER"),
                    ("OPERATOR INFORMATION", "MANURE HAULER"),
                ])
                if not value:
                    print('no structured address')
                    value = extract_address_from_section(manifest_text, "Facility Address:", "Contact Person")
            elif param_key == 'hauler_address':
                value = extract_structured_address(manifest_text, [
                    ("Address of Hauling Company/Person:", "Contact Person"),
                    ("MANURE HAULER INFORMATION", "DESTINATION INFORMATION"),
                    ("PROCESS WASTEWATER HAULER INFORMATION", "DESTINATION INFORMATION"),
                ])
            elif param_key == 'destination_address':
                value = extract_structured_address(manifest_text, [
                    ("Destination Address or Assessor", "Last date hauled"),
                    ("Destination Address or Assessor", "Street and nearest"),
                    ("DESTINATION INFORMATION", "Last date hauled"),
                ])
        
        data[column_name] = value
    
    return data


def extract_manifests_from_file(text_file, pdf_name):
    """Extract all manifests from a text output file."""
    with open(text_file, 'r', encoding='utf-8') as f:
        result_text = f.read()
    
    manifest_indices, manifest_blocks, manifest_page_ranges = identify_manifest_pages(result_text)
    
    if not manifest_indices:
        return []
    
    print(f"  {len(manifest_indices)} manifests in {pdf_name}")
    
    output_dir = os.path.dirname(text_file)
    original_pdf = os.path.join(os.path.dirname(os.path.dirname(output_dir)), "original", f"{os.path.basename(output_dir)}.pdf")
    manifests = []
    
    for i, (manifest_text, (start_page, end_page)) in enumerate(zip(manifest_blocks, manifest_page_ranges)):
        manifest_num = i + 1
        manifests.append(process_manifest_with_params(manifest_text, manifest_num, pdf_name, text_file))
        
        with open(os.path.join(output_dir, f"manifest_{manifest_num}.txt"), 'w', encoding='utf-8') as f:
            f.write(manifest_text)
        
        if start_page and end_page:
            with fitz.open(original_pdf) as doc, fitz.open() as new_doc:
                for page_num in range(start_page - 1, end_page):
                    if 0 <= page_num < len(doc):
                        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                if new_doc.page_count > 0:
                    new_doc.save(os.path.join(output_dir, f"manifest_{manifest_num}.pdf"))
    
    return manifests


def main():
    """Main function to extract all manifests from OCR output."""
    
    # Search for both fitz_output and marker_output folders
    # Pattern for: data/year/region/county/template/{engine}_output/pdf_name/pdf_name.txt
    output_folders = ["fitz_output", "marker_output", "tesseract_output"]
    
    file_extension = ".txt"
    all_manifests = []
    
    print(f"\nExtracting manifests from OCR output...")
    
    for folder_name in output_folders:
        file_pattern = f"ca_cafo_compliance/data/2024/R5/**/{folder_name}/**/*{file_extension}"
        output_files = glob.glob(file_pattern, recursive=True)
        
        # Filter out manifest_*.txt files (our own output)
        output_files = [f for f in output_files if not os.path.basename(f).startswith('manifest_')]
        
        if output_files:
            print(f"\nFound {len(output_files)} files in {folder_name}")
            if len(output_files) <= 3:
                for f in output_files:
                    print(f"  {f}")
            else:
                print(f"  Example: {output_files[0]}")
        
        for output_file in output_files:
            # Get the PDF name from the parent folder
            parent_dir = os.path.dirname(output_file)
            pdf_name = os.path.basename(parent_dir)
            
            manifests = extract_manifests_from_file(output_file, pdf_name)
            all_manifests.extend(manifests)
    
    if not all_manifests:
        print("No manifests found")
        return
    
    print(f"\nExtracted {len(all_manifests)} total manifests")
    
    # Save to CSV
    df = pd.DataFrame(all_manifests)
    csv_output = "ca_cafo_compliance/outputs/extracted_manifests.csv"
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    df.to_csv(csv_output, index=False)
    print(f"Saved to {csv_output}")
    
    for col in MANIFEST_COLUMNS:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(df)} extracted")

    # Compute extraction fractions
    n_total = len(df)
    n_manure = df['Total Manure Amount (tons)'].notnull().sum()
    n_wastewater = df['Total Process Wastewater Exports (Gallons)'].notnull().sum()
    frac_manure = n_manure / n_total if n_total else 0
    frac_wastewater = n_wastewater / n_total if n_total else 0

    summary_row = {
        'n_total': n_total,
        'n_manure': n_manure,
        'frac_manure': frac_manure,
        'n_wastewater': n_wastewater,
        'frac_wastewater': frac_wastewater
    }

    # Save summary CSV
    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv("ca_cafo_compliance/outputs/manifest_extraction_summary.csv", index=False)

    # Classify manifests by type
    manure_cols = ['Manure Haul Type', 'Total Manure Amount (tons)', 'Method Used to Determine Manure Volume']
    wastewater_cols = ['Wastewater Haul Type', 'Total Process Wastewater Exports (Gallons)', 'Method Used to Determine Wastewater Volume']

    has_manure = df[manure_cols].notna().any(axis=1)
    has_wastewater = df[wastewater_cols].notna().any(axis=1)

    print(f"  Manure manifests: {int(has_manure.sum())}")
    print(f"  Wastewater manifests: {int(has_wastewater.sum())}")
    print(f"  Both: {int((has_manure & has_wastewater).sum())}")
    print(f"  Unknown: {int((~has_manure & ~has_wastewater).sum())}")

    # Separate totals by destination type for each manifest category
    

    def to_numeric_series(series):
        return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')

    df_manure = df[has_manure]
    df_wastewater = df[has_wastewater]
    dest_col = 'Destination Type'

    print("\nManure-only summary by destination type:")
    manure_by_dest = df_manure.groupby(dest_col)['Total Manure Amount (tons)'].apply(lambda x: to_numeric_series(x).sum())
    print(manure_by_dest)

    print("\nWastewater-only summary by destination type:")
    wastewater_by_dest = df_wastewater.groupby(dest_col)['Total Process Wastewater Exports (Gallons)'].apply(lambda x: to_numeric_series(x).sum())
    print(wastewater_by_dest)
    df_unknown = df[~has_manure & ~has_wastewater]
    if not df_unknown.empty:
        print("Unknown examples:")
        cols_to_show = ['Source PDF', 'Manifest Number', 'Haul Date']
        print(df_unknown[cols_to_show].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
