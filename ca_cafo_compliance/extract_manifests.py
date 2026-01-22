#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import pymupdf as fitz
from helper_functions.read_report_helpers import (
    extract_address_from_section,
    is_address_label,
    clean_common_errors,
    TABLE_CELL_LABELS,
    find_parameter_value
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MANIFEST_PARAMS_DF = pd.read_csv(os.path.join(DATA_DIR, "manifest_parameters.csv"))
MANIFEST_PARAM_TYPES = dict(zip(MANIFEST_PARAMS_DF["parameter_key"], MANIFEST_PARAMS_DF["data_type"]))
MANIFEST_PARAM_DEFAULTS = dict(zip(MANIFEST_PARAMS_DF["parameter_key"], MANIFEST_PARAMS_DF["default"]))
MANIFEST_LOCATIONS_DF = pd.read_csv(os.path.join(DATA_DIR, "manifest_parameter_locations.csv"))
PARAM_TO_COLUMN = MANIFEST_PARAMS_DF.set_index("parameter_key")["parameter_name"].to_dict()
MANIFEST_COLUMNS = list(PARAM_TO_COLUMN.values()) + ["Source PDF", "Manifest Number"]

# Cleanup patterns
PLACEHOLDER_PATTERNS = [
    r"[_\s-]+",
    r"/\s*(Other|Broker|Farmer)\s*\(identify\)",
    r"please circle one",
    r"^[_\s-]+\(.*\)[_\s-]*$",
    r'!\[.*\.(png|jpg|jpeg|gif)\]',
    r'destination address or',
    r"assessor'?s parcel number",
]
DEST_TYPE_PLACEHOLDERS = [
    r'destination address or', r'destination address', r"assessor'?s parcel number",
    r'parcel number', r'apn:?', r'number and street', r'city', r'zip code',
    r'phone number', r'contact information', r'contact person', r'name',
    r'address', r'^or\s*$', r'^\s*$',
]
APN_PATTERNS = [r'APN:\s*[^\s,]+', r'\b\d{3,4}-\d{3,4}-\d{3,4}-\d{3,4}\b']


def get_section_text(text, section_start, section_end=None):
    """Extract text between section markers."""
    text_lower = text.lower()
    start_idx = text_lower.find(section_start.lower())
    if start_idx == -1:
        return text
    if section_end:
        end_idx = text_lower.find(section_end.lower(), start_idx)
        if end_idx != -1:
            return text[start_idx:end_idx]
    return text[start_idx:]


def _is_placeholder_value(value):
    """Check if value is a placeholder/label."""
    if value is None:
        return True
    cleaned = str(value).strip()
    if not cleaned or cleaned.endswith(":"):
        return True
    if re.search(r"\d", cleaned) is None and re.search(r"[A-Za-z]", cleaned) is None:
        return True
    if cleaned.lower() in TABLE_CELL_LABELS or is_address_label(cleaned):
        return True
    
    # Check all placeholder patterns
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return True
    
    # Check for label concatenations
    label_words = ['name', 'number', 'street', 'city', 'zip', 'code', 'phone', 'address', 'parcel', 'assessor', 'destination']
    words = cleaned.lower().split()
    return len(words) >= 3 and sum(1 for w in words if w in label_words) >= 3


def _clean_value(value, is_numeric=False, remove_apn=False, is_dest_type=False):
    """Unified value cleaning."""
    if value is None:
        return None
    
    value_str = str(value).strip()
    
    # Numeric cleaning
    if is_numeric:
        match = re.search(r"[\d][\d,.\s_]*", value_str)
        if match:
            normalized = re.sub(r"[,\s]", "", match.group(0))
            normalized = normalized.replace("_", ".", 1) if re.match(r"^\d+_\d+$", normalized) else normalized.replace("_", "")
            return normalized
        return value_str
    
    # Remove APN patterns
    if remove_apn:
        for pattern in APN_PATTERNS:
            value_str = re.sub(pattern, '', value_str, flags=re.IGNORECASE)
        value_str = re.sub(r'\*\*', '', value_str)  # Remove markdown
        value_str = re.sub(r'\s+', ' ', value_str).strip()
    
    # Destination type validation
    if is_dest_type:
        for pattern in DEST_TYPE_PLACEHOLDERS:
            if re.search(pattern, value_str, re.IGNORECASE):
                return None
        if _is_placeholder_value(value_str):
            return None
    
    return value_str if value_str else None


def identify_manifest_pages(result_text):    
    def has_manifest_header(text):
        """Check if text contains any manifest header pattern."""
        text_upper = text.upper()
        return any(h in text_upper for h in ["TRACKING", "MANIFEST", "ATTACHMENT"])
    
    # Check for page delimiter format (=== Page X ===)
    page_delimiter_pattern = r'=== Page (\d+) ==='
    page_matches = list(re.finditer(page_delimiter_pattern, result_text))
    
    if page_matches:
        # Parse pages with delimiters
        pages = {}
        for i, match in enumerate(page_matches):
            page_num = int(match.group(1))
            start_pos = match.end()
            if i + 1 < len(page_matches):
                end_pos = page_matches[i + 1].start()
            else:
                end_pos = len(result_text)
            pages[page_num] = result_text[start_pos:end_pos].strip()
        
        # Find manifests by looking for the header + OPERATOR INFORMATION
        manifest_blocks = []
        manifest_starts = []
        manifest_page_ranges = []
        
        sorted_pages = sorted(pages.keys())
        i = 0
        manifest_num = 0
        processed_pages = set()  # Track which pages we've already used
        
        while i < len(sorted_pages):
            page_num = sorted_pages[i]
            if page_num in processed_pages:
                i += 1
                continue
                
            page_text = pages[page_num]
            
            # Skip false positive page
            if "REQUIRED ATTACHMENTS" in page_text.upper():
                i += 1
                continue
            
            # Check if this page starts a manifest (has header + operator info)
            if has_manifest_header(page_text):
                manifest_num += 1
                manifest_starts.append(manifest_num)
                
                # Look for the next page to see if it's the AMOUNT HAULED page
                combined_text = page_text
                end_page = page_num
                processed_pages.add(page_num)
                
                # Check if next page(s) contain the AMOUNT HAULED sections
                next_idx = i + 1
                while next_idx < len(sorted_pages):
                    next_page_num = sorted_pages[next_idx]
                    if next_page_num in processed_pages:
                        next_idx += 1
                        continue
                    
                    next_page_text = pages[next_page_num]
                    next_page_upper = next_page_text.upper()
                    
                    combined_text += "\n\n" + next_page_text
                    end_page = next_page_num
                    processed_pages.add(next_page_num)
                    break
                
                manifest_blocks.append(combined_text)
                manifest_page_ranges.append((page_num, end_page))
            
            i += 1
        
        return manifest_starts, manifest_blocks, manifest_page_ranges
    else:
        print("NO HEADERS")
        return [], [result_text], []


def extract_value_from_row(text, row, ocr_format):
    """Extract parameter value from text using location config."""
    # Get section
    section_start = row.get('section_start', '')
    section_end = row.get('section_end', '')
    section_text = get_section_text(text, section_start, section_end if pd.notna(section_end) else None) if pd.notna(section_start) and section_start else text
    
    # Get extraction params
    row_search_text = row.get('row_search_text', '')
    search_direction = row.get('search_direction', 'right')
    ignore_before = row.get('ignore_before', '')
    ignore_after = row.get('ignore_after', '')
    item_order = int(row.get("item_order")) if pd.notna(row.get("item_order")) and row.get("item_order") not in ["", "NA"] else None
    
    if pd.isna(row_search_text) or not row_search_text or row_search_text == 'NA':
        return None
    
    lines = section_text.split('\n')
    
    # Find matching line
    for i, line in enumerate(lines):
        if row_search_text.lower() not in line.lower():
            continue
        
        value = None
        
        # Direction-based extraction
        if search_direction == 'right':
            if ocr_format == 'marker' and '|' in line:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                for j, cell in enumerate(cells):
                    if row_search_text.lower() in cell.lower():
                        idx = cell.lower().find(row_search_text.lower())
                        after = cell[idx + len(row_search_text):].strip().lstrip(':').strip()
                        if after and after.lower() not in ['', 'na', 'none']:
                            value = after
                            break
                        if j + 1 < len(cells) and cells[j + 1].lower() not in ['city', 'county', 'state', 'zip code', 'name', 'phone number']:
                            value = cells[j + 1]
                        break
            else:
                idx = line.lower().find(row_search_text.lower())
                after = line[idx + len(row_search_text):].strip().lstrip(':').strip()
                if not after and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not _is_placeholder_value(next_line) and not next_line.endswith(":") and not is_address_label(next_line):
                        after = next_line
                value = after if after else None
        
        elif search_direction in ['below', 'above']:
            candidates = []
            range_iter = range(i + 1, min(i + 50, len(lines))) if search_direction == 'below' else range(i - 1, max(i - 50, -1), -1)
            
            for j in range_iter:
                next_line = lines[j].strip()
                if not next_line or next_line.startswith('|--') or next_line.endswith(":") or is_address_label(next_line):
                    continue
                if next_line.lower() in TABLE_CELL_LABELS or _is_placeholder_value(next_line):
                    continue
                
                if ocr_format == 'marker' and '|' in next_line:
                    cells = [c.strip() for c in next_line.split('|') if c.strip()]
                    for cell in cells:
                        if cell and cell.lower() not in TABLE_CELL_LABELS and not is_address_label(cell) and not _is_placeholder_value(cell):
                            candidates.append(cell)
                            break
                else:
                    candidates.append(next_line)
                
                if item_order is None and candidates:
                    break
            
            if candidates:
                value = candidates[0] if item_order is None else candidates[item_order] if 0 <= (item_order or 0) < len(candidates) else candidates[-1] if item_order == -1 else None
        
        elif search_direction == 'left':
            idx = line.lower().find(row_search_text.lower())
            if idx > 0:
                before = line[:idx].strip().rstrip(':').strip()
                if before and not _is_placeholder_value(before):
                    value = before
            if not value and i > 0:
                prev_line = lines[i - 1].strip()
                if prev_line and not _is_placeholder_value(prev_line) and (re.search(r'\d', prev_line) or len(prev_line.split()) <= 5):
                    if not prev_line.endswith(":") and not is_address_label(prev_line):
                        value = prev_line
        
        # Apply ignore filters
        if value:
            if pd.notna(ignore_before) and ignore_before:
                idx = value.lower().find(str(ignore_before).lower())
                value = value[idx + len(str(ignore_before)):].strip() if idx != -1 else value
            if pd.notna(ignore_after) and ignore_after:
                idx = value.lower().find(str(ignore_after).lower())
                value = value[:idx].strip() if idx != -1 else value
            # Remove APN patterns
            for pattern in APN_PATTERNS:
                value = re.sub(pattern, '', value, flags=re.IGNORECASE)
            value = re.sub(r'\s+', ' ', value).strip()
        
        return value if value else None
    
    return None


def extract_structured_address(text, section_markers):
    """Extract address using multiple section markers."""
    for start_marker, end_marker in section_markers:
        address = extract_address_from_section(text, start_marker, end_marker)
        if address:
            return address
    return None


def extract_manifests_from_file(text_file, pdf_name):
    """Extract all manifests from OCR output file."""
    with open(text_file, 'r', encoding='utf-8') as f:
        result_text = f.read()
    
    manifest_indices, manifest_blocks, manifest_page_ranges = identify_manifest_pages(result_text)
    if not manifest_indices:
        return []
    
    print(f"  {len(manifest_indices)} manifests in {pdf_name}")
    
    output_dir = os.path.dirname(text_file)
    original_pdf = os.path.join(os.path.dirname(os.path.dirname(output_dir)), "original", f"{os.path.basename(output_dir)}.pdf")
    manifests = []
    ocr_format = 'marker' if 'marker' in text_file else 'fitz'
    
    for i, (manifest_text, (start_page, end_page)) in enumerate(zip(manifest_blocks, manifest_page_ranges)):
        manifest_num = i + 1
        data = {col: None for col in MANIFEST_COLUMNS}
        data["Source PDF"] = pdf_name
        data["Manifest Number"] = manifest_num
        
        manifest_text = clean_common_errors(manifest_text)
        template = 'manifest_attachment_d' if 'ATTACHMENT D' in manifest_text.upper() else 'manifest_r5'
        
        # Extract each field
        for param_key, column_name in PARAM_TO_COLUMN.items():
            matching = MANIFEST_LOCATIONS_DF[
                (MANIFEST_LOCATIONS_DF['template'] == template) &
                (MANIFEST_LOCATIONS_DF['ocr_format'] == ocr_format) &
                (MANIFEST_LOCATIONS_DF['parameter_key'] == param_key)
            ]
            if matching.empty:
                continue

            address_markers = {
                'origin_dairy_address': [("Facility Address:", "Contact Person"), ("OPERATOR INFORMATION", "PROCESS WASTEWATER HAULER"), ("OPERATOR INFORMATION", "MANURE HAULER")],
                'hauler_address': [("Address of Hauling Company/Person:", "Contact Person"), ("MANURE HAULER INFORMATION", "DESTINATION INFORMATION"), ("PROCESS WASTEWATER HAULER INFORMATION", "DESTINATION INFORMATION")],
                'destination_address': [("Destination Address or Assessor", "Last date hauled"), ("Destination Address or Assessor", "Street and nearest"), ("DESTINATION INFORMATION", "Last date hauled")],
            }

            for param_key, column_name in PARAM_TO_COLUMN.items():
                matching = MANIFEST_LOCATIONS_DF[
                    (MANIFEST_LOCATIONS_DF['template'] == template) &
                    (MANIFEST_LOCATIONS_DF['ocr_format'] == ocr_format) &
                    (MANIFEST_LOCATIONS_DF['parameter_key'] == param_key)
                ]
                if matching.empty:
                    continue

                row = matching.iloc[0]
                value = find_parameter_value(manifest_text, row, MANIFEST_PARAM_TYPES, MANIFEST_PARAM_DEFAULTS)

                # destination_type: normalize to allowed set if present
                if param_key == 'destination_type' and value:
                    valid = ['broker', 'farmer', 'composting facility', 'digester', 'other']
                    val_lower = str(value).lower()
                    value = next((v.title() if v != 'other' else 'Other' for v in valid if v in val_lower), value)

                # address: if still empty, try structured markers
                if value is None and param_key.endswith('_address'):
                    value = extract_structured_address(manifest_text, address_markers.get(param_key, []))

                value = _clean_value(
                    value,
                    is_numeric=MANIFEST_PARAM_TYPES.get(param_key) == "numeric",
                    remove_apn=(param_key == 'destination_address'),
                    is_dest_type=(param_key == 'destination_type'),
                )
                data[column_name] = value

            # Apply cleaning
            is_numeric = MANIFEST_PARAM_TYPES.get(param_key) == "numeric"
            remove_apn = param_key == 'destination_address'
            is_dest_type = param_key == 'destination_type'
            value = _clean_value(value, is_numeric, remove_apn, is_dest_type)
            
            data[column_name] = value
        
        manifests.append(data)
        
        # Save individual manifest
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
    """Extract manifests from OCR output."""
    output_folders = ["fitz_output", "marker_output", "tesseract_output"]
    all_manifests = []
    
    print(f"\nExtracting manifests from OCR output...")
    
    for folder_name in output_folders:
        file_pattern = f"ca_cafo_compliance/data/2024/R5/**/{folder_name}/**/*.txt"
        output_files = [f for f in glob.glob(file_pattern, recursive=True) if not os.path.basename(f).startswith('manifest_')]
        
        if output_files:
            print(f"\nFound {len(output_files)} files in {folder_name}")
            print(f"  Example: {output_files[0]}" if len(output_files) > 3 else '\n'.join(f"  {f}" for f in output_files))
        
        for output_file in output_files:
            pdf_name = os.path.basename(os.path.dirname(output_file))
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
        print(f"  {col}: {df[col].notna().sum()}/{len(df)} extracted")
    
    # Summary stats
    n_total = len(df)
    n_manure = df['Total Manure Amount (tons)'].notnull().sum()
    n_wastewater = df['Total Process Wastewater Exports (Gallons)'].notnull().sum()
    
    summary_df = pd.DataFrame([{
        'n_total': n_total,
        'n_manure': n_manure,
        'frac_manure': n_manure / n_total if n_total else 0,
        'n_wastewater': n_wastewater,
        'frac_wastewater': n_wastewater / n_total if n_total else 0
    }])
    summary_df.to_csv("ca_cafo_compliance/outputs/manifest_extraction_summary.csv", index=False)
    
    # Classify manifests
    manure_cols = ['Manure Haul Type', 'Total Manure Amount (tons)', 'Method Used to Determine Manure Volume']
    wastewater_cols = ['Wastewater Haul Type', 'Total Process Wastewater Exports (Gallons)', 'Method Used to Determine Wastewater Volume']
    has_manure = df[manure_cols].notna().any(axis=1)
    has_wastewater = df[wastewater_cols].notna().any(axis=1)
    
    print(f"\n  Manure manifests: {int(has_manure.sum())}")
    print(f"  Wastewater manifests: {int(has_wastewater.sum())}")
    print(f"  Both: {int((has_manure & has_wastewater).sum())}")
    print(f"  Unknown: {int((~has_manure & ~has_wastewater).sum())}")
    
    # Totals by destination
    to_numeric = lambda s: pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')
    
    print("\nManure-only summary by destination type:")
    print(df[has_manure].groupby('Destination Type')['Total Manure Amount (tons)'].apply(lambda x: to_numeric(x).sum()))
    
    print("\nWastewater-only summary by destination type:")
    print(df[has_wastewater].groupby('Destination Type')['Total Process Wastewater Exports (Gallons)'].apply(lambda x: to_numeric(x).sum()))
    
    df_unknown = df[~has_manure & ~has_wastewater]
    if not df_unknown.empty:
        print("\nUnknown examples:")
        print(df_unknown[['Source PDF', 'Manifest Number', 'Haul Date']].head(5).to_string(index=False))


if __name__ == "__main__":
    main()