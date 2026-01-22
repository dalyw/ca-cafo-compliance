#!/usr/bin/env python3
import os
os.environ['HF_HUB_OFFLINE'] = '1' # TODO: test this line
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import glob
import csv
import json
import cv2
import re
import pymupdf as fitz
import pytesseract
import numpy as np
from PIL import Image
import io
from pdf2image import convert_from_path
import tempfile
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from helper_functions.read_report_helpers import clean_common_errors, YEARS, REGIONS

# Configuration
OCR_ENGINE = "marker"  # Default OCR engine
TEST_MODE = False  # Process only test files
MAX_PAGES = 999  # Max pages per document
FITZ_OUTPUT_FOLDER = "fitz_output"
_MARKER_MODELS = load_all_models()

GDRIVE_BASE = '/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/ca_cafo_manifests'
GDRIVE_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GDRIVE_DATA_DIR = os.path.join(GDRIVE_BASE_DIR, "data")

manifest_terms = [
                "hauler", 
                "destination",
                "name of dairy",
                "hauling",
                "attachment d",
                "method",
                "manifest"
            ]

def extract_specific_pages(pdf_path, pages, output_path=None):
    """Extract specific pages from a PDF to a new manifest file."""
    
    if not pages:
        return None
    
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()
    added_pages = 0
    
    for page_num in pages:
        if 0 < page_num <= len(doc):
            new_doc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
            added_pages += 1
    
    if output_path is None:
        # Create a temporary file path for the extracted page(s)
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        output_path = tmp_path
    
    # Avoid saving zero-page PDFs
    if added_pages == 0:
        new_doc.close()
        doc.close()
        print("zero-page pdf")
        return None
    
    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    
    return output_path

def is_low_quality_ocr(text):
    """Detect low-quality OCR text with common errors."""
    if not text or len(text) < 100:
        return True
    
    # Count suspicious patterns per 1000 characters
    sample = text[:5000]
    issues = 0
    
    # Check for common OCR errors
    issues += sample.count('6') / 10  # '6' often replaces 'e' 
    issues += sample.count('1') / 10  # '1' often replaces 'l' or 'i'
    issues += len([w for w in sample.split() if len(w) <= 2 and w.isalpha()]) / 5  # Too many 2-letter words
    issues += sum(1 for c in sample if not c.isprintable() and c not in '\n\t') / 5  # Unprintable chars
    
    # Check for missing spaces (words too long)
    avg_word_len = sum(len(w) for w in sample.split()) / max(len(sample.split()), 1)
    if avg_word_len > 12:
        issues += 10
    
    return issues > 20  # Threshold for "low quality"


def has_missing_manifest_quantities(text):
    """Detect if manifest form is present but quantity values are missing or unreadable."""
    
    text_lower = text.lower()
        
    if not any(term in text_lower for term in manifest_terms):
        return False # N/A for non-manifests
    
    # Look for quantity fields with actual numeric values
    manure_pattern = r'manure:?\s*[_\s]*(?:tons|cubic yards)?[_\s]*(\d+[,\d]*(?:\.\d+)?)'
    wastewater_pattern = r'(?:process\s+)?wastewater:?\s*[_\s]*(?:gallons)?[_\s]*(\d+[,\d]*(?:\.\d+)?)'
    
    has_manure_number = bool(re.search(manure_pattern, text_lower))
    has_wastewater_number = bool(re.search(wastewater_pattern, text_lower))
    
    has_amount_section = 'amount hauled' in text_lower # manifest keywords exist but no quantities
    
    return has_amount_section and not (has_manure_number or has_wastewater_number)


def extract_text_from_pdf(pdf_path, method="fitz", pages_to_process=None, max_pages=999):
    """Universal text extraction supporting fitz, marker, and tesseract."""
    
    if method == "fitz":
        doc = fitz.open(pdf_path)
        pages = [p - 1 for p in pages_to_process if 0 < p <= len(doc)] if pages_to_process else range(len(doc))
        all_text = [f"=== Page {p + 1} ===\n{doc[p].get_text()}" 
                    for p in pages if doc[p].get_text().strip()]
        doc.close()
        full_text = "\n\n".join(all_text)
        
        # Check quality and fall back to marker if poor
        if is_low_quality_ocr(full_text):
            # TODO: remove this if template detection suffices
            return {"result_text": full_text, "extraction_method": "fitz", "status": "low_quality", "should_retry": True}
        
        return {"result_text": full_text, "extraction_method": "fitz", "status": "success"}
    
    elif method == "marker":
        pages_list = pages_to_process or list(range(1, fitz.open(pdf_path).page_count + 1))
        all_page_texts = []
        
        for page_num in pages_list:
            temp_pdf = extract_specific_pages(pdf_path, [page_num])
            if temp_pdf:
                page_text, _, _ = convert_single_pdf(str(temp_pdf), _MARKER_MODELS, max_pages=1)
                all_page_texts.append(f"=== Page {page_num} ===\n{page_text}")
                os.remove(temp_pdf)
        
        full_text = "\n\n".join(all_page_texts)
        return {"result_text": full_text, "extraction_method": "marker", "status": "success"}
    
    elif method == "tesseract":
        images = convert_from_path(pdf_path, dpi=300)[:max_pages]
        all_text = []
        
        for image in images:
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            text = pytesseract.image_to_string(thresh, config='--oem 1 --psm 4 -c preserve_interword_spaces=1')
            cleaned = "\n".join("".join(c for c in clean_common_errors(line) if c.isprintable()) 
                              for line in text.splitlines())
            all_text.append(cleaned)
        
        return {"result_text": "\n\n".join(all_text), "extraction_method": "tesseract", "status": "success"}


def process_pdf(pdf_path, ocr_engine="marker", max_pages=999, process_only_manifests=False):
    """Process a single PDF file and extract text."""
    
    print(f"Processing {pdf_path}")
    
    pages_to_process = None
    if process_only_manifests:
        
        manifest_pages = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().lower()
            
            # If no embedded text, try quick OCR at low resolution
            if len(text.strip()) < 50:
                pix = page.get_pixmap(dpi=72)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, config='--psm 3').lower()
            
            # Check for manifest-related terms
            if any(term in text for term in manifest_terms):
                manifest_pages.append(page_num + 1)
                # Always include the next page (page 2 of manifest with method/certification)
                if page_num + 1 < len(doc):
                    manifest_pages.append(page_num + 2)
        
        doc.close()
        
        manifest_pages = sorted(set(manifest_pages)) # Remove duplicates
        if manifest_pages:
            print(f"  Found {len(manifest_pages)} manifest page(s): {manifest_pages}")
            pages_to_process = manifest_pages
        else:  # Save empty results
            print("  No manifest pages found, skipping")

            paths = get_output_paths(pdf_path, "fitz")
            os.makedirs(paths['dir'], exist_ok=True)

            with open(paths['txt'], "w", encoding="utf-8") as f:
                f.write("")
            
            with open(paths['json'], "w", encoding="utf-8") as f:
                json.dump({"status": "no_manifests_found"}, f, indent=2)
            
            return 
    
    # Try fitz first, fall back to OCR
    result = extract_text_from_pdf(pdf_path, "fitz", pages_to_process)
    
    result_text = result.get("result_text", "") if result else ""
    should_retry = False
    
    # Check if we should retry with OCR
    if not result or len(result_text) < 100:
        should_retry = True
    elif result.get("should_retry"):
        should_retry = True
        print(f"  Low quality fitz output, retrying with {ocr_engine}")
    elif has_missing_manifest_quantities(result_text):
        should_retry = True
        print(f"  Manifest quantities missing/unreadable in fitz output, retrying with {ocr_engine}")
    
    if should_retry:
        if not result or len(result_text) < 100:
            print(f"  Falling back to {ocr_engine}")
        result = extract_text_from_pdf(pdf_path, ocr_engine, pages_to_process, max_pages)
    
    # Save results
    method = result.get("extraction_method", ocr_engine)
    paths = get_output_paths(pdf_path, method)
    os.makedirs(paths['dir'], exist_ok=True)

    text_content = result.get("result_text", "")

    if isinstance(text_content, list):
        text_content = "\n".join(text_content)
    
    with open(paths['txt'], "w", encoding="utf-8") as f:
        f.write(text_content)
    
    with open(paths['json'], "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Extraction complete via {method}")


def get_output_paths(pdf_path, method):
    """Return all output paths for a given PDF and method."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    parent_dir = os.path.dirname(os.path.dirname(pdf_path))
    folder = FITZ_OUTPUT_FOLDER if method == "fitz" else f"{method}_output"
    output_dir = os.path.join(parent_dir, folder, pdf_name)
    output_GDRIVE_dir = output_dir.replace(GDRIVE_DATA_DIR, GDRIVE_BASE)
    os.makedirs(output_GDRIVE_dir, exist_ok=True)
    return {
        'dir': output_GDRIVE_dir,
        'txt': os.path.join(output_GDRIVE_dir, f"{pdf_name}.txt"),
        'json': os.path.join(output_GDRIVE_dir, f"{pdf_name}.json"),
    }


def is_processed(pdf_path, ocr_engine):
    """Check if PDF has already been processed with non-empty output."""
    paths = get_output_paths(pdf_path, "fitz")
    if os.path.exists(paths['txt']) and os.path.getsize(paths['txt']) > 0:
        return True
    paths = get_output_paths(pdf_path, ocr_engine)
    return os.path.exists(paths['txt']) and os.path.getsize(paths['txt']) > 0


def collect_pdf_files(years=None, regions=REGIONS):
    """Collect all PDF files from the data directory."""
    pdf_files = []
    if not years:
        years = YEARS
    if not regions:
        regions = REGIONS
    for year in years:
        GDRIVE_BASE = '/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/ca_cafo_manifests'
        base_path = GDRIVE_BASE + f"/{year}"
        print(base_path)
        for region in REGIONS:
            region_path = os.path.join(base_path, region)
            if not os.path.exists(region_path):
                print(f"no {region} path")
                continue
            
            for county in os.listdir(region_path):
                county_path = os.path.join(region_path, county)
                if not os.path.isdir(county_path):
                    print(f"no {county} path")
                    continue
                
                for template in os.listdir(county_path):
                    template_path = os.path.join(county_path, template)
                    if not os.path.isdir(template_path):
                        continue
                    
                    folder_path = os.path.join(template_path, "original")
                    if os.path.exists(folder_path):
                        pdf_files.extend(glob.glob(os.path.join(folder_path, "*.pdf")))
                        pdf_files.extend(glob.glob(os.path.join(folder_path, "*.PDF")))
    
    return pdf_files


def update_reports_available_csv():
    """Update reports_available.csv with PDF counts."""
    region_county_map = {
        "5F": ["kern"],
        "5S": ["fresno_madera", "kings", "tulare_west"],
        "5R": [],
    }
    
    base_path = GDRIVE_BASE + "/2023/R5"
    csv_path = "ca_cafo_compliance/data/reports_available.csv"
    
    if not os.path.exists(csv_path):
        return
    
    pdf_counts = {}
    for region, counties in region_county_map.items():
        total = 0
        for county in counties:
            county_path = os.path.join(base_path, county)
            for _, _, files in os.walk(county_path):
                total += sum(1 for f in files if f.lower().endswith(".pdf"))
        pdf_counts[region] = total
    
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["region"] in pdf_counts:
                row["acquired"] = str(pdf_counts[row["region"]])
            rows.append(row)
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["region", "acquired", "total"])
        writer.writeheader()
        writer.writerows(rows)


def main(ocr_engine=OCR_ENGINE, test_mode=TEST_MODE, max_pages=MAX_PAGES,
         process_only_manifests=False):
    
    update_reports_available_csv()
    
    # Collect and sort PDF files
    pdf_files = collect_pdf_files([2024], ["R5"])
    
    # Filter out already processed files
    files_to_process = [f for f in pdf_files if not is_processed(f, ocr_engine)]    
    print(f"{len(files_to_process)} of {len(pdf_files)} remaining")
    
    if test_mode:
        white_river = [f for f in files_to_process if "White River Dairy" in f]
        files_to_process = white_river[:1] if white_river else files_to_process[:1]
        print(f"Test mode: processing {len(files_to_process)} file(s)")
    
    if not files_to_process:
        print("No files to process")
        return
    else:  # Process files
        for pdf_path in files_to_process:
            process_pdf(
                pdf_path,
                ocr_engine=ocr_engine,
                max_pages=max_pages,
                process_only_manifests=process_only_manifests
            )
    
    print("OCR complete")


if __name__ == "__main__":
    
    main(
        ocr_engine="marker",
        test_mode=False,
        max_pages=999,
        process_only_manifests=True
    )
