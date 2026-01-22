#!/usr/bin/env python3
import os

# # Set high-quality rendering settings BEFORE importing marker
# # These control the DPI/resolution marker uses internally
# os.environ.setdefault("PDFTEXT_CPU_WORKERS", "2")  # Parallel CPU workers
# os.environ.setdefault("SURYA_DET_DPI", "288")  # Detection DPI (default ~96)
# os.environ.setdefault("SURYA_LAYOUT_DPI", "288")  # Layout detection DPI
# os.environ.setdefault("SURYA_OCR_DPI", "288")  # OCR DPI for text recognition

# # Force all Surya models to use MPS (Apple Silicon GPU)
# os.environ.setdefault("TORCH_DEVICE", "mps")
# os.environ.setdefault("DETECTOR_DEVICE", "mps")
# os.environ.setdefault("LAYOUT_DEVICE", "mps")
# os.environ.setdefault("ORDER_DEVICE", "mps")
# os.environ.setdefault("RECOGNITION_DEVICE", "mps")
# os.environ.setdefault("TEXIFY_DEVICE", "mps")

# # Force OCR on all pages (don't rely on embedded text for scanned docs)
# os.environ.setdefault("OCR_ALL_PAGES", "true")

import glob
import csv
import json
import cv2
import pymupdf as fitz
import pytesseract
import numpy as np
from PIL import Image
import io
from pdf2image import convert_from_path
# from marker.convert import convert_single_pdf
# from marker.models import load_all_models
from helper_functions.read_report_helpers import clean_common_errors, YEARS, REGIONS

# Configuration
OCR_ENGINE = "marker"  # Default OCR engine
TEST_MODE = False  # Process only test files
MAX_PAGES = 999  # Max pages per document
FITZ_OUTPUT_FOLDER = "fitz_output"
_MARKER_MODELS = None  # Lazy-loaded marker models cached per process

# Required fields to validate manifest extraction quality
MANIFEST_REQUIRED_TERMS = [
    # At least one of these header patterns must be present
    # Note: we'll check for ANY of these, not ALL
]

# Header patterns - need at least one
MANIFEST_HEADER_PATTERNS = [
    "Manure / Process Wastewater Tracking Manifest",
    "Manure/Process Wastewater Tracking Manifest",  # No spaces around slash
    "ATTACHMENT D",  # R5 form variant
    "Tracking Manifest",  # Partial match
]

# Section patterns - need OPERATOR INFORMATION or similar
MANIFEST_SECTION_PATTERNS = [
    "OPERATOR INFORMATION",
    "Operator Information",
    "Name of Operator",
    "Name of Dairy Facility",
]
MANIFEST_VALIDATION_TERMS = [
    # Need at least some of these to consider extraction successful
    "Name of Dairy Facility",
    "Facility Address",
    "MANURE HAULER INFORMATION", 
    "DESTINATION INFORMATION",
    "MANURE AMOUNT HAULED",
]


def save_ocr_results(output_dir, pdf_name, text_content, json_data):
    """Save OCR results to text and JSON files."""
    text_file = os.path.join(output_dir, f"{pdf_name}.txt")
    json_file = os.path.join(output_dir, f"{pdf_name}.json")
    
    if isinstance(text_content, list):
        text_content = "\n".join(text_content)
    
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text_content)
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def scan_pdf_for_manifest_pages(pdf_path, search_terms=None):
    """
    Quickly scan a PDF for pages containing manifest-related keywords.
    Uses PyMuPDF for fast text extraction (embedded text layer).
    Falls back to quick low-res OCR if no text found.
    
    Args:
        pdf_path: Path to the PDF file
        search_terms: List of terms to search for (default: manifest-related terms)
    
    Returns:
        List of 1-indexed page numbers containing manifest keywords, or None if scan fails
    """
    if search_terms is None:
        # Include "attachment d" for R5 manifest forms that use that header
        search_terms = ["manifest", "hauler", "exporter", "importer", "recipient", 
                        "attachment d", "tracking manifest", "existing milk cow"]
    
    
    manifest_pages = []
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    for page_num in range(total_pages):
        page = doc[page_num]
        
        # Try to get embedded text first (very fast)
        text = page.get_text().lower()
        
        # If no embedded text, try quick OCR at low resolution
        if len(text.strip()) < 50:
            print("trying quick ocr")
            # Render at low DPI for quick scan
            pix = page.get_pixmap(dpi=72)
            
            # Use tesseract for quick scan
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, config='--psm 3').lower()
        
        # Check for manifest-related terms
        for term in search_terms:
            if term.lower() in text:
                manifest_pages.append(page_num + 1)  # 1-indexed
                break
    
    doc.close()
    return manifest_pages


def extract_specific_pages(pdf_path, pages, output_path=None):
    """
    Extract specific pages from a PDF to a new PDF file.
    
    Args:
        pdf_path: Source PDF path
        pages: List of 1-indexed page numbers to extract
        output_path: Output PDF path (default: temp file)
    
    Returns:
        Path to the extracted PDF, or None on failure
    """
    
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
        output_path = pdf_path.replace(".pdf", "_manifest_pages.pdf").replace(".PDF", "_manifest_pages.pdf")
    
    # Avoid saving zero-page PDFs
    if added_pages == 0:
        new_doc.close()
        doc.close()
        return None
    
    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    
    return output_path
    

def validate_manifest_extraction(text):
    """
    Check if extracted text contains the required manifest fields.
    
    Returns:
        Tuple of (is_valid, match_count, total_validation_terms)
    """
    text_upper = text.upper()
    
    # Check for at least one header pattern
    has_header = any(pattern.upper() in text_upper for pattern in MANIFEST_HEADER_PATTERNS)
    if not has_header:
        return False, 0, len(MANIFEST_VALIDATION_TERMS)
    
    # Check for at least one section pattern
    has_section = any(pattern.upper() in text_upper for pattern in MANIFEST_SECTION_PATTERNS)
    if not has_section:
        return False, 0, len(MANIFEST_VALIDATION_TERMS)
    
    # Count validation terms found
    matches = sum(1 for term in MANIFEST_VALIDATION_TERMS if term.upper() in text_upper)
    
    # Need at least 3 of the validation terms
    is_valid = matches >= 3
    return is_valid, matches, len(MANIFEST_VALIDATION_TERMS)


def process_pdf_fitz(pdf_path, pages_to_process=None):
    """
    Extract text from PDF using PyMuPDF (fitz) embedded text layer.
    This is fast and works well for PDFs with embedded text.
    
    Args:
        pdf_path: Path to the PDF file
        pages_to_process: Optional list of 1-indexed page numbers to process
    
    Returns:
        Dict with result_text and metadata, or None on failure
    """
    try:
        doc = fitz.open(pdf_path)
        all_text = []
        
        if pages_to_process:
            pages = [p - 1 for p in pages_to_process if 0 < p <= len(doc)]
            print(f"  Extracting text from {len(pages)} manifest pages with fitz: {pages_to_process}")
        else:
            pages = range(len(doc))
        
        for page_num in pages:
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                all_text.append(f"=== Page {page_num + 1} ===\n{text}")
        
        doc.close()
        
        full_text = "\n\n".join(all_text)
        
        return {
            "result_text": full_text,
            "pages_processed": pages_to_process or "all",
            "extraction_method": "fitz",
            "status": "success"
        }
    except Exception as e:
        print(f"  Error extracting with fitz: {e}")
        return None


def process_pdf_marker(pdf_path, pages_to_process=None):
    """
    Process a PDF file with Marker OCR.
    
    Args:
        pdf_path: Path to the PDF file
        pages_to_process: Optional list of 1-indexed page numbers to process
    
    Returns:
        Dict with result_text and metadata, or None on failure
    """
    global _MARKER_MODELS
    try:
        # import models as needed # TODO: do this at top
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models
        
        # Cache models per process to avoid reloading for every PDF
        if _MARKER_MODELS is None:
            _MARKER_MODELS = load_all_models()
        model_lst = _MARKER_MODELS
        
        # If specific pages requested, process each page separately to add page markers
        if pages_to_process:
            all_page_texts = []
            for page_num in pages_to_process:
                # Extract single page to temp PDF
                temp_pdf = extract_specific_pages(pdf_path, [page_num])
                if temp_pdf:
                    page_text, _, _ = convert_single_pdf(
                        str(temp_pdf),
                        model_lst,
                        max_pages=1,
                        metadata=None,
                        langs=None,
                        batch_multiplier=1
                    )
                    all_page_texts.append(f"=== Page {page_num} ===\n{page_text}")
                    # Clean up temp file
                    if os.path.exists(temp_pdf):
                        os.remove(temp_pdf)
            
            full_text = "\n\n".join(all_page_texts)
            print(f"  Processed {len(pages_to_process)} manifest pages with marker: {pages_to_process}")
        else:
            # Process whole document - add page markers by processing page-by-page
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            all_page_texts = []
            for page_num in range(1, total_pages + 1):
                temp_pdf = extract_specific_pages(pdf_path, [page_num])
                if temp_pdf:
                    page_text, _, _ = convert_single_pdf(
                        str(temp_pdf),
                        model_lst,
                        max_pages=1,
                        metadata=None,
                        langs=None,
                        batch_multiplier=1
                    )
                    all_page_texts.append(f"=== Page {page_num} ===\n{page_text}")
                    if os.path.exists(temp_pdf):
                        os.remove(temp_pdf)
            
            full_text = "\n\n".join(all_page_texts)
        
        return {
            "result_text": full_text,
            "markdown": full_text,
            "pages_processed": pages_to_process or "all",
            "extraction_method": "marker",
            "status": "success"
        }
    
    except ImportError as e:
        print(f"  Error: Marker not installed. Run: pip install marker-pdf")
        return None
    except Exception as e:
        print(f"  Error processing PDF with Marker: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_pdf_tesseract(pdf_path, max_pages=10):
    """Process a PDF file with Tesseract OCR."""
    images = convert_from_path(pdf_path, dpi=300)[:max_pages]
    print(f"  Processing {len(images)} pages")
    
    all_text = []
    for i, image in enumerate(images):
        # Preprocess
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR
        ocr_config = r'--oem 1 --psm 4 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(thresh, config=ocr_config, lang="eng")
        
        # Clean up
        lines = []
        for line in text.splitlines():
            line = clean_common_errors(line)
            leading = len(line) - len(line.lstrip())
            line = " " * leading + " ".join(line.strip().split())
            lines.append("".join(c for c in line if c.isprintable()))
        
        all_text.append("\n".join(lines))
    
    return {
        "result_text": "\n\n".join(all_text),
        "extraction_method": "tesseract",
        "status": "success",
    }


def process_pdf(pdf_path, ocr_engine="marker", max_pages=999, 
                process_only_manifests=False):
    """
    Process a single PDF file and extract text.
    
    Strategy:
    1. First try fitz (PyMuPDF) to extract embedded text - fast and reliable
    2. Validate that required manifest fields are present
    3. Fall back to the configured OCR engine if fitz extraction is insufficient
    
    Args:
        pdf_path: Path to the PDF file
        ocr_engine: Fallback OCR engine to use ("marker" or "tesseract")
        max_pages: Maximum pages to process
        process_only_manifests: If True, only process pages containing "manifest"
    """
    marker_folder_name = f"{ocr_engine}_output"
    fitz_folder_name = FITZ_OUTPUT_FOLDER
    
    print(f"Processing {pdf_path}")
    
    pdf_dir = os.path.dirname(pdf_path)
    parent_dir = os.path.dirname(pdf_dir)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    fitz_output_dir = os.path.join(parent_dir, fitz_folder_name)
    fallback_output_dir = os.path.join(parent_dir, marker_folder_name)
    fitz_pdf_folder = os.path.join(fitz_output_dir, pdf_name)
    fallback_pdf_folder = os.path.join(fallback_output_dir, pdf_name)
    
    # Ensure base dirs exist
    os.makedirs(fitz_output_dir, exist_ok=True)
    os.makedirs(fallback_output_dir, exist_ok=True)
    
    # Skip if already processed via fitz
    fitz_txt = os.path.join(fitz_pdf_folder, f"{pdf_name}.txt")
    fitz_json = os.path.join(fitz_pdf_folder, f"{pdf_name}.json")
    if os.path.exists(fitz_txt) and os.path.exists(fitz_json):
        print(f"  Already processed with fitz: {pdf_name}")
        return
    
    # Skip if already processed via fallback OCR
    fb_txt = os.path.join(fallback_pdf_folder, f"{pdf_name}.txt")
    fb_json = os.path.join(fallback_pdf_folder, f"{pdf_name}.json")
    if os.path.exists(fb_txt) and os.path.exists(fb_json):
        print(f"  Already processed with {ocr_engine}: {pdf_name}")
        return
    
    # Scan for manifest pages if requested, and optionally materialize a manifest-only PDF
    pages_to_process = None
    working_pdf_path = pdf_path
    if process_only_manifests:
        print("  Scanning for manifest pages...")
        manifest_pages = scan_pdf_for_manifest_pages(pdf_path)
        
        if manifest_pages:
            print(f"  Found manifest content on pages: {manifest_pages}")
            # Filter out pages beyond document length to avoid zero-page saves
            try:
                page_count = fitz.open(pdf_path).page_count
            except Exception:
                page_count = None
            if page_count is not None:
                manifest_pages = [p for p in manifest_pages if 0 < p <= page_count]
            pages_to_process = manifest_pages
            if not pages_to_process:
                print("  Manifest page numbers are out of range; skipping file")
                os.makedirs(fitz_pdf_folder, exist_ok=True)
                save_ocr_results(fitz_pdf_folder, pdf_name, "", {"status": "no_manifests_found"})
                return
            # Create (or reuse) a manifest-only PDF so downstream calls don't fail on missing files
            manifest_pdf_path = pdf_path.replace('.PDF', '_manifest_pages.pdf').replace('.pdf', '_manifest_pages.pdf')
            if not os.path.exists(manifest_pdf_path):
                manifest_pdf_path = extract_specific_pages(pdf_path, pages_to_process, output_path=manifest_pdf_path)
            if manifest_pdf_path is None:
                print("  No manifest pages; skipping file")
                os.makedirs(fitz_pdf_folder, exist_ok=True)
                save_ocr_results(fitz_pdf_folder, pdf_name, "", {"status": "no_manifests_found"})
                return
            working_pdf_path = manifest_pdf_path
        else:
            print("  No manifest pages found, skipping file")
            os.makedirs(fitz_pdf_folder, exist_ok=True)
            save_ocr_results(fitz_pdf_folder, pdf_name, "", {"status": "no_manifests_found"})
            return
    
    # STEP 1: Try fitz extraction first (fast, uses embedded text)
    result = None
    print("  Trying fitz (embedded text) extraction...")
    fitz_result = process_pdf_fitz(working_pdf_path, pages_to_process)
    
    if fitz_result:
        text = fitz_result.get("result_text", "")
        is_valid, matches, total = validate_manifest_extraction(text)
        
        if is_valid:
            print(f"  ✓ Fitz extraction successful ({matches}/{total} validation terms found)")
            result = fitz_result
        else:
            print(f"  ✗ Fitz extraction insufficient ({matches}/{total} validation terms found)")
    
    # STEP 2: Fall back to OCR if fitz didn't work
    if result is None:
        print(f"  Falling back to {ocr_engine} OCR...")
        
        if ocr_engine == "marker":
            result = process_pdf_marker(working_pdf_path, pages_to_process)
        elif ocr_engine == "tesseract":
            result = process_pdf_tesseract(working_pdf_path, max_pages)
        
        if result:
            text = result.get("result_text", "")
            is_valid, matches, total = validate_manifest_extraction(text)
            if is_valid:
                print(f"  ✓ {ocr_engine.title()} extraction successful ({matches}/{total} validation terms)")
            else:
                print(f"  ⚠ {ocr_engine.title()} extraction completed but low quality ({matches}/{total} validation terms)")
    
    if result:
        text_content = result.get("result_text", "")
        method = result.get("extraction_method", ocr_engine)
        
        if method == "fitz":
            os.makedirs(fitz_pdf_folder, exist_ok=True)
            save_dir = fitz_pdf_folder
        else:
            os.makedirs(fallback_pdf_folder, exist_ok=True)
            save_dir = fallback_pdf_folder
        
        save_ocr_results(save_dir, pdf_name, text_content, result)
        print(f"  Extraction complete via {method}")
    else:
        print(f"  Extraction failed")


def is_processed(pdf_path, ocr_engine):
    """Check if PDF has been successfully processed with non-empty output."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    base_dir = os.path.dirname(os.path.dirname(pdf_path))
    
    # Prefer fitz output if present
    fitz_txt = os.path.join(base_dir, FITZ_OUTPUT_FOLDER, pdf_name, f"{pdf_name}.txt")
    if os.path.exists(fitz_txt) and os.path.getsize(fitz_txt) > 0:
        return True
    
    folder_name = f"{ocr_engine}_output"
    txt_path = os.path.join(base_dir, folder_name, pdf_name, f"{pdf_name}.txt")
    if os.path.exists(txt_path):
        return os.path.getsize(txt_path) > 0
    return False


def collect_pdf_files():
    """Collect all PDF files from the data directory."""
    pdf_files = []
    
    for year in YEARS:
        base_path = f"ca_cafo_compliance/data/{year}"
        for region in REGIONS:
            region_path = os.path.join(base_path, region)
            if not os.path.exists(region_path):
                continue
            
            for county in os.listdir(region_path):
                county_path = os.path.join(region_path, county)
                if not os.path.isdir(county_path):
                    continue
                
                for template in os.listdir(county_path):
                    template_path = os.path.join(county_path, template)
                    if not os.path.isdir(template_path):
                        continue
                    
                    for folder in ["non-readable", "readable", "original"]:
                        folder_path = os.path.join(template_path, folder)
                        if os.path.exists(folder_path):
                            pdf_files.extend(glob.glob(os.path.join(folder_path, "*.pdf")))
                            pdf_files.extend(glob.glob(os.path.join(folder_path, "*.PDF")))
    
    return pdf_files


def sort_pdf_files(pdf_files):
    """Sort PDF files by priority (2024 first, R5 first)."""
    def priority(path):
        return (
            0 if "2024" in path else 1,
            0 if "/R5/" in path else 1,
        )
    return sorted(pdf_files, key=priority)


def update_reports_available_csv():
    """Update reports_available.csv with PDF counts."""
    region_county_map = {
        "5F": ["kern"],
        "5S": ["fresno_madera", "kings", "tulare_west"],
        "5R": [],
    }
    
    base_path = "ca_cafo_compliance/data/2023/R5"
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
    # Update reports CSV
    update_reports_available_csv()
    
    # Collect and sort PDF files
    pdf_files = collect_pdf_files()
    pdf_files = sort_pdf_files(pdf_files)
    
    # Filter out already processed files
    files_to_process = [f for f in pdf_files if not is_processed(f, ocr_engine)]
    processed_count = len(pdf_files) - len(files_to_process)
    
    print(f"Already processed: {processed_count} files")
    print(f"To process: {len(files_to_process)} files")
    print(f"OCR engine: {ocr_engine}")
    if process_only_manifests:
        print("Mode: Manifest pages only")
    
    # Test mode filtering
    if test_mode:
        white_river = [f for f in files_to_process if "White River Dairy" in f]
        files_to_process = white_river[:1] if white_river else files_to_process[:1]
        print(f"Test mode: processing {len(files_to_process)} file(s)")
    
    if not files_to_process:
        print("No files to process")
        return
    
    # Process files
    for pdf_path in files_to_process:
        process_pdf(
            pdf_path,
            ocr_engine=ocr_engine,
            max_pages=max_pages,
            process_only_manifests=process_only_manifests
        )
    
    print("OCR complete")


if __name__ == "__main__":

    manifests_only = True
    test_mode = False
    ocr_engine="marker"
    
    main(
        ocr_engine=ocr_engine,
        test_mode=test_mode,
        max_pages=999,
        process_only_manifests=manifests_only
    )
