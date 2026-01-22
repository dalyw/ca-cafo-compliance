#!/usr/bin/env python3

import os
import glob
import csv
import json
import pymupdf as fitz
from helper_functions.read_report_helpers import clean_common_errors, YEARS, REGIONS

# OCR engine constants
TESSERACT = "tesseract"
LLMWHISPERER = "llmwhisperer"
MARKER = "marker"

# Configuration
OCR_ENGINE = MARKER  # Default OCR engine
TEST_MODE = False  # Process only test files
MAX_PAGES = 999  # Max pages per document

# LLMWhisperer API settings
LLMWHISPERER_API_KEY = os.getenv("LLMWHISPERER_API_KEY", "")
LLMWHISPERER_BASE_URL = "https://llmwhisperer-api.us-central.unstract.com/api/v2"


def get_output_folder_name(ocr_engine):
    """Get the output folder name for the OCR engine."""
    return {
        TESSERACT: "tesseract_ocr_output",
        LLMWHISPERER: "llmwhisperer_output",
        MARKER: "marker_output",
    }.get(ocr_engine, "ocr_output")


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
        search_terms = ["manifest", "hauler", "exporter", "importer", "recipient"]
    
    
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
            
            # Use tesseract for quick scan if available
            try:
                import pytesseract
                from PIL import Image
                import io
                
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, config='--psm 3').lower()
            except ImportError:
                # No tesseract, skip OCR scan
                text = ""
        
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
    
    for page_num in pages:
        if 0 < page_num <= len(doc):
            new_doc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
    
    if output_path is None:
        output_path = pdf_path.replace(".pdf", "_manifest_pages.pdf").replace(".PDF", "_manifest_pages.pdf")
    
    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    
    return output_path
    

def process_pdf_marker(pdf_path, pages_to_process=None):
    """
    Process a PDF file with Marker OCR.
    
    Args:
        pdf_path: Path to the PDF file
        pages_to_process: Optional list of 1-indexed page numbers to process
    
    Returns:
        Dict with result_text and metadata, or None on failure
    """
    try:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models
        
        model_lst = load_all_models()
        
        # If specific pages requested, extract them first
        temp_pdf = None
        process_path = pdf_path
        
        if pages_to_process:
            temp_pdf = extract_specific_pages(pdf_path, pages_to_process)
            if temp_pdf:
                process_path = temp_pdf
                print(f"  Processing {len(pages_to_process)} manifest pages: {pages_to_process}")
        
        full_text, images, out_meta = convert_single_pdf(
            str(process_path),
            model_lst,
            max_pages=999,
            metadata=None,
            langs=None,
            batch_multiplier=1
        )
        
        # Clean up temp file
        if temp_pdf and os.path.exists(temp_pdf):
            os.remove(temp_pdf)
        
        return {
            "result_text": full_text,
            "markdown": full_text,
            "pages_processed": pages_to_process or "all",
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


def process_pdf_llmwhisperer(pdf_path, max_pages=999):
    """Process a PDF file with LLMWhisperer API."""
    import requests
    import time
    
    if not LLMWHISPERER_API_KEY:
        raise ValueError("LLMWHISPERER_API_KEY not set in environment variables")
    
    headers = {"unstract-key": LLMWHISPERER_API_KEY}
    clean_filename = os.path.basename(pdf_path).replace(" ", "_")
    
    params = {
        "mode": "form",
        "output_mode": "line-printer",
        "page_separator": "<<<",
        "force_text_processing": "true",
        "timeout": 300,
    }
    if max_pages < 999:
        params["pages_to_extract"] = f"1-{max_pages}"
    
    print(f"  Sending to LLMWhisperer API")
    
    with open(pdf_path, "rb") as f:
        files = {"file": (clean_filename, f, "application/pdf")}
        response = requests.post(
            f"{LLMWHISPERER_BASE_URL}/whisper",
            headers=headers,
            files=files,
            params=params,
            timeout=600
        )
    
    if response.status_code == 202:
        result = response.json()
        whisper_hash = result.get("whisper_hash")
        print(f"  Polling for results...")
        
        for attempt in range(300):
            time.sleep(2)
            status_response = requests.get(
                f"{LLMWHISPERER_BASE_URL}/whisper-status",
                headers=headers,
                params={"whisper_hash": whisper_hash},
                timeout=30
            )
            
            if status_response.status_code == 200:
                status_result = status_response.json()
                status = status_result.get("status")
                
                if status == "processed":
                    retrieve_response = requests.get(
                        f"{LLMWHISPERER_BASE_URL}/whisper-retrieve",
                        headers=headers,
                        params={"whisper_hash": whisper_hash},
                        timeout=30
                    )
                    if retrieve_response.status_code == 200:
                        return retrieve_response.json()
                elif status not in ["processing", "accepted"]:
                    print(f"  Error: {status}")
                    return None
        
        print("  Timeout waiting for results")
        return None
        
    elif response.status_code == 200:
        result = response.json()
        return {"text": result.get("extraction", {}).get("result_text", "")}
    else:
        print(f"  Error: {response.status_code} - {response.text}")
        return None


def process_pdf_tesseract(pdf_path, max_pages=10):
    """Process a PDF file with Tesseract OCR."""
    import numpy as np
    
    try:
        import cv2
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        raise ImportError(
            "Tesseract requires: pip install opencv-python pytesseract pdf2image"
        ) from e
    
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
    
    return {"result_text": "\n\n".join(all_text), "status": "success"}


def process_pdf(pdf_path, ocr_engine=MARKER, max_pages=999, 
                process_only_manifests=False):
    """
    Process a single PDF file and extract text using OCR.
    
    Args:
        pdf_path: Path to the PDF file
        ocr_engine: OCR engine to use (MARKER, TESSERACT, or LLMWHISPERER)
        max_pages: Maximum pages to process
        process_only_manifests: If True, only process pages containing "manifest"
    """
    folder_name = get_output_folder_name(ocr_engine)
    
    print(f"Processing {pdf_path}")
    
    pdf_dir = os.path.dirname(pdf_path)
    parent_dir = os.path.dirname(pdf_dir)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    output_dir = os.path.join(parent_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_folder = os.path.join(output_dir, pdf_name)
    os.makedirs(pdf_folder, exist_ok=True)
    
    # Check if already processed
    text_file = os.path.join(output_dir, f"{pdf_name}.txt")
    json_file = os.path.join(output_dir, f"{pdf_name}.json")
    
    if os.path.exists(text_file) and os.path.exists(json_file):
        print(f"  Already processed: {pdf_name}")
        return
    
    # Scan for manifest pages if requested
    pages_to_process = None
    if process_only_manifests:
        print("  Scanning for manifest pages...")
        manifest_pages = scan_pdf_for_manifest_pages(pdf_path)
        
        if manifest_pages:
            print(f"  Found manifest content on pages: {manifest_pages}")
            pages_to_process = manifest_pages
        else:
            print("  No manifest pages found, skipping file")
            # Save empty result to mark as processed
            save_ocr_results(pdf_folder, pdf_name, "", {"status": "no_manifests_found"})
            return
    
    # Process with selected OCR engine
    result = None
    
    if ocr_engine == MARKER:
        result = process_pdf_marker(pdf_path, pages_to_process)
    elif ocr_engine == LLMWHISPERER:
        result = process_pdf_llmwhisperer(pdf_path, max_pages)
    elif ocr_engine == TESSERACT:
        result = process_pdf_tesseract(pdf_path, max_pages)
    
    if result:
        text_content = result.get("result_text", "")
        save_ocr_results(pdf_folder, pdf_name, text_content, result)
        print(f"  Extraction complete")
    else:
        print(f"  Extraction failed")


def is_processed(pdf_path, ocr_engine):
    """Check if PDF has been successfully processed with non-empty output."""
    folder_name = get_output_folder_name(ocr_engine)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    txt_path = os.path.join(
        os.path.dirname(os.path.dirname(pdf_path)),
        folder_name,
        pdf_name,
        f"{pdf_name}.txt",
    )
    
    if os.path.exists(txt_path):
        try:
            return os.path.getsize(txt_path) > 0
        except OSError:
            return False
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
    
    if ocr_engine == LLMWHISPERER and not LLMWHISPERER_API_KEY:
        print("Get API key at: https://unstract.com/llmwhisperer/")
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
    ocr_engine=MARKER
    
    main(
        ocr_engine=ocr_engine,
        test_mode=test_mode,
        max_pages=999,
        process_only_manifests=manifests_only
    )
