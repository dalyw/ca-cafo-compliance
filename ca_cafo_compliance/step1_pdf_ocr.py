#!/usr/bin/env python3
import os
import glob
import csv
import json
import cv2
import requests
import time
import pandas as pd
import pymupdf as fitz
import pytesseract
from pytesseract import Output
import numpy as np
from PIL import Image
from PIL import Image as PILImage
import io
from pdf2image import convert_from_path
import tempfile
from dotenv import load_dotenv
from helpers_pdf_metrics import YEARS, REGIONS, GDRIVE_BASE

load_dotenv()  # to load LLMWhisperer API key

# Configuration
TEST_MODE = False  # Process only test files
# LLMWhisperer API settings
LLMWHISPERER_API_KEY = os.getenv("LLMWHISPERER_API_KEY", "")
LLMWHISPERER_BASE_URL = "https://llmwhisperer-api.us-central.unstract.com/api/v2"

repo_base_dir = os.path.dirname(os.path.abspath(__file__))

manifest_specific_terms = [
    "hauler info",
    "destination",
    "method used",
    "operator shall",
    "d-2",
    "solids content",
    "hauler signature",
    "hauling event",
    "complete one",
]


def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image while keeping the full image in view (no corner cropping).

    Note: matches the behavior of imutils.rotate_bound by rotating *clockwise*
    for positive angles (OpenCV uses counter-clockwise angles by default).
    """
    h, w = image.shape[:2]
    cX, cY = (w / 2.0, h / 2.0)

    # rotate clockwise for positive angles (imutils compatibility)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2.0) - cX
    M[1, 2] += (nH / 2.0) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def _correct_orientation_osd(image_bgr: np.ndarray) -> np.ndarray:
    """Use Tesseract OSD to deskew/rotate a page image into reading orientation."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
    except pytesseract.TesseractError:
        return image_bgr
    rotate = float(results.get("rotate", 0))
    if rotate:
        return _rotate_bound(image_bgr, angle=rotate)
    return image_bgr


def convert_pages_safe(pdf_path, first_p, last_p, dpi_list=(350, 250, 200, 150)):
    last_err = None
    for dpi in dpi_list:
        try:
            return (
                convert_from_path(pdf_path, dpi=dpi, first_page=first_p, last_page=last_p),
                dpi,
            )
        except PILImage.DecompressionBombError as e:
            last_err = e
    raise last_err


def extract_specific_pages(pdf_path, pages, output_path=None):
    """Extract specific pages from a PDF to a new manifest file."""

    if not pages:
        return None

    doc = fitz.open(pdf_path)
    new_doc = fitz.open()
    added_pages = 0

    for page_num in pages:
        if 0 < page_num <= len(doc):
            new_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
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


def needs_handwritten_analysis(text):
    """Detect if manifest needs handwritten OCR analysis (R5-2013-0122 or handwritten forms)."""
    text_upper = text.upper()

    # R5-2013-0122 template: identified by "R5-2013-0122" or "CUBIC YARDS"
    if "R5-2013-0122" in text_upper or "CUBIC YARDS" in text_upper:
        return True

    return False


def find_manifest_pages(pdf_path, *, detect_orientation=False):
    manifest_pages = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().lower()

        # If no embedded text, use OCR at low resolution to identify manifest pages
        if len(text.strip()) < 50:
            pix = page.get_pixmap(dpi=300)  # can use 144 for testing or faster results
            pil_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            if detect_orientation:
                img_bgr = _correct_orientation_osd(img_bgr)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(img_rgb, config="--psm 3").lower()

        if any(term in text for term in manifest_specific_terms):
            manifest_pages.append(page_num + 1)
            if page_num + 1 < len(doc):
                manifest_pages.append(
                    page_num + 2
                )  # include following page for later processing too

    doc.close()
    return sorted(set(manifest_pages))  # remove duplicates


def pages_to_extract_str(pages: list[int]) -> str:
    pages = sorted(set(int(p) for p in pages))
    if not pages:
        return ""
    ranges = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = prev = p
    ranges.append((start, prev))
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


def _extract_llmwhisperer_text(payload: dict) -> str:
    """
    Incorporates your working-code expectation:
      payload['extraction']['result_text']
    plus some safe fallbacks.
    """
    if not isinstance(payload, dict):
        return ""

    # what your original "200 response" path expected
    t = (payload.get("extraction") or {}).get("result_text")
    if isinstance(t, str) and t.strip():
        return t

    # other common possibilities (defensive)
    for k in ("result_text", "text", "extracted_text", "content"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # sometimes nested differently
    data = payload.get("data") or {}
    if isinstance(data, dict):
        for k in ("result_text", "text", "extracted_text"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v

    return ""


def extract_text_llmwhisperer(
    pdf_path: str, pages_to_process=None, max_pages=999, *, keep_raw=False
):
    if not LLMWHISPERER_API_KEY:
        raise ValueError("LLMWHISPERER_API_KEY not set in environment variables")

    # Use a unique separator so we can reliably split pages
    sep = "<<<PAGE_BREAK>>>"

    params = {
        "mode": "form",
        "timeout": 300,
        "output_mode": "line-printer",
        "page_separator": sep,
        "force_text_processing": "true",
    }

    if pages_to_process:
        params["pages_to_extract"] = pages_to_extract_str(pages_to_process)
        page_nums = sorted(set(pages_to_process))
    elif max_pages < 999:
        params["pages_to_extract"] = f"1-{max_pages}"
        page_nums = list(range(1, max_pages + 1))
    else:
        page_nums = None

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    post_headers = {
        "unstract-key": LLMWHISPERER_API_KEY,
        "Content-Type": "application/pdf",
    }
    get_headers = {"unstract-key": LLMWHISPERER_API_KEY}

    response = requests.post(
        f"{LLMWHISPERER_BASE_URL}/whisper",
        headers=post_headers,
        params=params,
        data=pdf_bytes,
        timeout=600,
    )

    # Two possible workflows (based on your working code)
    if response.status_code == 202:
        job = response.json()
        whisper_hash = job.get("whisper_hash")
        if not whisper_hash:
            raise RuntimeError(f"No whisper_hash in 202 response: {job}")

        for _ in range(300):
            time.sleep(2)
            status_response = requests.get(
                f"{LLMWHISPERER_BASE_URL}/whisper-status",
                headers=get_headers,
                params={"whisper_hash": whisper_hash},
                timeout=30,
            )
            status_response.raise_for_status()
            status_result = status_response.json()
            status = status_result.get("status")

            if status == "processed":
                retrieve_response = requests.get(
                    f"{LLMWHISPERER_BASE_URL}/whisper-retrieve",
                    headers=get_headers,
                    params={"whisper_hash": whisper_hash},
                    timeout=60,
                )
                retrieve_response.raise_for_status()
                payload = retrieve_response.json()

                raw_text = _extract_llmwhisperer_text(payload)
                chunks = [c.strip() for c in raw_text.split(sep)] if raw_text else []
                if len(chunks) <= 1 and raw_text and "<<<" in raw_text:
                    print("LLMWhisperer returned only one chunk, splitting on <<<")
                    chunks = [c.strip() for c in raw_text.split("<<<")]

                # If we didn't know pages beforehand, assume sequential numbering
                if page_nums is None:
                    page_nums = list(range(1, len(chunks) + 1))

                out = []
                for i, chunk in enumerate(chunks):
                    if i >= len(page_nums):
                        break
                    if chunk:
                        out.append(f"=== Page {page_nums[i]} ===\n{chunk}")

                result = {
                    "result_text": "\n\n".join(out),
                    "extraction_method": "llmwhisperer",
                    "page_count": len(out),
                }
                if keep_raw:
                    result["raw_response"] = payload
                return result

            if status not in ("processing", "accepted"):
                raise RuntimeError(f"LLMWhisperer error status: {status_result}")

        raise TimeoutError("Timeout waiting for LLMWhisperer results")

    elif response.status_code == 200:
        payload = response.json()
        raw_text = _extract_llmwhisperer_text(payload)
        chunks = [c.strip() for c in raw_text.split(sep)] if raw_text else []
        if len(chunks) <= 1 and raw_text and "<<<" in raw_text:
            print("LLMWhisperer returned only one chunk, splitting on <<<")
            chunks = [c.strip() for c in raw_text.split("<<<")]

        if page_nums is None:
            page_nums = list(range(1, len(chunks) + 1))

        out = []
        for i, chunk in enumerate(chunks):
            if i >= len(page_nums):
                break
            if chunk:
                out.append(f"=== Page {page_nums[i]} ===\n{chunk}")

        result = {
            "result_text": "\n\n".join(out),
            "extraction_method": "llmwhisperer",
            "page_count": len(out),
        }
        if keep_raw:
            result["raw_response"] = payload
        return result

    else:
        raise RuntimeError(f"LLMWhisperer error {response.status_code}: {response.text[:500]}")


def extract_text_from_pdf(pdf_path, method="fitz", pages_to_process=None):
    if pages_to_process is None:
        # Process all pages if none specified
        doc = fitz.open(pdf_path)
        pages_to_process = list(range(1, len(doc) + 1))
        doc.close()

    if method == "fitz":
        all_text = []
        with fitz.open(pdf_path) as doc:
            for p in pages_to_process:
                text = doc[p - 1].get_text()
                if text.strip():
                    all_text.append(f"=== Page {p} ===\n{text}")

            full_text = "\n\n".join(all_text)

            return {"result_text": full_text, "extraction_method": "fitz"}

    elif method == "tesseract":
        all_text = []

        for page_num in pages_to_process:
            images, used_dpi = convert_pages_safe(pdf_path, page_num, page_num)
            image = images[0]

            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            img = _correct_orientation_osd(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            text = pytesseract.image_to_string(
                thresh, config="--oem 1 --psm 4 -c preserve_interword_spaces=1"
            )
            all_text.append(f"=== Page {page_num} ===\n{text}")

        full_text = "\n\n".join(all_text)
        return {"result_text": full_text, "extraction_method": "tesseract"}

    elif method == "llmwhisperer":
        return extract_text_llmwhisperer(
            pdf_path, pages_to_process=pages_to_process, keep_raw=False
        )


def extract_text_auto(pdf_path, pages_to_process=None):
    results = {}

    # 1) try text layer
    results["fitz"] = extract_text_from_pdf(
        pdf_path, method="fitz", pages_to_process=pages_to_process
    )

    # 2) tesseract first for scans
    results["tesseract"] = extract_text_from_pdf(
        pdf_path, method="tesseract", pages_to_process=pages_to_process
    )
    tesseract_text = results["tesseract"].get("result_text", "")

    # 3) LLMWhisperer for better analysis
    if needs_handwritten_analysis(tesseract_text):
        results["llmwhisperer"] = extract_text_from_pdf(
            pdf_path, method="llmwhisperer", pages_to_process=pages_to_process
        )
        return results, "llmwhisperer"

    return results, "tesseract"


def extract_pdf_text(pdf_path, process_only_manifests=False):
    """Process a single PDF file and extract text."""

    print(f"Processing {pdf_path}")

    pages_to_process = None
    if process_only_manifests:
        manifest_pages = find_manifest_pages(pdf_path, detect_orientation=True)
        if manifest_pages:
            print(f"  Found {len(manifest_pages)} likely manifest page(s): {manifest_pages}")
            pages_to_process = manifest_pages
        else:  # Save empty results
            print("  No manifest pages found, skipping")
            paths = get_output_paths(pdf_path, "fitz", mkdir=True)
            os.makedirs(paths["dir"], exist_ok=True)
            with open(paths["txt"], "w", encoding="utf-8") as f:
                f.write("no_manifests_found")
            with open(paths["json"], "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "extraction_method": "fitz",
                        "final_method": "fitz",
                        "page_count": 0,
                    },
                    f,
                    indent=2,
                )
            return

    results, final_method = extract_text_auto(pdf_path, pages_to_process)

    # Always save fitz and tesseract; only save llmwhisperer if it's the final method
    methods_to_save = ["fitz", "tesseract"]

    if final_method == "llmwhisperer":
        methods_to_save.append("llmwhisperer")

    for method in methods_to_save:
        if method in results:
            result = results[method]
            result["final_method"] = final_method
            print(f"  Saving {method} output")
            paths = get_output_paths(pdf_path, method, mkdir=True)
            with open(paths["txt"], "w", encoding="utf-8") as f:
                f.write(result["result_text"])
            page_count = result["result_text"].count("=== Page ")
            with open(paths["json"], "w", encoding="utf-8") as f:
                minimal = {
                    "extraction_method": method,
                    "final_method": result.get("final_method"),
                    "page_count": page_count,
                }
                json.dump(minimal, f, indent=2, ensure_ascii=False)

    print(f"  Extraction complete via {final_method}")


def get_output_paths(pdf_path, method, mkdir=False):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    folder = f"{method}_output"

    parts = os.path.normpath(pdf_path).split(os.sep)
    i = parts.index("ca_cafo_manifests")
    year = parts[i + 1]
    region = parts[i + 2]
    county = parts[i + 3]
    template = parts[i + 4]

    out_dir = os.path.join(GDRIVE_BASE, year, region, county, template, folder, pdf_name)

    if mkdir:
        os.makedirs(out_dir, exist_ok=True)

    return {
        "dir": out_dir,
        "txt": os.path.join(out_dir, f"{pdf_name}.txt"),
        "json": os.path.join(out_dir, f"{pdf_name}.json"),
    }


def is_processed(pdf_path):
    """Check if PDF has already been processed with non-empty output."""
    # check if either fitz or tesseract output exists
    for method in ["fitz", "tesseract"]:
        paths = get_output_paths(pdf_path, method, mkdir=False)
        if os.path.exists(paths["txt"]) and os.path.getsize(paths["txt"]) > 0:
            return True
    return False


def collect_pdf_files(years=None, regions=REGIONS):
    """Collect all PDF files from the data directory."""
    pdf_files = []
    if not years:
        years = YEARS
    if not regions:
        regions = REGIONS
    for year in years:
        base_path = GDRIVE_BASE + f"/{year}"
        print(base_path)
        for region in regions:
            region_path = os.path.join(base_path, region)
            if not os.path.exists(region_path):
                print(f"no {region} path")
                continue

            for county in os.listdir(region_path):
                print(f" Collecting PDFs for {county}")
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
                print(f" Collected {len(pdf_files)} PDFs")

    return pdf_files


def update_reports_available_csv():
    """Update reports_available.csv with PDF counts."""
    region_county_map = {
        "5F": ["kern"],
        "5S": ["fresno_madera", "kings", "tulare_west"],
        "5R": [],
    }

    csv_path = "ca_cafo_compliance/data/reports_available.csv"
    pdf_counts = {region: 0 for region in region_county_map}

    for year in YEARS:
        for region, counties in region_county_map.items():
            for county in counties:
                county_path = os.path.join(GDRIVE_BASE, str(year), region, county)
                for _, _, files in os.walk(county_path):
                    pdf_counts[region] += sum(1 for f in files if f.lower().endswith(".pdf"))

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


def main(test_mode=TEST_MODE, process_only_manifests=False, process_missing_pages=False):

    # If processing missing pages, do that instead
    if process_missing_pages:

        df = pd.read_csv(
            os.path.join(
                repo_base_dir,
                "outputs",
                "2024_files_by_template_manual_counts_discrepancies.csv",
            )
        )

        # Filter for rows with 'missing' in notes and non-null page ranges
        missing_rows = df[
            (df["notes"].str.contains("missing", case=False, na=False))
            & (df["missing_page_start"].notna())
            & (df["missing_page_end"].notna())
        ]

        print(f"{len(missing_rows)} files with missing pages to process")

        for idx, row in missing_rows.iterrows():
            page_start = int(row["missing_page_start"])
            page_end = int(row["missing_page_end"])
            pdf_path = os.path.join(
                GDRIVE_BASE,
                "2024",
                "R5",
                row["county"],
                row["template"],
                "original",
                row["filename"],
            )

            print(f"\nProcessing: {row['filename']} pages {page_start}-{page_end}")
            pages_to_process = list(range(page_start, page_end + 1))
            result = extract_text_from_pdf(
                pdf_path, method="llmwhisperer", pages_to_process=pages_to_process
            )
            result["final_method"] = "llmwhisperer"
            result["source"] = "missing_pages_recovery"

            # Append to existing file or create new one
            paths = get_output_paths(pdf_path, "llmwhisperer", mkdir=True)
            mode = "a" if os.path.exists(paths["txt"]) else "w"
            with open(paths["txt"], mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n=== RECOVERED MISSING PAGES ===\n\n")
                f.write(result["result_text"])

            # Update JSON metadata
            existing_meta = {}
            if os.path.exists(paths["json"]):
                with open(paths["json"], "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)

            existing_meta.update(
                {
                    "recovered_pages": pages_to_process,
                    "recovery_method": "llmwhisperer",
                    "recovery_page_count": len(pages_to_process),
                }
            )

            with open(paths["json"], "w", encoding="utf-8") as f:
                json.dump(existing_meta, f, indent=2, ensure_ascii=False)

        print("Missing pages processing complete")
        return

    update_reports_available_csv()

    # Collect and sort PDF files
    pdf_files = collect_pdf_files([2024], ["R5"])

    # Filter out already processed files
    files_to_process = []
    for pdf_path in pdf_files:
        if not is_processed(pdf_path):
            files_to_process.append(pdf_path)
    print(f"{len(files_to_process)} of {len(pdf_files)} remaining")

    if test_mode:
        white_river = [f for f in files_to_process if "White River Dairy" in f]
        files_to_process = white_river[:1] if white_river else files_to_process[:1]
        print(f"Test mode: processing {len(files_to_process)} file(s)")

    if not files_to_process:
        print("No files to process")
    else:  # Process files
        for pdf_path in files_to_process:
            extract_pdf_text(pdf_path, process_only_manifests=process_only_manifests)


if __name__ == "__main__":
    main(test_mode=False, process_only_manifests=False)
    main(test_mode=False, process_only_manifests=False, process_missing_pages=True)
    print("OCR complete")
