#!/usr/bin/env python3
import os
import glob
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
import io
from pdf2image import convert_from_path
from dotenv import load_dotenv
from helpers import PATH_TO_PDF_DATA

load_dotenv()

# Configuration
TEST_MODE = False
LLMWHISPERER_API_KEY = os.getenv("LLMWHISPERER_API_KEY", "")
LLMWHISPERER_BASE_URL = "https://llmwhisperer-api.us-central.unstract.com/api/v2"

REPO_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_TERMS = [
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


def rotate_image_keep_full_view(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image clockwise while keeping full image in view."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    mat = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    mat[0, 2] += new_w / 2.0 - cx
    mat[1, 2] += new_h / 2.0 - cy
    return cv2.warpAffine(image, mat, (new_w, new_h))


def deskew_image_with_tesseract_osd(image_bgr: np.ndarray) -> np.ndarray:
    """Use Tesseract OSD to deskew/rotate page image."""
    try:
        results = pytesseract.image_to_osd(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), output_type=Output.DICT
        )
        rotate = float(results.get("rotate", 0))
        return rotate_image_keep_full_view(image_bgr, rotate) if rotate else image_bgr
    except pytesseract.TesseractError:
        return image_bgr


def convert_pdf_pages_with_fallback_dpi(pdf_path, first_p, last_p, dpi_list=(350, 250, 200, 150)):
    """Try converting PDF pages at progressively lower DPI to avoid decompression bomb."""
    last_err = None
    for dpi in dpi_list:
        try:
            return convert_from_path(pdf_path, dpi=dpi, first_page=first_p, last_page=last_p), dpi
        except Image.DecompressionBombError as e:
            last_err = e
    raise last_err


def requires_handwritten_ocr(text):
    """Detect if manifest needs handwritten OCR (R5-2013-0122 or CUBIC YARDS)."""
    text_upper = text.upper()
    # Check for short text
    if len(text.strip()) < 200:
        return True
    # Check for gibberish (non-alphanumeric ratio)
    gibberish_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(
        1, len(text)
    )
    if gibberish_ratio > 0.15:
        return True
    return "R5-2013-0122" in text_upper or "CUBIC YARDS" in text_upper


def detect_manifest_pages_in_pdf(pdf_path, *, detect_orientation=False):
    """Find pages containing manifest-specific terms."""
    manifest_pages = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().lower()

        # OCR if insufficient embedded text
        if len(text.strip()) < 50:
            pix = page.get_pixmap(dpi=300)
            pil_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            if detect_orientation:
                img_bgr = deskew_image_with_tesseract_osd(img_bgr)
            text = pytesseract.image_to_string(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), config="--psm 3"
            ).lower()

        if any(term in text for term in MANIFEST_TERMS):
            manifest_pages.extend([page_num + 1, page_num + 2])  # include next page

    doc.close()
    return sorted(set(p for p in manifest_pages if p <= len(doc)))


def pages_list_to_range_string(pages: list[int]) -> str:
    """Convert page list to range string (e.g., [1,2,3,5,6] -> '1-3,5-6')."""
    pages = sorted(set(int(p) for p in pages))
    if not pages:
        return ""
    ranges, start = [], pages[0]
    prev = start
    for p in pages[1:] + [None]:
        if p != prev + 1:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = p
        prev = p
    return ",".join(ranges)


def extract_text_from_llmwhisperer_payload(payload: dict) -> str:
    """Extract text from LLMWhisperer response payload."""
    if not isinstance(payload, dict):
        return ""
    for key in ("result_text", "text", "extracted_text", "content"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def extract_pdf_text_llmwhisperer_api(
    pdf_path: str, pages_to_process=None, max_pages=999, *, keep_raw=False
):
    """Extract text using LLMWhisperer API."""
    if not LLMWHISPERER_API_KEY:
        raise ValueError("LLMWHISPERER_API_KEY not set")

    sep = "<<<PAGE_BREAK>>>"
    params = {
        "mode": "form",
        "timeout": 300,
        "output_mode": "line-printer",
        "page_separator": sep,
        "force_text_processing": "true",
    }

    if pages_to_process:
        params["pages_to_extract"] = pages_list_to_range_string(pages_to_process)
        page_nums = sorted(set(pages_to_process))
    elif max_pages < 999:
        params["pages_to_extract"] = f"1-{max_pages}"
        page_nums = list(range(1, max_pages + 1))
    else:
        page_nums = None

    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{LLMWHISPERER_BASE_URL}/whisper",
            headers={"unstract-key": LLMWHISPERER_API_KEY, "Content-Type": "application/pdf"},
            params=params,
            data=f.read(),
            timeout=600,
        )

    def build_result(payload):
        """Build formatted result from LLMWhisperer payload."""
        raw_text = extract_text_from_llmwhisperer_payload(payload)
        chunks = [c.strip() for c in raw_text.split(sep)] if raw_text else []
        if len(chunks) <= 1 and "<<<" in raw_text:
            chunks = [c.strip() for c in raw_text.split("<<<")]

        nums = page_nums or list(range(1, len(chunks) + 1))
        out = [
            f"=== Page {nums[i]} ===\n{chunk}"
            for i, chunk in enumerate(chunks)
            if i < len(nums) and chunk
        ]

        result = {
            "result_text": "\n\n".join(out),
            "extraction_method": "llmwhisperer",
            "page_count": len(out),
        }
        if keep_raw:
            result["raw_response"] = payload
        return result

    if response.status_code == 200:
        return build_result(response.json())

    if response.status_code == 202:
        whisper_hash = response.json().get("whisper_hash")
        if not whisper_hash:
            raise RuntimeError(f"No whisper_hash in 202 response")

        for _ in range(300):
            time.sleep(2)
            status_resp = requests.get(
                f"{LLMWHISPERER_BASE_URL}/whisper-status",
                headers={"unstract-key": LLMWHISPERER_API_KEY},
                params={"whisper_hash": whisper_hash},
                timeout=30,
            )
            status_resp.raise_for_status()
            status = status_resp.json().get("status")

            if status == "processed":
                retrieve_resp = requests.get(
                    f"{LLMWHISPERER_BASE_URL}/whisper-retrieve",
                    headers={"unstract-key": LLMWHISPERER_API_KEY},
                    params={"whisper_hash": whisper_hash},
                    timeout=60,
                )
                retrieve_resp.raise_for_status()
                return build_result(retrieve_resp.json())

            if status not in ("processing", "accepted"):
                raise RuntimeError(f"LLMWhisperer error status: {status}")

        raise TimeoutError("Timeout waiting for LLMWhisperer results")

    raise RuntimeError(f"LLMWhisperer error {response.status_code}: {response.text[:500]}")


def extract_pdf_text_by_method(pdf_path, method="fitz", pages_to_process=None):
    """Extract text from PDF using specified method."""
    if pages_to_process is None:
        with fitz.open(pdf_path) as doc:
            pages_to_process = list(range(1, len(doc) + 1))

    if method == "fitz":
        all_text = []
        with fitz.open(pdf_path) as doc:
            for p in pages_to_process:
                text = doc[p - 1].get_text()
                if text.strip():
                    all_text.append(f"=== Page {p} ===\n{text}")
        return {"result_text": "\n\n".join(all_text), "extraction_method": "fitz"}

    if method == "tesseract":
        all_text = []
        for page_num in pages_to_process:
            images, _ = convert_pdf_pages_with_fallback_dpi(pdf_path, page_num, page_num)
            img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            img = deskew_image_with_tesseract_osd(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(
                thresh, config="--oem 1 --psm 4 -c preserve_interword_spaces=1"
            )
            all_text.append(f"=== Page {page_num} ===\n{text}")
        return {"result_text": "\n\n".join(all_text), "extraction_method": "tesseract"}

    if method == "llmwhisperer":
        return extract_pdf_text_llmwhisperer_api(pdf_path, pages_to_process=pages_to_process)


def auto_select_pdf_text_extraction(pdf_path, pages_to_process=None):
    """Auto-select best extraction method."""
    results = {
        "fitz": extract_pdf_text_by_method(pdf_path, "fitz", pages_to_process),
        "tesseract": extract_pdf_text_by_method(pdf_path, "tesseract", pages_to_process),
    }

    if requires_handwritten_ocr(results["tesseract"].get("result_text", "")):
        results["llmwhisperer"] = extract_pdf_text_by_method(
            pdf_path, "llmwhisperer", pages_to_process
        )
        return results, "llmwhisperer"

    return results, "tesseract"


def get_extraction_output_paths(pdf_path, method, mkdir=False):
    """Get output paths for extraction results."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    parts = os.path.normpath(pdf_path).split(os.sep)
    idx = parts.index("Manure Trucking Network Analysis")
    data, region, county, template = parts[idx + 1 : idx + 5]

    out_dir = os.path.join(
        PATH_TO_PDF_DATA, region, county, template, f"{method}_output", pdf_name
    )
    if mkdir:
        os.makedirs(out_dir, exist_ok=True)

    return {
        "dir": out_dir,
        "txt": os.path.join(out_dir, f"{pdf_name}.txt"),
        "json": os.path.join(out_dir, f"{pdf_name}.json"),
    }


def pdf_already_processed(pdf_path):
    """Check if PDF already processed."""
    return any(
        os.path.exists(paths["txt"]) and os.path.getsize(paths["txt"]) > 0
        for paths in [get_extraction_output_paths(pdf_path, m) for m in ["fitz", "tesseract"]]
    )


def process_and_save_pdf_text(
    pdf_path, process_only_manifests=False, override_method=None, override_pages=None
):
    """
    Process single PDF and extract text.
    If override_method is set, only that method is used for extraction (for special cases like missing pages).
    If override_pages is set, only those pages are processed.
    """
    print(f"Processing {pdf_path}")

    pages_to_process = override_pages
    if pages_to_process is None and process_only_manifests:
        manifest_pages = detect_manifest_pages_in_pdf(pdf_path, detect_orientation=True)
        if not manifest_pages:
            print("  No manifest pages found, skipping")
            paths = get_extraction_output_paths(pdf_path, "fitz", mkdir=True)
            with open(paths["txt"], "w") as f:
                f.write("no_manifests_found")
            with open(paths["json"], "w") as f:
                json.dump(
                    {"extraction_method": "fitz", "final_method": "fitz", "page_count": 0},
                    f,
                    indent=2,
                )
            return
        print(f"  Found {len(manifest_pages)} manifest page(s): {manifest_pages}")
        pages_to_process = manifest_pages

    if override_method:
        # Only extract using the override method
        result = extract_pdf_text_by_method(
            pdf_path, method=override_method, pages_to_process=pages_to_process
        )
        result["final_method"] = override_method
        print(f"  Saving {override_method} output (override)")
        paths = get_extraction_output_paths(pdf_path, override_method, mkdir=True)
        with open(paths["txt"], "w", encoding="utf-8") as f:
            f.write(result["result_text"])
        with open(paths["json"], "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Extraction complete via {override_method}")
        return

    results, final_method = auto_select_pdf_text_extraction(pdf_path, pages_to_process)
    methods_to_save = ["fitz", "tesseract"] + (
        ["llmwhisperer"] if final_method == "llmwhisperer" else []
    )

    for method in methods_to_save:
        if method in results:
            result = results[method]
            result["final_method"] = final_method
            print(f"  Saving {method} output")
            paths = get_extraction_output_paths(pdf_path, method, mkdir=True)
            with open(paths["txt"], "w", encoding="utf-8") as f:
                f.write(result["result_text"])
            with open(paths["json"], "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "extraction_method": method,
                        "final_method": final_method,
                        "page_count": result["result_text"].count("=== Page "),
                    },
                    f,
                    indent=2,
                )

    print(f"  Extraction complete via {final_method}")


def recover_missing_manifest_pages():
    """Process missing pages identified in manual discrepancy file."""
    df = pd.read_csv(
        os.path.join(
            REPO_BASE_DIR, "output_data", "2024_files_by_template_manual_discrepancies.csv"
        )
    )

    missing_rows = df[
        (df["notes"].astype(str).str.contains("missing", case=False, na=False))
        & df["missing_page_start"].notna()
        & df["missing_page_end"].notna()
    ]

    print(f"{len(missing_rows)} files with missing pages to process")

    for _, row in missing_rows.iterrows():
        page_start, page_end = int(row["missing_page_start"]), int(row["missing_page_end"])
        pdf_path = os.path.join(
            PATH_TO_PDF_DATA,
            "R5",
            row["county"],
            row["template"],
            "original",
            row["filename"],
        )

        print(f"\nProcessing: {row['filename']} pages {page_start}-{page_end}")
        pages_to_process = list(range(page_start, page_end + 1))
        # Use process_and_save_pdf_text with explicit override arguments for missing pages
        process_and_save_pdf_text(
            pdf_path, override_method="llmwhisperer", override_pages=pages_to_process
        )

    print("Missing pages processing complete")


def main(test_mode=TEST_MODE, process_only_manifests=False, process_missing_pages_flag=False):
    """Main processing function."""
    if process_missing_pages_flag:
        recover_missing_manifest_pages()
        return

    pdf_files = []
    region_path = os.path.join(PATH_TO_PDF_DATA, "R5")
    for county in os.listdir(region_path):
        county_path = os.path.join(region_path, county)
        if not os.path.isdir(county_path):
            continue

        print(f" Collecting PDFs for {county}")
        for template in os.listdir(county_path):
            folder_path = os.path.join(county_path, template, "original")
            pdf_files.extend(glob.glob(os.path.join(folder_path, "*.pdf")))
            pdf_files.extend(glob.glob(os.path.join(folder_path, "*.PDF")))
        print(f" Collected {len(pdf_files)} PDFs")
    files_to_process = [f for f in pdf_files if not pdf_already_processed(f)]
    print(f"{len(files_to_process)} of {len(pdf_files)} remaining")

    if test_mode:
        white_river = [f for f in files_to_process if "White River Dairy" in f]
        files_to_process = white_river[:1] if white_river else files_to_process[:1]
        print(f"Test mode: processing {len(files_to_process)} file(s)")

    if not files_to_process:
        print("No files to process")
        return

    for pdf_path in files_to_process:
        process_and_save_pdf_text(pdf_path, process_only_manifests=process_only_manifests)


if __name__ == "__main__":
    main(test_mode=False, process_only_manifests=False)
    main(test_mode=False, process_only_manifests=False, process_missing_pages_flag=True)
    print("OCR complete")
