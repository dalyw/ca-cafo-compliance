
# CA CAFO Manifest Pre-processing

This repository processes California CAFO (Concentrated Animal Feeding Operation) manure and wastewater manifests. The current codebase handles PDF OCR, manifest page detection, parameter extraction, geocoding, and final CSV generation.

## Repository Layout

```
ca-cafo-compliance/
├── helpers.py
├── step0_summarize_pdfs.py
├── step1_pdf_ocr.py
├── step2_extract_manifest_parameters.py
├── step3_create_final_manifests.py
├── step4_consolidate_manifests.sh
├── data/
│   ├── county_region.csv
│   ├── parameter_locations.csv
│   ├── parameters.csv
│   ├── templates.csv
│   └── zipcode_to_county.csv
├── output_data/
└── requirements.txt
```

## Requirements

Install Python 3.9 or newer, then install the Python dependencies listed in [requirements.txt](requirements.txt).

If using the LLMWhisperer fallback in step 1, set up `LLMWHISPERER_API_KEY` in your environment. (Get free key online)

## Setup

1. Create and activate a virtual environment.
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows: venv\Scripts\activate
	```
2. Install the Python dependencies with `pip install -r requirements.txt`.
3. Make sure the data path in [helpers.py](helpers.py) points to your local copy of the Google Drive data folder. The current code uses a hardcoded `PATH_TO_PDF_DATA` value.

## Current Workflow

Run the scripts from the repository root.

1. [step0_summarize_pdfs.py]
```bash
python step1_pdf_ocr.py
```
(step0_summarize_pdfs.py) scans the Region 5 folder, counts manifest text files for each PDF, writes [output_data/2024_files_by_template.csv](output_data/2024_files_by_template.csv), updates the Google Drive copy of `2024_files_by_template_manual.csv`, and writes [output_data/2024_files_by_template_manual_discrepancies.csv](output_data/2024_files_by_template_manual_discrepancies.csv).

2. [step1_pdf_ocr.py](step1_pdf_ocr.py) extracts text from PDFs in `PATH_TO_PDF_DATA/R5/.../original/`. It saves OCR output under each PDF directory in `fitz_output/`, `tesseract_output/`, and, when needed, `llmwhisperer_output/`. It also supports a recovery mode that reprocesses missing pages listed in [output_data/2024_files_by_template_manual_discrepancies.csv](output_data/2024_files_by_template_manual_discrepancies.csv).

3. [step2_extract_manifest_parameters.py](step2_extract_manifest_parameters.py) reads OCR text files, identifies manifest page ranges, extracts fields, writes per-manifest text and PDF files next to each source OCR file, and saves [output_data/all_manifests_as_written_automatic.csv](output_data/all_manifests_as_written_automatic.csv).

4. [step3_create_final_manifests.py](step3_create_final_manifests.py) merges the validated manual CSV with the extracted manifest CSV, geocodes origin and destination addresses, backfills missing values, splits manure and wastewater rows, and writes [output_data/processed_manure_manifests.csv](output_data/processed_manure_manifests.csv) and [output_data/processed_wastewater_manifests.csv](output_data/processed_wastewater_manifests.csv).
5. [step4_consolidate_manifests.sh](step4_consolidate_manifests.sh) copies manifest text and PDF files into `all_manifests/<region>/<county>/`, preferring LLMWhisperer outputs over Tesseract when both exist.

The main scripts execute the `main()` functions when run directly, and they assume the repo sits alongside the Google Drive data path in [helpers.py](helpers.py).