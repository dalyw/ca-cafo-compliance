
# CA CAFO Manifest Pre-processing

This repository processes and analyzes California CAFO (Concentrated Animal Feeding Operation) manure and wastewater manifests. It includes tools for PDF OCR, parameter extraction, geocoding, and visualization.

Code drafted in part based on psuedo-code prompts to Claude Sonnet and GPT 4.1 via GitHub CoPilot. AI generated code is always manually reviewed for accuracy.

## Folder Structure

```
ca-cafo-compliance/
├── ca_cafo_compliance/
│   ├── __init__.py
│   ├── helpers_geocoding.py         # Address normalization and geocoding utilities
│   ├── helpers_pdf_metrics.py       # Parameter definitions and PDF parsing helpers
│   ├── helpers_plotting.py          # Plotting and visualization helpers
│   ├── manifest_app.py              # Streamlit app for interactive maps
│   ├── step0_summarize_pdfs.py      # Checking total manifest counts
│   ├── step1_pdf_ocr.py             # Extract text from PDFs using OCR
│   ├── step2_extract_manifest_parameters.py  # Extract structured data from OCR text
│   ├── step3_create_final_manifests.py       # Merge, clean, and finalize manifest data
│   ├── step4_plotting.py            # Generate plots and summary figures
│   ├── data/
│   │   ├── county_region.csv
│   │   ├── parameter_locations.csv
│   │   ├── parameters.csv
│   │   ├── templates.csv
│   │   ├── zipcode_to_county.csv
│   ├── compiled_data/                     # Output CSVs, maps, and intermediate files
│   └── local/                       # Local scripts and violation conversion tools
├── requirements.txt                 # Python dependencies
```

## Setup Instructions

1. **Install Python 3.9+** (Anaconda/miniconda recommended)

2. **Create and activate a virtual environment:**
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows: venv\Scripts\activate
	```

3. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

## Main Processing Steps

Run each script from the project root:

1. **OCR PDFs:**
	```bash
	python ca_cafo_compliance/step1_pdf_ocr.py
	```
Data is referenced from `Manure Trucking Network Analysis` folder. Update based on PDF path as needed.

2. **Extract manifest parameters:**
	```bash
	python ca_cafo_compliance/step2_extract_manifest_parameters.py
	```
3. **Create final manifests:**
	```bash
	python ca_cafo_compliance/step3_create_final_manifests.py
	```
4. **Generate plots and figures:**
	```bash
	python ca_cafo_compliance/step4_plotting.py
	```

To view the manifest destinations:

```bash
streamlit run ca_cafo_compliance/manifest_app.py
```

or
```bash
open ca_cafo_compliance/compiled_data/2024_destination_address_map.html
'''