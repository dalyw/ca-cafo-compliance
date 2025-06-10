# CA CAFO Compliance Data Analysis

This project analyzes California CAFO (Concentrated Animal Feeding Operation) compliance data from annual reports. It processes PDF reports, extracts data using OCR, and performs various analyses on manure management, nutrient application, and compliance metrics.

## Project Structure

```
ca-cafo-compliance/
├── ca_cafo_compliance/
│   ├── __init__.py
│   ├── app.py                 # Streamlit web application
│   ├── consolidate_data.py    # Data consolidation and geocoding
│   ├── conversion_factors.py  # Constants and conversion factors
│   ├── parameter_locations.csv # OCR parameter locations
│   ├── parameters.csv         # Parameter definitions
│   ├── read_reports.py        # PDF processing and data extraction
│   └── requirements.txt       # Python dependencies
├── data/                      # Raw PDF reports
│   └── 2023/                  # Reports by year
│       └── Region_5/          # Reports by region
│           └── Merced/        # Reports by county
│               └── Template_1/ # Reports by template type
│                   ├── original/      # Original PDFs
│                   ├── ocr_output/    # OCR text output
│                   └── handwriting_ocr_output/ # Handwriting OCR output
├── outputs/                   # Processed data
│   ├── 2023/                  # Outputs by year
│   │   └── Region_5/          # Outputs by region
│   │       └── Merced/        # Outputs by county
│   │           └── Template_1/ # Outputs by template type
│   │               ├── Merced_2023_Template_1.csv
│   │               └── Merced_2023_Template_1_manifests.csv
│   └── consolidated/          # Consolidated data files
│       ├── 2023_Region_5_master.csv
│       └── geocoding_cache.json
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r ca_cafo_compliance/requirements.txt
```

2. Place PDF reports in the appropriate directories under `data/` following the structure:
```
data/
└── YEAR/
    └── REGION/
        └── COUNTY/
            └── TEMPLATE/
                └── original/
                    └── *.pdf
```

## Usage

1. Extract text from PDF reports:

2. Process text from extracted reports and consolidate data:
```bash
python -m ca_cafo_compliance.read_reports
```

3. Run the Streamlit app locally (or at ):
```bash
streamlit run ca_cafo_compliance/app.py
```

## Features

- PDF text extraction using OCR
- Handwriting recognition for handwritten fields
- Data extraction from structured and unstructured text
- Geocoding of facility addresses
- Interactive data visualization
- Compliance metric calculations
- Manure and nutrient tracking
- Wastewater analysis

## Data Processing Pipeline

1. **PDF Text Extraction** (`extract_pdf_text.py`):
   - Extracts text from PDFs using OCR

1a. **Supplemental Handwitten PDF Text Extraction**:
   - Used HandwritingOCR to process handwritten reports
   - Manually saved these reports into the relevant region/county/template folder under "handwriting_ocr_output/"

2. **Data Consolidation** (`read_reports.py`):
   - Extracts data based on parameter locations
   - Calculates compliance metrics
   - Combines data from multiple reports
   - Geocodes facility addresses
   - Creates master datasets by region

3. **Data Visualization** (`app.py`):
   - Interactive web interface using Streamlit
   - Embedded maps
   - Compliance metric analysis
   - Data filtering and exploration


## License

This project is licensed under the MIT License - see the LICENSE file for details.