# CA CAFO Compliance Data Analysis

This project analyzes California CAFO (Concentrated Animal Feeding Operation) compliance data from annual reports. It processes PDF reports, extracts data using OCR, and performs various analyses on manure management, nutrient application, and compliance metrics.

## Project Structure

```
ca-cafo-compliance/
├── ca_cafo_compliance/
│   ├── __init__.py
│   ├── app.py                 # Streamlit web application for data visualization
│   ├── extract_pdf_text.py    # PDF text extraction using OCR
│   ├── read_reports.py        # PDF processing and data extraction
│   ├── helper_functions/      # Utility functions for data processing
│   ├── data/                  # Data storage and configuration
│   │   ├── parameter_locations.csv # OCR parameter locations for each template
│   │   └── parameters.csv     # Parameter definitions and validation rules
│   ├── images/               # Static images for the web application
│   ├── outputs/              # Processed data and analysis results
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

1. Install system dependencies:
```bash
# macOS
brew install tesseract
brew install poppler  # Required for pdf2image

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Required Python packages:
```
# Core dependencies
pandas>=1.5.0
numpy>=1.21.0
streamlit>=1.22.0
plotly>=5.13.0

# PDF Processing
pytesseract>=0.3.10
pdf2image>=1.16.0
opencv-python>=4.7.0
PyPDF2>=3.0.0

# Data Processing
requests>=2.28.0
dash>=2.9.0
dash-core-components>=2.0.0
dash-html-components>=2.0.0

# Optional: Google Document AI (for alternate OCR)
google-cloud-documentai>=2.20.0
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
```bash
python -m ca_cafo_compliance.extract_pdf_text
```

2. Process text from extracted reports and consolidate data:
```bash
python -m ca_cafo_compliance.read_reports
```

3. Run the Streamlit app locally (or at https://cal-cafo-compliance.streamlit.app):
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
   - Processes both typed and handwritten text
   - Saves extracted text in structured format

1a. **Supplemental Handwritten PDF Text Extraction**:
   - Uses HandwritingOCR to process handwritten reports
   - Manually saved these reports into the relevant region/county/template folder under "handwriting_ocr_output/"

2. **Data Extraction and Consolidation** (`read_reports.py`):
   - Uses `parameter_locations.csv` to locate values in the extracted text:
     - Each template type has specific coordinates for parameter values
     - Parameters are defined with x,y coordinates and expected data types
     - Values are extracted based on these coordinates and validated
   - Calculates compliance metrics
   - Combines data from multiple reports
   - Geocodes facility addresses
   - Creates master datasets by region

3. **Data Visualization** (`app.py`):
   - Interactive web interface using Streamlit
   - Embedded maps showing facility locations
   - Compliance metric analysis and visualization
   - Data filtering and exploration tools
   - Export capabilities for processed data

## Parameter Location System

The parameter location system is a key component of the data extraction process:

1. **Template Definition**:
   - Each report template type has its own set of parameter definitions
   - Parameters are defined in `parameter_locations.csv` with:
     - Search text patterns to locate values
     - Expected data types

2. **Value Extraction**:
   - The script reads the OCR output text
   - For each parameter:
     - Searches for the defined search text pattern
     - Locates the associated value based on search direction, "left", "right", "above" or "below"
     - Applies value patterns to extract and validate the data
   - Handles missing or invalid values appropriately

3. **Calculated Parameter Calculations**:
   - Calculates additional parameters, e.g. Wastewater to Milk Ratio, from reported values

## License

This project is licensed under the MIT License - see the LICENSE file for details.