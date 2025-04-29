# CA CAFO Compliance Data Explorer

This application provides an interactive dashboard to explore and analyze California Confined Animal Feeding Operation (CAFO) compliance data.

## Project Structure

```
.
├── ca_cafo_compliance/          # Main project directory
│   ├── read_reports.py         # PDF report processing
│   ├── consolidate_data.py     # Data consolidation utilities
│   ├── app.py                  # Streamlit dashboard application
│   ├── conversion_factors.py   # Unit conversion utilities
│   └── parameter_locations.csv # Parameter mapping data
├── data/                       # Input data directory
│   ├── 2023/                  # Data organized by year
│   │   ├── R2/               # Region-specific data
│   │   ├── R3/
│   │   ├── R5/
│   │   ├── R7/
│   │   └── R8/
│   └── 2024/
├── outputs/                    # Generated output files
│   ├── consolidated/          # Consolidated CSV files
│   ├── 2023/                 # Processed data by year
│   └── 2024/
└── pyproject.toml             # Poetry dependency management

```

## Setup

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/ca-cafo-compliance.git
cd ca-cafo-compliance
```

3. Install dependencies using Poetry:
```bash
poetry install
```

4. Activate the Poetry shell:
```bash
poetry shell
```

## Running the Application

1. Make sure you have processed data in the `outputs/consolidated/` directory.

2. Start the Streamlit app:
```bash
streamlit run ca_cafo_compliance/app.py
```

3. The application will open in your default web browser. If it doesn't, navigate to the URL shown in the terminal (typically http://localhost:8501).

## Features

- Filter data by year, region, and county
- View key metrics and statistics
- Interactive visualizations of records by region and county
- Download filtered data as CSV
- Explore raw data in a tabular format

## Data Structure

The application expects CSV files in the `outputs/consolidated/` directory. These files should contain processed CAFO compliance data with the following columns:
- Year
- Region
- County
- Other relevant metrics

## Contributing

Feel free to submit issues and enhancement requests!