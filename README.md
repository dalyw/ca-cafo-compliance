# CA CAFO Compliance Analysis

This project analyzes California Concentrated Animal Feeding Operations (CAFO) compliance data, focusing on nutrient management and reporting requirements.

## Project Overview

The project processes and analyzes CAFO reports from CA Regional Water Boards to track compliance with nutrient management plans and manure application requirements. It includes tools for:

- Reading and parsing reports based on general report structure, and potentially additioanl consulting report structures
- Calculating manure and milk production metrics
- Processing nutrient application data
- Visualizing reporting and likely violations data

## Project Structure

```
.
├── ca_cafo_compliance/          # Main project directory
│   ├── read_r2_reports.py      # R2 report processing
│   ├── read_r5_reports.py      # R5 report processing
│   ├── calculate_manure_milk.py # Manure and milk calculations
│   ├── plotting.py             # Data visualization
│   ├── conversion_factors.py   # Unit conversion utilities
│   └── parameter_locations.csv # Parameter mapping data
├── data/                       # Input data directory
├── outputs/                    # Generated output files
└── pyproject.toml             # Project dependencies
```

## Setup

This project uses Poetry for dependency management. To set up the project:

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository and install dependencies:
   ```bash
   git clone [repository-url]
   cd ca-cafo-compliance
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Dependencies

- Python 3.11+
- pandas
- matplotlib
- pdfplumber
- pypdf

## Usage

The project provides several Python scripts for different aspects of CAFO compliance analysis:

- `read_r2_reports.py`: Process R2 annual reports
- `read_r5_reports.py`: Process R5 nutrient management reports
- To add additional regions
- `calculate_manure_milk.py`: Calculate manure and milk production metrics
- `plotting.py`: Generate visualizations of compliance data

## Data Sources

- R2 Annual Reports
- R5 Nutrient Management Reports
- Parameter location mapping data

## Outputs

Generated outputs are stored in the `outputs/` directory and include:
- Processed compliance data
- Visualizations
- Analysis reports

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License