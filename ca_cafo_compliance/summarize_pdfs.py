import csv
import os
from pathlib import Path
from helper_functions.read_report_helpers import GDRIVE_BASE

BASE_DIR = Path(GDRIVE_BASE)
LOCAL_BASE_DIR = Path(__file__).resolve().parent
REGION = "R5"
COUNTIES = ["fresno_madera", "kern", "kings", "tulare_west", "tulare_east", "rancho_cordova"]

def get_files_by_template(year, output_path):
    """ Returns a list of dicts with county, template, and filename """
    files_list = []
    year_dir = BASE_DIR / str(year) / REGION
    # data is structured under ca_cafo_manifests/year/region/county/template
    for county in COUNTIES:
        county_dir = year_dir / county
        if county_dir.exists():  # Each subdirectory in county (except . files) is a template
            for template_dir in county_dir.iterdir():
                if template_dir.name.startswith('.') or not template_dir.is_dir():
                    continue
                                    
                # PDFs are in the 'original' folder
                original_dir = template_dir / "original"
                # marker_dir = template_dir / "marker_output"
                tesseract_dir = template_dir / "tesseract_output"
                # fitz_dir = template_dir / "fitz_output"
                llmwhisperer_dir = template_dir / "llmwhisperer_output"
                if original_dir.exists():
                    for pdf_file in original_dir.iterdir():
                        if pdf_file.suffix.lower() == '.pdf':
                            file_data = {
                                'county': county,
                                'template': template_dir.name,
                                'filename': pdf_file.name,
                            }
                        # count number of manifest_{#}.txt files by checking llmwhisperer then tesseract
                        manifest_count = 0
                        for dir_path in [llmwhisperer_dir, tesseract_dir]:
                            facility_dir = dir_path / pdf_file.stem
                            if facility_dir.exists():
                                manifest_count = len(list(facility_dir.glob('manifest_*.txt')))
                                break
                        file_data['manifest_count'] = manifest_count
                        files_list.append(file_data)
    
    # Save files to csv
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['county', 'template', 'filename', 'manifest_count'])
        writer.writeheader()
        writer.writerows(files_list)
    
    # save to gdrive BASE_DIR
    gdrive_output_path = os.path.join(GDRIVE_BASE, output_path.name)
    with open(gdrive_output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['county', 'template', 'filename', 'manifest_count'])
        writer.writeheader()
        writer.writerows(files_list)

    return files_list

    
#Get all 2023 files by template and save to CSV
output_path_2023 = LOCAL_BASE_DIR / "outputs" / "2023_files_by_template.csv"
output_path_2024 = LOCAL_BASE_DIR / "outputs" / "2024_files_by_template.csv"
files_2023 = get_files_by_template(2023, output_path_2023)
files_2024 = get_files_by_template(2024, output_path_2024)
print(f"{sum(file['manifest_count'] for file in files_2023)} manifests from {len(files_2023)} files in 2023")
print(f"{sum(file['manifest_count'] for file in files_2024)} manifests from {len(files_2024)} files in 2024")