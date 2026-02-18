import csv
import os
import pandas as pd
from pathlib import Path
from ca_cafo_compliance.helpers_pdf_metrics import GDRIVE_BASE

BASE_DIR = Path(GDRIVE_BASE)
LOCAL_BASE_DIR = Path(__file__).resolve().parent
REGION = "R5"
COUNTIES = [
    "fresno_madera",
    "kern",
    "kings",
    "tulare_west",
    "tulare_east",
    "rancho_cordova",
]


def get_files_by_template(year, output_path, gdrive_output_path):
    """Returns a list of dicts with county, template, and filename"""
    files_list = []
    year_dir = BASE_DIR / str(year) / REGION
    # data is structured under ca_cafo_manifests/year/region/county/template
    for county in COUNTIES:
        county_dir = year_dir / county
        if county_dir.exists():  # Each subdirectory in county (except . files) is a template
            for template_dir in county_dir.iterdir():
                if template_dir.name.startswith(".") or not template_dir.is_dir():
                    continue

                # PDFs are in the 'original' folder
                original_dir = template_dir / "original"
                tesseract_dir = template_dir / "tesseract_output"
                # fitz_dir = template_dir / "fitz_output"
                llmwhisperer_dir = template_dir / "llmwhisperer_output"
                if original_dir.exists():
                    for pdf_file in original_dir.iterdir():
                        if pdf_file.suffix.lower() == ".pdf":
                            file_data = {
                                "county": county,
                                "template": template_dir.name,
                                "filename": pdf_file.name,
                            }
                        # count number of manifest_{#}.txt files by checking llmwhisperer then tesseract
                        manifest_count = 0
                        for dir_path in [llmwhisperer_dir, tesseract_dir]:
                            facility_dir = dir_path / pdf_file.stem
                            if facility_dir.exists():
                                count = len(list(facility_dir.glob("manifest_*.txt")))
                                if count > 0:
                                    manifest_count = count
                                    break
                        file_data["manifest_count"] = manifest_count
                        files_list.append(file_data)

    # Save files to csv
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["county", "template", "filename", "manifest_count"])
        writer.writeheader()
        writer.writerows(files_list)

    # save to gdrive BASE_DIR
    with open(gdrive_output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["county", "template", "filename", "manifest_count"])
        writer.writeheader()
        writer.writerows(files_list)

    return files_list


# Get all 2023 files by template and save to CSV
output_path_2023 = LOCAL_BASE_DIR / "outputs" / "2023_files_by_template.csv"
output_path_2024 = LOCAL_BASE_DIR / "outputs" / "2024_files_by_template.csv"
gdrive_output_path_2023 = os.path.join(GDRIVE_BASE, "2023_files_by_template.csv")
gdrive_output_path_2024 = os.path.join(GDRIVE_BASE, "2024_files_by_template.csv")

files_2023 = get_files_by_template(2023, output_path_2023, gdrive_output_path_2023)
files_2024 = get_files_by_template(2024, output_path_2024, gdrive_output_path_2024)
print(
    f"{sum(file['manifest_count'] for file in files_2023)} manifests from {len(files_2023)} files in 2023"
)
print(
    f"{sum(file['manifest_count'] for file in files_2024)} manifests from {len(files_2024)} files in 2024"
)

# Convert files_2024 list to DataFrame for easier lookup
files_2024_df = pd.DataFrame(files_2024)

# load "2024_files_by_template_manual_counts.csv"
# update the manifest_count to match the CURRENT counts
gdrive_manual_counts_path = os.path.join(GDRIVE_BASE, "2024_files_by_template_manual_counts.csv")
manual_counts = pd.read_csv(gdrive_manual_counts_path)
for index, row in manual_counts.iterrows():
    # update manifest_count in manual_counts to match files_2024 for that manifest
    matching_rows = files_2024_df[files_2024_df["filename"] == row["filename"]]
    if len(matching_rows) > 0:
        manual_counts.at[index, "manifest_count"] = matching_rows["manifest_count"].values[0]
manual_counts.to_csv(gdrive_manual_counts_path, index=False)

# Print rows of manual_counts where manifest_count ~= manual_count
print(manual_counts[manual_counts["manifest_count"] != manual_counts["manual_count"]])
# save this as a separate csv for what needs to be manually adjusted
manual_counts[manual_counts["manifest_count"] != manual_counts["manual_count"]].to_csv(
    LOCAL_BASE_DIR / "outputs" / "2024_files_by_template_manual_counts_discrepancies.csv",
    index=False,
)

# Count total facilities with manifests in manual and auto.
# Print of facilities with the wrong number of manifests flagged in auto
manual_counts["manifest_count"] != manual_counts["manual_count"]
print(
    f"Total facilities with the wrong number of manifests flagged in auto: {len(manual_counts[manual_counts['manifest_count'] != manual_counts['manual_count']])}"
)
# give as %
percent_wrong = (
    len(manual_counts[manual_counts["manifest_count"] != manual_counts["manual_count"]])
    / len(manual_counts)
    * 100
)
print(f"Facilities wit % wrong {percent_wrong}%")

# total manifest counts for manual and auto
auto_count = manual_counts["manifest_count"].sum()
manual_count_total = manual_counts["manual_count"].sum()

# missing manifests
missing_rows = manual_counts[manual_counts["manual_count"] > manual_counts["manifest_count"]]
missing_count = (missing_rows["manual_count"] - missing_rows["manifest_count"]).sum()
percent_missing = (missing_count / manual_count_total * 100) if manual_count_total > 0 else 0

# false positives
false_positive_rows = manual_counts[
    manual_counts["manual_count"] < manual_counts["manifest_count"]
]
false_positive_count = (
    false_positive_rows["manifest_count"] - false_positive_rows["manual_count"]
).sum()
percent_false_positives = (
    (false_positive_count / manual_count_total * 100) if manual_count_total > 0 else 0
)

print(f" manual count {manual_count_total}")
print(f" auto count {auto_count}")
print(f"Missing {missing_count} ({percent_missing:.2f}%)")
print(f"False positives {false_positive_count} ({percent_false_positives:.2f}%)")

# print min and max of manual_counts['manual_count']
print(f"Min manual count {manual_counts['manual_count'].min()}")
print(f"Max manual count {manual_counts['manual_count'].max()}")
