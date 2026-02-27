import pandas as pd
import numpy as np
import os
import glob

from helpers_calculated_metrics import (
    calculate_consultant_metrics,
    parameters,
    calculate_metrics,
)
from helpers_pdf_metrics import (
    GDRIVE_BASE,
    YEARS,
    REGIONS,
    build_parameter_dicts,
    coerce_numeric_columns,
    extract_parameters_from_text,
    find_pdf_files,
    load_ocr_text,
)
from ca_cafo_compliance.helpers_geocoding import enrich_address_columns

# TODO:
# use https://github.com/reglab/cal-ff/tree/main/cacafo
# for facility list and location cross-checking

read_reports = True
consolidate_data = True


def find_fuzzy_match(row, cadd_facilities):
    """Find fuzzy match between row and CADD facilities."""
    if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
        return None

    distances = []
    for _, cadd_row in cadd_facilities.iterrows():
        if pd.isna(cadd_row["Latitude"]) or pd.isna(cadd_row["Longitude"]):
            continue

        lat1, lon1 = float(row["latitude"]), float(row["longitude"])
        lat2, lon2 = float(cadd_row["Latitude"]), float(cadd_row["Longitude"])
        distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111000

        name1_words = set(str(row["dairy_name"]).lower().split())
        name2_words = set(str(cadd_row["FacilityName"]).lower().split())
        common_words = name1_words.intersection(name2_words)

        if distance <= 100 and len(common_words) > 0:
            distances.append((distance, cadd_row))

    return min(distances, key=lambda x: x[0])[1] if distances else None


def main():
    """Main function to process all PDF files and extract data."""

    param_dicts = build_parameter_dicts()

    parameter_locations = pd.read_csv(
        "ca_cafo_compliance/data/parameter_locations.csv", dtype=str,
    )
    parameter_locations["item_order"] = parameter_locations["item_order"].astype(
        "Int64"
    )

    all_params = parameters["parameter_key"].unique().tolist()
    available_templates = parameter_locations["template"].unique()

    # Load county to region mapping
    county_region_df = pd.read_csv("ca_cafo_compliance/data/county_region.csv")
    county_region_map = dict(
        zip(
            county_region_df["county_name"],
            zip(county_region_df["region"], county_region_df["sub_region"]),
        )
    )

    # Discover all year/region/county/template folders from gdrive
    folders = sorted(glob.glob(os.path.join(GDRIVE_BASE, "*", "*", "*", "*", "")))
    for folder in folders:
        parts = folder.rstrip(os.sep).split(os.sep)
        # .../{year}/{region}/{county}/{template}
        year, region, county, template = parts[-4], parts[-3], parts[-2], parts[-1]
        if template not in available_templates:
            continue
        print(f"processing {template} in {county}")
        output_dir = f"ca_cafo_compliance/outputs/{year}/{region}/{county}/{template}"
        name = f"{county.capitalize()}_{year}_{template}"

        # Process files based on template type
        if template == "r8_csv" and region == "R8":
            # Process R8 CSV files
            animals_path = os.path.join(folder, "R8_animals.csv")
            animals_df = pd.read_csv(animals_path)

            # Create mapping from R8 column names to parameter names
            r8_template_params = parameter_locations[
                parameter_locations["template"] == "r8_csv"
            ]

            # Initialize all parameters as NA
            df = pd.DataFrame()
            for param in all_params:
                df[param] = np.nan

            # Map columns based on parameter_locations
            for _, row in r8_template_params.iterrows():
                if pd.notna(
                    row["row_search_text"]
                ):  # Only map if row_search_text exists
                    param_key = row["parameter_key"]
                    source_col = row["row_search_text"]
                    if source_col in animals_df.columns:
                        df[param_key] = animals_df[source_col]

            # Add region and template info
            df["region"] = region
            df["template"] = template
            df["year"] = year
        else:
            # Process PDF files
            pdf_files = find_pdf_files(folder)
            if not pdf_files:
                continue

            # Process PDFs sequentially
            results = []
            for pdf_file in pdf_files:
                result = {col: None for col in all_params}
                result["filename"] = os.path.basename(pdf_file)
                ocr_text = load_ocr_text(pdf_file)

                result.update(
                    extract_parameters_from_text(
                        ocr_text,
                        template,
                        parameter_locations,
                        param_dicts["key_to_type"],
                        param_dicts["key_to_default"],
                    )
                )

                if result is not None:
                    results.append(result)

            # Create DataFrame and initialize all parameters as NA
            df = pd.DataFrame(results)
            for param in all_params:
                if param not in df.columns:
                    df[param] = np.nan

        # Convert numeric columns and calculate metrics
        coerce_numeric_columns(df)
        df = calculate_metrics(df)

        # Add region and sub_region information
        df["region"] = region
        df["template"] = template
        df["year"] = year
        # Get sub_region from county mapping
        county_key = county.capitalize()
        if county_key in county_region_map:
            _, sub_region = county_region_map[county_key]
            df["sub_region"] = sub_region
        else:
            df["sub_region"] = None

        # Geocode addresses and extract location data
        enrich_address_columns(df, "dairy_address", cache)

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        for f in os.listdir(output_dir):
            if f.endswith(".csv"):
                os.remove(os.path.join(output_dir, f))
        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

    # Consolidate data
    if consolidate_data:
        # Load CADD data
        cadd_facilities = pd.read_csv(
            "ca_cafo_compliance/data/CADD/"
            "CADD_Facility General Information_v1.0.0.csv"
        )
        cadd_herd_size = pd.read_csv(
            "ca_cafo_compliance/data/CADD/" "CADD_Facility Herd Size_v1.0.0.csv"
        )

        # Collect all data for the all_master file
        all_dataframes = []

        for year in YEARS:
            base_path = f"ca_cafo_compliance/outputs/{year}"
            for region in REGIONS:
                region_path = os.path.join(base_path, region)
                if not os.path.exists(region_path):
                    continue

                # Collect and process CSV files
                csv_files = glob.glob(
                    os.path.join(region_path, "**/*.csv"), recursive=True
                )
                if not csv_files:
                    continue

                # Combine all CSVs
                dfs = []
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    df["year"] = year
                    df["region"] = region
                    df["filename"] = os.path.basename(csv_file)
                    path_parts = csv_file.split(os.sep)
                    region_idx = path_parts.index(region)
                    if region_idx + 2 < len(path_parts):
                        df["template"] = path_parts[region_idx + 2]
                    dfs.append(df)

                if not dfs:
                    continue

                # Combine all data
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df = combined_df.dropna(how="all")

                # Fuzzy match with CADD data
                fuzzy_matches = []
                for _, row in combined_df.iterrows():
                    match = find_fuzzy_match(row, cadd_facilities)
                    row_dict = row.to_dict()
                    if match is not None:
                        row_dict["CADDID"] = match["CADDID"]
                    fuzzy_matches.append(row_dict)

                # Create final dataframe
                final_df = pd.DataFrame(fuzzy_matches)

                # Merge with CADD herd size data
                if "CADDID" in final_df.columns:
                    current_year_herd = cadd_herd_size[
                        cadd_herd_size["Year"] == int(year)
                    ].copy()
                    if not current_year_herd.empty:
                        final_df = pd.merge(
                            final_df,
                            current_year_herd,
                            left_on="CADDID",
                            right_on="CADDID",
                            how="left",
                        )

                # Calculate consultant metrics for R5 and 2023
                if year == 2023 and region == "R5":
                    consultant_metrics = calculate_consultant_metrics(final_df)
                    consultant_metrics = consultant_metrics.rename(
                        columns=param_dicts["key_to_name"]
                    )
                    metrics_file = (
                        f"ca_cafo_compliance/outputs/consolidated/"
                        f"{year}_{region}_consultant_metrics.csv"
                    )
                    consultant_metrics.to_csv(metrics_file, index=False)

                # Convert to pretty names and save individual region files
                final_df_pretty = final_df.rename(columns=param_dicts["key_to_name"])
                output_file = (
                    f"ca_cafo_compliance/outputs/consolidated/"
                    f"{year}_{region}_master.csv"
                )
                final_df_pretty.to_csv(output_file, index=False)
                print(f"Saved consolidated data to {output_file}")
                print(f"Total records: {len(final_df)}")

                # Add to all_dataframes for the comprehensive file (use original column names)
                all_dataframes.append(final_df)

        # Create all_master file with all years and regions
        if all_dataframes:
            all_master_df = pd.concat(all_dataframes, ignore_index=True)

            # Now rename columns after concatenation
            all_master_df = all_master_df.rename(columns=param_dicts["key_to_name"])

            all_master_output_file = (
                "ca_cafo_compliance/outputs/consolidated/all_master.csv"
            )
            all_master_df.to_csv(all_master_output_file, index=False)
            print(f"Saved all_master data to {all_master_output_file}")
            print(f"Total records in all_master: {len(all_master_df)}")
        else:
            print("No data found to create all_master file")


if __name__ == "__main__":
    main()
