import pandas as pd
import numpy as np
import os
import glob
import gc
import sys

from helper_functions.metrics_helpers import *
from helper_functions.read_report_helpers import YEARS, REGIONS, clean_common_errors, get_default_value, find_parameter_value
from helper_functions.geocoding_helpers import (
    geocode_address,
    find_cached_address,
    extract_address_components,
    load_geocoding_cache,
)

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

    snake_to_pretty = dict(
        zip(parameters["parameter_key"], parameters["parameter_name"])
    )
    params = {
        "snake_to_pretty": snake_to_pretty,
        "data_types": dict(zip(parameters["parameter_key"], parameters["data_type"])),
        "defaults": dict(zip(parameters["parameter_key"], parameters["default"])),
    }

    dtype_dict = {
        col: str
        for col in [
            "region",
            "template",
            "parameter_key",
            "page_search_text",
            "search_direction",
            "row_search_text",
            "column_search_text",
            "ignore_before",
            "value_pattern",
        ]
    }
    dtype_dict["item_order"] = "Int64"
    parameter_locations = pd.read_csv(
        "ca_cafo_compliance/data/parameter_locations.csv", dtype=dtype_dict
    )

    all_params = parameters["parameter_key"].unique().tolist()
    available_templates = parameter_locations["template"].unique()

    cache = load_geocoding_cache()

    # Load zipcode to county mapping
    zipcode_df = pd.read_csv("ca_cafo_compliance/data/zipcode_to_county.csv")
    zipcode_df = zipcode_df[["zip", "county_name"]].drop_duplicates()
    zipcode_df["zip"] = zipcode_df["zip"].astype(str)

    # Load county to region mapping
    county_region_df = pd.read_csv("ca_cafo_compliance/data/county_region.csv")
    county_region_map = dict(
        zip(
            county_region_df["county_name"],
            zip(county_region_df["region"], county_region_df["sub_region"]),
        )
    )

    # Process reports
    for year in YEARS:
        base_data_path = f"ca_cafo_compliance/data/{year}"
        base_output_path = f"ca_cafo_compliance/outputs/{year}"

        for region in REGIONS:
            region_data_path = os.path.join(base_data_path, region)
            region_output_path = os.path.join(base_output_path, region)
            if not os.path.exists(region_data_path):
                continue

            for county in [
                d
                for d in os.listdir(region_data_path)
                if os.path.isdir(os.path.join(region_data_path, d))
            ]:
                county_data_path = os.path.join(region_data_path, county)
                county_output_path = os.path.join(region_output_path, county)

                for template in [
                    d
                    for d in os.listdir(county_data_path)
                    if os.path.isdir(os.path.join(county_data_path, d))
                ]:
                    print(f"processing {template} in {county}")
                    if template not in available_templates:
                        continue

                    folder = os.path.join(county_data_path, template)
                    output_dir = os.path.join(county_output_path, template)
                    name = f"{county.capitalize()}_{year}_{template}"
                    template_params = parameter_locations[
                        parameter_locations["template"] == template
                    ]

                    # Process files based on template type
                    if template == "r8_csv" and region == "R8":
                        # Process R8 CSV files
                        animals_path = os.path.join(
                            base_data_path, "R8", "all_r8", "r8_csv", "R8_animals.csv"
                        )
                        manure_path = os.path.join(
                            base_data_path, "R8", "all_r8", "r8_csv", "R8_manure.csv"
                        )
                        animals_df = pd.read_csv(animals_path)
                        manure_df = pd.read_csv(manure_path)

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
                        ocr_dir = os.path.join(folder, "ocr_output")
                        ai_ocr_dir = os.path.join(folder, "ai_ocr_output")
                        if not os.path.exists(ocr_dir) and not os.path.exists(
                            ai_ocr_dir
                        ):
                            continue

                        pdf_files = []
                        for text_file in glob.glob(os.path.join(ocr_dir, "*.txt")):
                            pdf_name = os.path.basename(text_file).replace(
                                ".txt", ".pdf"
                            )
                            pdf_path = os.path.join(folder, "original", pdf_name)
                            if os.path.exists(pdf_path):
                                pdf_files.append(pdf_path)

                        if os.path.exists(ai_ocr_dir):
                            for text_file in glob.glob(
                                os.path.join(ai_ocr_dir, "*.txt")
                            ):
                                pdf_name = os.path.basename(text_file).replace(
                                    ".txt", ".pdf"
                                )
                                pdf_path = os.path.join(folder, "original", pdf_name)
                                if (
                                    os.path.exists(pdf_path)
                                    and pdf_path not in pdf_files
                                ):
                                    pdf_files.append(pdf_path)

                        if not pdf_files:
                            continue

                        # Process PDFs sequentially
                        results = []
                        for pdf_file in pdf_files:
                            result = {col: None for col in all_params}
                            result["filename"] = os.path.basename(pdf_file)
                            ocr_text = None

                            pdf_dir = os.path.dirname(pdf_file)
                            parent_dir = os.path.dirname(pdf_dir)
                            pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]

                            for ocr_dir in ["llmwhisperer_output", "tesseract_output", "fitz_output"]:
                                text_file = os.path.join(parent_dir, ocr_dir, f"{pdf_name}.txt")
                                if os.path.exists(text_file):
                                    with open(text_file, "r") as f:
                                        text = f.read()
                                        ocr_text = clean_common_errors(text)

                            if not ocr_text:
                                # Use defaults for all parameters if OCR text is missing
                                for _, row in template_params.iterrows():
                                    param_key = row["parameter_key"]
                                    result[param_key] = get_default_value(param_key, params["data_types"], params["defaults"])
                                return result
                            # Process main report parameters
                            for _, row in template_params.iterrows():
                                param_key = row["parameter_key"]
                                value = find_parameter_value(ocr_text, row, params["data_types"], params["defaults"])
                                result[param_key] = value

                            if result is not None:
                                results.append(result)

                        # Create DataFrame and initialize all parameters as NA
                        df = pd.DataFrame(results)
                        for param in all_params:
                            if param not in df.columns:
                                df[param] = np.nan

                    # Convert numeric columns and calculate metrics
                    for col in df.columns:
                        if params["data_types"].get(col) == "numeric":
                            df[col] = pd.to_numeric(df[col], errors="coerce")
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
                    (
                        df["latitude"],
                        df["longitude"],
                        df["city"],
                        df["state"],
                        df["zip"],
                        df["county"],
                    ) = (None, None, None, None, None, None)

                    for idx, row in df.iterrows():
                        if pd.isna(row["dairy_address"]):
                            continue
                        lat, lng = geocode_address(row["dairy_address"], cache)
                        if lat is not None and lng is not None:
                            df.at[idx, "latitude"] = lat
                            df.at[idx, "longitude"] = lng
                            # Get the formatted address from cache
                            cached_addr = find_cached_address(
                                row["dairy_address"], cache
                            )
                            if cached_addr and "address" in cache[cached_addr]:
                                formatted_address = cache[cached_addr]["address"]
                                df.at[
                                    idx, "address"
                                ] = formatted_address  # REPLACING machine-read with formatted address
                                # Extract city, state, and zip from formatted address
                                city, state, zip_code = extract_address_components(
                                    formatted_address
                                )
                                df.at[idx, "city"] = city
                                df.at[idx, "state"] = state
                                df.at[idx, "zip"] = zip_code
                                # Look up county from zip code
                                if zip_code:
                                    county_match = zipcode_df[
                                        zipcode_df["zip"] == zip_code
                                    ]
                                    if not county_match.empty:
                                        df.at[idx, "county"] = county_match.iloc[0][
                                            "county_name"
                                        ]

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
            if not os.path.exists(base_path):
                continue

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
                        columns=params["snake_to_pretty"]
                    )
                    metrics_file = (
                        f"ca_cafo_compliance/outputs/consolidated/"
                        f"{year}_{region}_consultant_metrics.csv"
                    )
                    consultant_metrics.to_csv(metrics_file, index=False)

                # Convert to pretty names and save individual region files
                final_df_pretty = final_df.rename(columns=params["snake_to_pretty"])
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
            all_master_df = all_master_df.rename(columns=params["snake_to_pretty"])

            all_master_output_file = (
                "ca_cafo_compliance/outputs/consolidated/all_master.csv"
            )
            all_master_df.to_csv(all_master_output_file, index=False)
            print(f"Saved all_master data to {all_master_output_file}")
            print(f"Total records in all_master: {len(all_master_df)}")
        else:
            print("No data found to create all_master file")

    # Cleanup and clear any large variables
    gc.collect()
    for name in list(sys.modules.keys()):
        if name.startswith("ca_cafo_compliance"):
            try:
                del sys.modules[name]
            except (KeyError, AttributeError):
                pass

    print("Script completed")


if __name__ == "__main__":
    main()
