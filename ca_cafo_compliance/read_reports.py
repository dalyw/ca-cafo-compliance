import pandas as pd
import numpy as np
import os
import glob
from ca_cafo_compliance.helper_functions.read_report_helpers import cf
from ca_cafo_compliance.helper_functions.read_report_helpers import *
from ca_cafo_compliance.helper_functions.geocoding_helpers import *
import multiprocessing as mp
from functools import partial
import json
import re

read_reports = True
consolidate_data = True

def calculate_metrics(df):
    """Calculate all metrics for the dataframe."""
    # Calculate annual milk production
    df['avg_milk_prod_kg_per_cow'] = df['avg_milk_lb_per_cow_day'] * cf['LBS_TO_KG']
    df['avg_milk_prod_l_per_cow'] = df['avg_milk_lb_per_cow_day'] * cf['LBS_TO_KG'] * cf['KG_PER_L_MILK']
    df['reported_annual_milk_production_l'] = df['avg_milk_lb_per_cow_day'] * cf['LBS_TO_KG'] * cf['KG_PER_L_MILK'] * (df['avg_milk_cows'] + df['avg_dry_cows']) * cf['DAYS_PER_YEAR']

    # Calculate herd size
    df['total_herd_size'] = (
        df['avg_milk_cows'] + df['avg_dry_cows'] + df['avg_bred_heifers'] + df['avg_heifers'] + 
        df['avg_calves_4_6_mo'] + df['avg_calves_0_3_mo'] + df['avg_other']
    )

    # Calculate nutrient metrics
    nutrient_types = ["n", "p", "k", "salt"]
    for nutrient in nutrient_types:
        # Total Applied
        dry_key = f"applied_{nutrient}_dry_manure_lbs"
        ww_key = f"applied_ww_{nutrient}_lbs"
        df[f'total_applied_{nutrient}_lbs'] = df[dry_key] + df[ww_key]

        if nutrient == "n":
            dry_key_reported = "total_manure_gen_n_after_nh3_losses_lbs"
        else:
            dry_key_reported = f"total_manure_gen_{nutrient}_lbs"
        ww_key_reported = f"total_ww_gen_{nutrient}_lbs"
        df[f'total_reported_{nutrient}_lbs'] = df[dry_key_reported] + df[ww_key_reported]

        # Unaccounted for
        exports_key = f"total_exports_{nutrient}_lbs"
        df[f'unaccounted_for_{nutrient}_lbs'] = (
            df[dry_key_reported] + 
            df[ww_key_reported] - 
            df[f'total_applied_{nutrient}_lbs'] - 
            df[exports_key]
        )

    # Calculate wastewater metrics
    df['total_ww_gen_liters'] = df["total_ww_gen_gals"] * 3.78541
    df['wastewater_to_reported'] = df['total_ww_gen_liters'] / df['reported_annual_milk_production_l'].replace(0, np.nan)

    # Calculate estimated annual milk production (L)
    est_milk_col = 'estimated_annual_milk_production_l'
    if 'avg_milk_lb_per_cow_day' in df.columns and df['avg_milk_lb_per_cow_day'].notna().any():
        df[est_milk_col] = df['avg_milk_lb_per_cow_day'] * cf['LBS_TO_KG'] * cf['KG_PER_L_MILK'] * (df['avg_milk_cows'] + df['avg_dry_cows']) * cf['DAYS_PER_YEAR']
    else:
        df[est_milk_col] = cf['DEFAULT_MILK_PRODUCTION'] * (df['avg_milk_cows'] + df['avg_dry_cows']) * cf['DAYS_PER_YEAR'] * cf['LBS_TO_KG'] * cf['KG_PER_L_MILK']

    # Calculate estimated wastewater generation (L) and wastewater to estimated milk retio
    df['wastewater_estimated'] = df[est_milk_col] * cf['L_WW_PER_L_MILK_LOW']
    df['wastewater_to_estimated'] = df['total_ww_gen_liters'] / df[est_milk_col].replace(0, np.nan)
    df['wastewater_ratio_discrepancy'] = df['wastewater_to_estimated'] - df['wastewater_to_reported']

    # Calculate manure metrics
    denom = (
        df["avg_milk_cows"] + df["avg_dry_cows"] +
        (df["avg_bred_heifers"] + df["avg_heifers"]) * cf['HEIFER_FACTOR'] +
        (df["avg_calves_4_6_mo"] + df["avg_calves_0_3_mo"]) * cf['CALF_FACTOR']
    )
    df['calculated_manure_factor'] = df["total_manure_gen_tons"] / denom
    df.loc[denom <= 0, 'calculated_manure_factor'] = np.nan
    df['manure_factor_discrepancy'] = df['calculated_manure_factor'] - cf['MANURE_FACTOR_AVERAGE']

    # Calculate nitrogen metrics
    df['usda_nitrogen_estimate_lbs'] = df["total_manure_gen_tons"] * cf['MANURE_N_CONTENT'] * 2000  # tons to lbs
    df['ucce_nitrogen_estimate_lbs'] = (
        df["avg_milk_cows"] * cf['MANURE_FACTOR_AVERAGE'] +
        df["avg_dry_cows"] * cf['MANURE_FACTOR_AVERAGE'] +
        (df["avg_bred_heifers"] + df["avg_heifers"]) * cf['HEIFER_FACTOR'] * cf['MANURE_FACTOR_AVERAGE'] +
        (df["avg_calves_4_6_mo"] + df["avg_calves_0_3_mo"]) * cf['CALF_FACTOR'] * cf['MANURE_FACTOR_AVERAGE']
    ) * cf['MANURE_N_CONTENT']

    if 'total_manure_gen_n_after_nh3_losses_lbs' in df.columns:
        reported_n = df['total_manure_gen_n_after_nh3_losses_lbs']
        df['nitrogen_discrepancy'] = reported_n - df['usda_nitrogen_estimate_lbs']
        df['usda_nitrogen_pct_deviation'] = (reported_n - df['usda_nitrogen_estimate_lbs']) / df['usda_nitrogen_estimate_lbs'].replace(0, np.nan) * 100
        df['ucce_nitrogen_pct_deviation'] = (reported_n - df['ucce_nitrogen_estimate_lbs']) / df['ucce_nitrogen_estimate_lbs'].replace(0, np.nan) * 100
    else:
        for col in ['nitrogen_discrepancy', 'usda_nitrogen_pct_deviation', 'ucce_nitrogen_pct_deviation']:
            df[col] = np.nan

    return df



def calculate_consultant_metrics(df):
    """Calculate average under/over-reporting metrics for each consultant."""
    # Group by consultant
    consultant_groups = df.groupby('consultant')
    
    metrics = []
    for consultant, group in consultant_groups:
        manure_avg = group['calculated_manure_factor'].mean() if 'calculated_manure_factor' in group.columns else None
        manure_std = group['calculated_manure_factor'].std() if 'calculated_manure_factor' in group.columns else None
        
        wastewater_avg = group['wastewater_to_reported'].mean() if 'wastewater_to_reported' in group.columns else None
        wastewater_std = group['wastewater_to_reported'].std() if 'wastewater_to_reported' in group.columns else None
        
        nitrogen_usda_avg = group['usda_nitrogen_pct_deviation'].mean() if 'usda_nitrogen_pct_deviation' in group.columns else None
        nitrogen_usda_std = group['usda_nitrogen_pct_deviation'].std() if 'usda_nitrogen_pct_deviation' in group.columns else None
        
        nitrogen_ucce_avg = group['ucce_nitrogen_pct_deviation'].mean() if 'ucce_nitrogen_pct_deviation' in group.columns else None
        nitrogen_ucce_std = group['ucce_nitrogen_pct_deviation'].std() if 'ucce_nitrogen_pct_deviation' in group.columns else None
        
        metrics.append({
            'consultant': consultant,
            'manure_factor_avg': manure_avg,
            'manure_factor_std': manure_std,
            'wastewater_ratio_avg': wastewater_avg,
            'wastewater_ratio_std': wastewater_std,
            'usda_nitrogen_pct_dev_avg': nitrogen_usda_avg,
            'usda_nitrogen_pct_dev_std': nitrogen_usda_std,
            'ucce_nitrogen_pct_dev_avg': nitrogen_ucce_avg,
            'ucce_nitrogen_pct_dev_std': nitrogen_ucce_std,
            'facility_count': len(group)
        })
    
    return pd.DataFrame(metrics)

def main(test_mode=False):
    """Main function to process all PDF files and extract data."""
    if read_reports:
        # Load parameters and create mappings
        parameters = pd.read_csv('ca_cafo_compliance/data/parameters.csv')
        snake_to_pretty = dict(zip(parameters['parameter_key'], parameters['parameter_name']))
        params = {
                'snake_to_pretty': snake_to_pretty,
                'data_types': dict(zip(parameters['parameter_key'], parameters['data_type'])),
            }
        
        dtype_dict = {col: str for col in ['region', 'template', 'parameter_key', 'page_search_text', 
                                          'search_direction', 'row_search_text', 'column_search_text',
                                          'ignore_before', 'value_pattern']}
        dtype_dict['item_order'] = 'Int64'
        parameter_locations = pd.read_csv('ca_cafo_compliance/data/parameter_locations.csv', dtype=dtype_dict)
        
        # Get all unique parameter keys from both files
        params_df = pd.read_csv('ca_cafo_compliance/data/parameters.csv')
        all_params = params_df['parameter_key'].unique().tolist()
        
        available_templates = parameter_locations['template'].unique()
        num_cores = 1 if test_mode else max(1, mp.cpu_count() - 3)
        
        for year in YEARS:
            base_data_path = f"ca_cafo_compliance/data/{year}"
            base_output_path = f"ca_cafo_compliance/outputs/{year}"
            for region in REGIONS:
                region_data_path = os.path.join(base_data_path, region)
                region_output_path = os.path.join(base_output_path, region)
                if not os.path.exists(region_data_path):
                    continue
                for county in [d for d in os.listdir(region_data_path) if os.path.isdir(os.path.join(region_data_path, d))]:
                    county_data_path = os.path.join(region_data_path, county)
                    county_output_path = os.path.join(region_output_path, county)
                    for template in [d for d in os.listdir(county_data_path) if os.path.isdir(os.path.join(county_data_path, d))]:
                        print(f'processing {template} in {county}')
                        if template not in available_templates:
                            continue
                        folder = os.path.join(county_data_path, template)
                        output_dir = os.path.join(county_output_path, template)
                        name = f"{county.capitalize()}_{year}_{template}"
                        template_params = parameter_locations[parameter_locations['template'] == template]
                        columns = template_params['parameter_key'].unique().tolist()
                        # Process files based on template type
                        if template == 'r8_csv' and region == 'R8':
                            # Process R8 CSV files
                            animals_path = os.path.join(base_data_path, 'R8', 'all_r8', 'r8_csv', 'R8_animals.csv')
                            manure_path = os.path.join(base_data_path, 'R8', 'all_r8', 'r8_csv', 'R8_manure.csv')
                            animals_df = pd.read_csv(animals_path)
                            manure_df = pd.read_csv(manure_path)
                            df = pd.merge(animals_df, manure_df, on='Facility Name', how='outer', suffixes=('', '_manure'))
                            results = []
                            for _, row in df.iterrows():
                                # Initialize all parameters as None
                                result = {param: None for param in all_params}
                                result['filename'] = 'R8_animals.csv'
                                for _, param_row in template_params.iterrows():
                                    param_key = param_row['parameter_key']
                                    col_name = param_row['column_search_text']
                                    if col_name in df.columns:
                                        if params['data_types'].get(param_key) == 'numeric':
                                            value = pd.to_numeric(row[col_name], errors='coerce')
                                            if pd.isna(value):
                                                value = 0
                                        else:
                                            value = row[col_name]
                                        result[param_key] = value
                                    elif param_row['row_search_text'] in df.columns:
                                        col_name = param_row['row_search_text']
                                        if params['data_types'].get(param_key) == 'numeric':
                                            value = pd.to_numeric(row[col_name], errors='coerce')
                                            if pd.isna(value):
                                                value = 0
                                        else:
                                            value = row[col_name]
                                        result[param_key] = value
                                results.append(result)
                            df = pd.DataFrame(results)
                        else:
                            # Process PDF files
                            ocr_dir = os.path.join(folder, 'ocr_output')
                            ai_ocr_dir = os.path.join(folder, 'ai_ocr_output')
                            if not os.path.exists(ocr_dir) and not os.path.exists(ai_ocr_dir):
                                continue
                            pdf_files = []
                            for text_file in glob.glob(os.path.join(ocr_dir, '*.txt')):
                                pdf_name = os.path.basename(text_file).replace('.txt', '.pdf')
                                pdf_path = os.path.join(folder, 'original', pdf_name)
                                if os.path.exists(pdf_path):
                                    pdf_files.append(pdf_path)
                            if os.path.exists(ai_ocr_dir):
                                for text_file in glob.glob(os.path.join(ai_ocr_dir, '*.txt')):
                                    pdf_name = os.path.basename(text_file).replace('.txt', '.pdf')
                                    pdf_path = os.path.join(folder, 'original', pdf_name)
                                    if os.path.exists(pdf_path) and pdf_path not in pdf_files:
                                        pdf_files.append(pdf_path)
                            if test_mode:
                                pdf_files = pdf_files[:2]
                            if not pdf_files:
                                continue
                            with mp.Pool(num_cores) as pool:
                                process_pdf_partial = partial(process_pdf, template_params=template_params, 
                                                           columns=all_params, data_types=params['data_types'])
                                results = pool.map(process_pdf_partial, pdf_files)
                            df = pd.DataFrame([r for r in results if r is not None])
                        # Convert numeric columns
                        for col in df.columns:
                            if params['data_types'].get(col) == 'numeric':
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Calculate metrics
                        df = calculate_metrics(df)
                        # Save results
                        os.makedirs(output_dir, exist_ok=True)
                        for f in os.listdir(output_dir):
                            if f.endswith('.csv'):
                                os.remove(os.path.join(output_dir, f))
                        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    if consolidate_data:
        # Load CADD data
        cadd_facilities = pd.read_csv('ca_cafo_compliance/data/CADD/CADD_Facility General Information_v1.0.0.csv')
        cadd_herd_size = pd.read_csv('ca_cafo_compliance/data/CADD/CADD_Facility Herd Size_v1.0.0.csv')
        
        # Load geocoding cache
        with open("ca_cafo_compliance/outputs/geocoding_cache.json", 'r') as f:
            cache = json.load(f)

        for year in YEARS:
            base_path = f"ca_cafo_compliance/outputs/{year}"
            if not os.path.exists(base_path):
                continue
                
            for region in REGIONS:
                region_path = os.path.join(base_path, region)
                if not os.path.exists(region_path):
                    continue
                    
                # Collect and process CSV files
                csv_files = glob.glob(os.path.join(region_path, "**/*.csv"), recursive=True)
                if not csv_files:
                    continue
                    
                # Combine all CSVs
                dfs = []
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    df['year'] = year
                    df['region'] = region
                    df['filename'] = os.path.basename(csv_file)
                    path_parts = csv_file.split(os.sep)
                    region_idx = path_parts.index(region)
                    if region_idx + 2 < len(path_parts):
                        df['template'] = path_parts[region_idx + 2]
                    dfs.append(df)
                    
                if not dfs:
                    continue
                    
                # Combine all data
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df = combined_df.dropna(how='all')
                combined_df['consultant'] = combined_df['template'].map(consultant_mapping).fillna('Unknown')

                # Load zipcode to county mapping
                zipcode_df = pd.read_csv('ca_cafo_compliance/data/zipcode_to_county.csv')
                zipcode_df = zipcode_df[['zip', 'county_name']].drop_duplicates()
                zipcode_df['zip'] = zipcode_df['zip'].astype(str)

                # Extract zipcode from dairy_address if it exists
                def extract_zipcode(address):
                    if pd.isna(address):
                        return None
                    # Look for 5-digit zipcode pattern
                    match = re.search(r'\b\d{5}\b', str(address))
                    return match.group(0) if match else None

                combined_df['zip'] = combined_df['dairy_address'].apply(extract_zipcode)
                
                # Merge with zipcode data to get county
                combined_df = pd.merge(
                    combined_df,
                    zipcode_df,
                    left_on='zip',
                    right_on='zip',
                    how='left'
                )

                # Initialize latitude and longitude columns
                combined_df['latitude'] = None
                combined_df['longitude'] = None

                # Geocode addresses first
                print("Geocoding addresses...")
                for idx, row in combined_df.iterrows():
                    if pd.isna(row['dairy_address']):
                        continue
                    lat, lng = geocode_address(row['dairy_address'], cache)
                    if lat is not None and lng is not None:
                        combined_df.at[idx, 'latitude'] = lat
                        combined_df.at[idx, 'longitude'] = lng

                # Fuzzy match with CADD data
                def find_fuzzy_match(row):
                    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                        return None
                    
                    distances = []
                    for _, cadd_row in cadd_facilities.iterrows():
                        if pd.isna(cadd_row['Latitude']) or pd.isna(cadd_row['Longitude']):
                            continue
                        
                        lat1, lon1 = float(row['latitude']), float(row['longitude'])
                        lat2, lon2 = float(cadd_row['Latitude']), float(cadd_row['Longitude'])
                        distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111000
                        
                        name1_words = set(str(row['dairy_name']).lower().split())
                        name2_words = set(str(cadd_row['FacilityName']).lower().split())
                        common_words = name1_words.intersection(name2_words)
                        
                        if distance <= 100 and len(common_words) > 0:
                            distances.append((distance, cadd_row))
                    
                    return min(distances, key=lambda x: x[0])[1] if distances else None

                # Apply fuzzy matching
                fuzzy_matches = []
                for _, row in combined_df.iterrows():
                    match = find_fuzzy_match(row)
                    row_dict = row.to_dict()
                    if match is not None:
                        # Only merge CADDID for matching
                        row_dict['CADDID'] = match['CADDID']
                    fuzzy_matches.append(row_dict)

                # Create final dataframe
                final_df = pd.DataFrame(fuzzy_matches)

                # Extract city, state, and county from address before conversion
                def extract_address_components(address):
                    if pd.isna(address):
                        return None, None, None
                    # Split address by commas
                    parts = str(address).split(',')
                    if len(parts) >= 3:
                        city = parts[-2].strip()
                        state = parts[-1].strip()
                        # County might be in the address or from zipcode mapping
                        county = None
                        for part in parts[:-2]:
                            if 'county' in part.lower():
                                county = part.strip()
                                break
                        return city, state, county
                    return None, None, None

                # Extract address components
                address_components = final_df['dairy_address'].apply(extract_address_components)
                final_df['city'] = address_components.apply(lambda x: x[0] if x else None)
                final_df['state'] = address_components.apply(lambda x: x[1] if x else None)
                # Use county from zipcode mapping if available, otherwise from address
                final_df['county'] = final_df['county_name'].fillna(
                    address_components.apply(lambda x: x[2] if x else None)
                )

                # Merge with CADD herd size data only for matched facilities
                if 'CADDID' in final_df.columns:
                    current_year_herd = cadd_herd_size[cadd_herd_size['Year'] == int(year)].copy()
                    if not current_year_herd.empty:
                        final_df = pd.merge(
                            final_df,
                            current_year_herd,
                            left_on='CADDID',
                            right_on='CADDID',
                            how='left'
                        )

                # Calculate consultant metrics for R5 and 2023
                if year == 2023 and region == "R5":
                    consultant_metrics = calculate_consultant_metrics(final_df)
                    # Convert consultant metrics to pretty names before saving
                    unmapped_cols = [col for col in consultant_metrics.columns if col not in params['snake_to_pretty']]
                    if unmapped_cols:
                        print(f"\nUnmapped columns in consultant metrics:")
                        for col in unmapped_cols:
                            print(f"  - {col}")
                    consultant_metrics = consultant_metrics.rename(columns=params['snake_to_pretty'])
                    metrics_file = f"ca_cafo_compliance/outputs/consolidated/{year}_{region}_consultant_metrics.csv"
                    consultant_metrics.to_csv(metrics_file, index=False)

                # Convert to pretty names only at the very end
                unmapped_cols = [col for col in final_df.columns if col not in params['snake_to_pretty']]
                if unmapped_cols:
                    print(f"\nUnmapped columns in final dataframe:")
                    for col in unmapped_cols:
                        print(f"  - {col}")
                final_df = final_df.rename(columns=params['snake_to_pretty'])

                # Save consolidated data
                output_file = f"ca_cafo_compliance/outputs/consolidated/{year}_{region}_master.csv"
                final_df.to_csv(output_file, index=False)
                print(f"Saved consolidated data to {output_file}")
                print(f"Total records: {len(final_df)}")

if __name__ == "__main__":
    main(test_mode=False)