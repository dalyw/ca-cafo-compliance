#!/usr/bin/env python3

import pandas as pd
import os
import glob
import numpy as np

def consolidate_data():
    """Consolidate data from all counties and templates into region-level master CSVs."""
    years = [2023, 2024]
    regions = ['R2', 'R3', 'R5', 'R7', 'R8']
    
    for year in years:
        base_path = f"outputs/{year}"
        if not os.path.exists(base_path):
            print(f"Year folder not found: {base_path}")
            continue
            
        for region in regions:
            region_path = os.path.join(base_path, region)
            if not os.path.exists(region_path):
                print(f"Region folder not found: {region_path}")
                continue
                
            print(f"\nProcessing {year} {region}")
            
            # Get all CSV files in this region (recursively through counties and templates)
            csv_files = glob.glob(os.path.join(region_path, "**/*.csv"), recursive=True)
            
            if not csv_files:
                print(f"No CSV files found in {region_path}")
                continue
            
            # Read and combine all CSVs
            dfs = []
            for csv_file in csv_files:
                try:
                    # Extract metadata from file path
                    rel_path = os.path.relpath(csv_file, region_path)
                    parts = rel_path.split(os.sep)
                    county = parts[0]
                    template = os.path.splitext(parts[1])[0].split('_')[-1]
                    
                    # Read the CSV and add metadata
                    df = pd.read_csv(csv_file)
                    df = df.dropna(how='all')
                    df['Year'] = year
                    df['Region'] = region
                    df['County'] = county
                    df['Template'] = template
                    
                    dfs.append(df)
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df = combined_df.dropna(how='all')
                
                os.makedirs("outputs/consolidated", exist_ok=True)
                output_file = f"outputs/consolidated/{year}_{region}_master.csv"
                combined_df.to_csv(output_file, index=False)
                print(f"Saved consolidated data to {output_file}")
                print(f"Total records: {len(combined_df)}")

if __name__ == "__main__":
    consolidate_data() 