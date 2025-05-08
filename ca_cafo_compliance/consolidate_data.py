#!/usr/bin/env python3

import pandas as pd
import os
import glob
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import json
import re

GEOCODING_CACHE_FILE = "outputs/geocoding_cache.json"

def load_geocoding_cache():
    """Load previously geocoded addresses from cache file."""
    if os.path.exists(GEOCODING_CACHE_FILE):
        try:
            with open(GEOCODING_CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not read geocoding cache file. Starting with empty cache.")
    return {}

def save_geocoding_cache(cache):
    """Save geocoded addresses to cache file."""
    os.makedirs(os.path.dirname(GEOCODING_CACHE_FILE), exist_ok=True)
    with open(GEOCODING_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def normalize_address(address):
    """Normalize address string for consistent caching."""
    if pd.isna(address) or not isinstance(address, str):
        return None
    # Convert to lowercase and remove extra whitespace
    return ' '.join(address.lower().split())

def geocode_address(address, cache):
    """Convert address to latitude and longitude using Geopy with caching."""
    if pd.isna(address) or not isinstance(address, str):
        return None, None
    
    # Normalize address for cache lookup
    normalized_address = normalize_address(address)
    if normalized_address in cache:
        return cache[normalized_address]['lat'], cache[normalized_address]['lng']
    
    # If not in cache, geocode the address
    geolocator = Nominatim(user_agent="ca_cafo_compliance")
    try:
        # Add delay to avoid hitting rate limits
        time.sleep(1)
        location = geolocator.geocode(address)
        if location:
            # Save to cache using normalized address as key
            cache[normalized_address] = {
                'lat': location.latitude,
                'lng': location.longitude,
                'original_address': address  # Store original for reference
            }
            save_geocoding_cache(cache)
            return location.latitude, location.longitude
        return None, None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error for address '{address}': {e}")
        return None, None

def consolidate_data():
    """Consolidate data from all counties and templates into region-level master CSVs."""
    years = [2023, 2024]
    regions = ['R2', 'R3', 'R5', 'R7', 'R8']
    
    # Load geocoding cache at start
    geocoding_cache = load_geocoding_cache()
    print(f"Loaded {len(geocoding_cache)} cached addresses")
    
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
                    df = pd.read_csv(csv_file)
                    # Add metadata columns if they don't exist
                    if 'Year' not in df.columns:
                        df['Year'] = year
                    if 'Region' not in df.columns:
                        df['Region'] = region
                    if 'County' not in df.columns:
                        # Extract county from path
                        path_parts = csv_file.split(os.sep)
                        county_idx = path_parts.index(region) + 1
                        if county_idx < len(path_parts):
                            df['County'] = path_parts[county_idx]
                    if 'Template' not in df.columns:
                        # Extract template from path
                        path_parts = csv_file.split(os.sep)
                        template_idx = path_parts.index(region) + 2
                        if template_idx < len(path_parts):
                            df['Template'] = path_parts[template_idx]
                    if 'filename' not in df.columns:
                        df['filename'] = os.path.basename(csv_file)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df = combined_df.dropna(how='all')
                
                # Drop rows where all non-metadata columns are zeros or NaN
                metadata_cols = ['Year', 'Region', 'County', 'Template', 'filename']
                data_cols = [col for col in combined_df.columns if col not in metadata_cols]
                
                # Function to check if a row contains only zeros or NaN in data columns
                def is_empty_row(row):
                    for col in data_cols:
                        val = row[col]
                        if pd.notna(val) and val != 0:
                            return False
                    return True
                
                # Filter out empty rows
                combined_df = combined_df[~combined_df.apply(is_empty_row, axis=1)]
                
                # Geocode addresses
                print("Geocoding addresses...")
                address_col = None
                for possible_col in ['Dairy Address', 'Facility Address']:
                    if possible_col in combined_df.columns:
                        address_col = possible_col
                        break
                
                if address_col:
                    # Initialize latitude and longitude columns
                    combined_df['Latitude'] = None
                    combined_df['Longitude'] = None
                    
                    # Process each unique address
                    unique_addresses = combined_df[address_col].dropna().unique()
                    new_geocodes = 0
                    cached_geocodes = 0
                    
                    for address in unique_addresses:
                        normalized_address = normalize_address(address)
                        if normalized_address in geocoding_cache:
                            cached_geocodes += 1
                            lat = geocoding_cache[normalized_address]['lat']
                            lng = geocoding_cache[normalized_address]['lng']
                        else:
                            print(f"Geocoding new address: {address}")
                            lat, lng = geocode_address(address, geocoding_cache)
                            if lat is not None:
                                new_geocodes += 1
                        
                        # Apply coordinates to all matching rows using exact match
                        mask = combined_df[address_col] == address
                        combined_df.loc[mask, 'Latitude'] = lat
                        combined_df.loc[mask, 'Longitude'] = lng
                    
                    print(f"Geocoding complete: {cached_geocodes} from cache, {new_geocodes} new addresses geocoded")
                
                os.makedirs("outputs/consolidated", exist_ok=True)
                output_file = f"outputs/consolidated/{year}_{region}_master.csv"
                combined_df.to_csv(output_file, index=False)
                print(f"Saved consolidated data to {output_file}")
                print(f"Total records: {len(combined_df)}")

if __name__ == "__main__":
    consolidate_data()