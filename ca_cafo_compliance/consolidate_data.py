#!/usr/bin/env python3

import pandas as pd
import os
import glob
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import json
import re
from conversion_factors import *
from datetime import datetime

GEOCODING_CACHE_FILE = "outputs/geocoding_cache.json"

def save_geocoding_cache(cache):
    """Save geocoded addresses to cache file."""
    os.makedirs(os.path.dirname(GEOCODING_CACHE_FILE), exist_ok=True)
    with open(GEOCODING_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def normalize_address(address):
    """Normalize address string for searching."""
    if pd.isna(address) or not isinstance(address, str):
        return None
        
    address = address.lower()
    address = re.sub(r'[.,]', '', address)
    
    replacements = {
        'avenue': 'ave',
        'street': 'st',
        'road': 'rd',
        'boulevard': 'blvd',
        'highway': 'hwy'
    }
    for old, new in replacements.items():
        address = address.replace(old, new)
    
    address = re.sub(r'\b(ca|california)\b', '', address)
    address = re.sub(r'\b(inc|llc)\b', '', address) 
    return ' '.join(address.split())

def find_cached_address(address, cache):
    """Find a cached address by searching normalized versions."""
    if pd.isna(address) or not isinstance(address, str):
        return None
        
    # First try exact match
    if address in cache:
        return address
        
    # Then try normalized match
    normalized = normalize_address(address)
    if not normalized:
        return None
        
    # Search through cache keys
    for cached_addr in cache.keys():
        if normalize_address(cached_addr) == normalized:
            return cached_addr
            
    return None

def geocode_address(address, cache, try_again=False):
    """Convert address to latitude and longitude using Geopy with caching."""
    if pd.isna(address) or not isinstance(address, str):
        return None, None
    
    # Check cache first
    cached_addr = find_cached_address(address, cache)
    if cached_addr:
        cached_result = cache[cached_addr]
        if try_again and (cached_result['lat'] is None or cached_result['lng'] is None):
            print(f"Retrying previously failed address: {address}")
        else:
            print(f"Found address in cache: {address}")
            return cached_result['lat'], cached_result['lng']
    
    # Try different address formats
    address_formats = [
        address,
        f"{address}, California",
        *[address.replace(f" {abbr} ", f" {full} ") for abbr, full in {
            "AVE": "Avenue",
            "ST": "Street",
            "RD": "Road",
            "BLVD": "Boulevard",
            "HWY": "Highway"
        }.items()]
    ]
    
    geolocator = Nominatim(user_agent="ca_cafo_compliance")
    for addr_format in address_formats:
        try:
            time.sleep(1)  # Rate limiting
            location = geolocator.geocode(addr_format)
            
            if location:
                cache[address] = {
                    'lat': location.latitude,
                    'lng': location.longitude,
                    'timestamp': datetime.now().isoformat(),
                    'successful_format': addr_format
                }
                save_geocoding_cache(cache)
                print(f"Successfully geocoded address using format: {addr_format}")
                return location.latitude, location.longitude
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Geocoding error for address format '{addr_format}': {e}")
            continue
    
    # Cache failure
    cache[address] = {
        'lat': None,
        'lng': None,
        'error': "All address formats failed",
        'timestamp': datetime.now().isoformat()
    }
    save_geocoding_cache(cache)
    return None, None

with open(GEOCODING_CACHE_FILE, 'r') as f:
    geocoding_cache = json.load(f)

for year in YEARS:
    base_path = f"outputs/{year}"
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
        
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Add metadata columns
                df['Year'] = year
                df['Region'] = region
                df['filename'] = os.path.basename(csv_file)
                
                # Extract county and template from path
                path_parts = csv_file.split(os.sep)
                region_idx = path_parts.index(region)
                if region_idx + 1 < len(path_parts):
                    df['County'] = path_parts[region_idx + 1]
                if region_idx + 2 < len(path_parts):
                    df['Template'] = path_parts[region_idx + 2]
                
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        if not dfs:
            continue
            
        # Combine and clean data
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.dropna(how='all')
        
        # Filter out empty rows
        metadata_cols = ['Year', 'Region', 'County', 'Template', 'filename']
        data_cols = [col for col in combined_df.columns if col not in metadata_cols]
        combined_df = combined_df[~combined_df.apply(
            lambda row: all(pd.isna(val) or val == 0 for val in row[data_cols]), 
            axis=1
        )]
        
        # Process addresses
        print("Geocoding addresses...")
        address_col = next((col for col in ['Dairy Address', 'Facility Address'] 
                            if col in combined_df.columns), None)
        
        if address_col:
            combined_df['Latitude'] = None
            combined_df['Longitude'] = None
            
            unique_addresses = combined_df[address_col].dropna().unique()
            new_geocodes = 0
            
            for address in unique_addresses:
                lat, lng = geocode_address(address, geocoding_cache, try_again=False)
                if lat is not None:
                    new_geocodes += 1
                
                mask = combined_df[address_col] == address
                combined_df.loc[mask, 'Latitude'] = lat
                combined_df.loc[mask, 'Longitude'] = lng
            
            print(f"Geocoding complete: {new_geocodes} addresses geocoded")
        
        output_file = f"outputs/consolidated/{year}_{region}_master.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"Saved consolidated data to {output_file}")
        print(f"Total records: {len(combined_df)}")