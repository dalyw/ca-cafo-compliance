#!/usr/bin/env python3

import pandas as pd
import os
import glob
import time
import json
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from geopy.geocoders import Nominatim, ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from conversion_factors import *

GEOCODING_CACHE_FILE = "outputs/geocoding_cache.json"
R2_COUNTIES = ["Alameda", "Contra Costa", "Marin", "Napa",  "San Francisco", 
               "San Mateo", "Santa Clara", "Solano", "Sonoma"]

def save_geocoding_cache(cache):
    """Save geocoded addresses to cache file."""
    os.makedirs(os.path.dirname(GEOCODING_CACHE_FILE), exist_ok=True)
    with open(GEOCODING_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def normalize_address(address):
    """Normalize address string for searching."""
    if pd.isna(address) or not isinstance(address, str):
        return None
        
    address = address.replace(": ", "")
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
        
    if address in cache:
        return address
        
    # Try normalized match
    normalized = normalize_address(address)
    if not normalized:
        return None
        
    # Search through cache keys
    for cached_addr in cache.keys():
        if normalize_address(cached_addr) == normalized:
            return cached_addr
            
    return None

def geocode_address(address, cache, county=None, try_again=False):
    """Convert address to latitude and longitude using Geopy with caching."""
    if pd.isna(address) or not isinstance(address, str):
        return None, None, None
    
    clean_address = address.replace(": ", "")
    
    # Check cache first
    cached_addr = find_cached_address(clean_address, cache)
    if cached_addr:
        cached_result = cache[cached_addr]
        if try_again and (cached_result['lat'] is None or cached_result['lng'] is None):
            print(f"Retrying previously failed address: {clean_address}")
        else:
            print(f"Found address in cache: {clean_address}")
            return cached_result['lat'], cached_result['lng'], cached_result.get('county')
    
    address_formats = []
    parts = clean_address.split()
    if len(parts) >= 3:
        # Try to identify street number, name, and city
        street_number = parts[0]
        street_name = ' '.join(parts[1:-2])  # Everything between number and city
        city = parts[-2]
        state_zip = parts[-1]
        
        # Create a more structured address format
        formatted_address = f"{street_number} {street_name}, {city}, CA {state_zip}"
        
        # Try different address formats with priority
        address_formats = [
            formatted_address,  # Most structured format first
            clean_address,      # Original format
            f"{clean_address}, California"  # Simple addition of state
        ]
        
        # Add try counties for R2
        if county == "all_r2":
            # Add formats with each R2 county
            for r2_county in R2_COUNTIES:
                address_formats.append(f"{formatted_address}, {r2_county} County, CA")
        elif county and county not in ["all_r2", "all_r7"]:
            address_formats.append(f"{formatted_address}, {county} County, CA")
    else:
        # If we can't parse the address well, just try a few basic formats
        address_formats = [
            clean_address,
            f"{clean_address}, California"
        ]
        
        # Add county-specific formats if needed
        if county == "all_r2":
            # Add formats with each R2 county
            for r2_county in R2_COUNTIES:
                address_formats.append(f"{clean_address}, {r2_county} County, CA")
        elif county and county not in ["all_r2", "all_r7"]:
            address_formats.append(f"{clean_address}, {county} County, CA")
    
    # Try multiple geocoding services
    geocoders = [
        ArcGIS(user_agent="ca_cafo_compliance"),  # Try ArcGIS first as it's more reliable
        Nominatim(user_agent="ca_cafo_compliance", timeout=30)  # Increased timeout for Nominatim
    ]
    
    successful_locations = []
    max_retries = 2
    retry_delay = 3  # seconds
    
    for geolocator in geocoders:
        for addr_format in address_formats:
            for attempt in range(max_retries):
                try:
                    # Add exponential backoff for retries
                    if attempt > 0:
                        time.sleep(retry_delay * (2 ** attempt))
                    else:
                        time.sleep(1)  # Basic rate limiting
                    
                    location = geolocator.geocode(addr_format)
                    
                    if location:
                        # Verify in California
                        if location.address and ('California' in location.address or 'CA' in location.address):
                            # Create a location key for deduplication
                            loc_key = f"{location.latitude:.6f},{location.longitude:.6f}"
                            
                            # Check if we already have this location
                            if not any(f"{loc['lat']:.6f},{loc['lng']:.6f}" == loc_key for loc in successful_locations):
                                # Try to extract county from ArcGIS response
                                county = None
                                if isinstance(geolocator, ArcGIS):
                                    try:
                                        # Use ArcGIS reverse geocoding to get county
                                        reverse = geolocator.reverse(f"{location.latitude}, {location.longitude}")
                                        if reverse and reverse.raw:
                                            address_components = reverse.raw.get('address', {})
                                            county = address_components.get('County')
                                    except Exception as e:
                                        print(f"Error getting county from ArcGIS: {e}")
                                
                                successful_locations.append({
                                    'lat': location.latitude,
                                    'lng': location.longitude,
                                    'format': addr_format,
                                    'geocoder': geolocator.__class__.__name__,
                                    'address': location.address,
                                    'county': county
                                })
                            
                            # If we have at least one successful location, we can stop trying more formats
                            if successful_locations:
                                break
                    
                except (GeocoderTimedOut, GeocoderServiceError) as e:
                    print(f"Geocoding error for address format '{addr_format}' (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:  # Last attempt
                        print(f"Failed to geocode after {max_retries} attempts")
                    continue
                except Exception as e:
                    print(f"Unexpected error for address format '{addr_format}': {e}")
                    break
            
            # If we have a successful location, no need to try more formats
            if successful_locations:
                break
        
        # If we have a successful location, no need to try more geocoders
        if successful_locations:
            break
    
    if successful_locations:
        # If multiple locations found, print them
        if len(successful_locations) > 1:
            print(f"Multiple unique locations found for address: {clean_address}")
            for loc in successful_locations:
                print(f"  - {loc['address']} ({loc['lat']}, {loc['lng']})")
        
        # Use the first successful location
        best_location = successful_locations[0]
        cache[clean_address] = {
            'lat': best_location['lat'],
            'lng': best_location['lng'],
            'timestamp': datetime.now().isoformat(),
            'successful_format': best_location['format'],
            'geocoder': best_location['geocoder'],
            'address': best_location['address'],
            'county': best_location['county']
        }
        save_geocoding_cache(cache)
        print(f"Successfully geocoded address using format: {best_location['format']} from {best_location['geocoder']}")
        return best_location['lat'], best_location['lng'], best_location['county']
    
    # Try Google Maps fallback
    try:
        search_url = f"https://www.google.com/maps/search/{clean_address.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(search_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                if tag.get('property') == 'og:latitude':
                    lat = float(tag.get('content'))
                    lng = float(soup.find('meta', property='og:longitude').get('content'))
                    cache[clean_address] = {
                        'lat': lat,
                        'lng': lng,
                        'timestamp': datetime.now().isoformat(),
                        'successful_format': 'Google Maps fallback',
                        'geocoder': 'Google Maps'
                    }
                    save_geocoding_cache(cache)
                    print(f"Successfully geocoded address using Google Maps fallback")
                    return lat, lng, None
    except Exception as e:
        print(f"Google Maps fallback failed: {e}")
    
    # Cache failure
    cache[clean_address] = {
        'lat': None,
        'lng': None,
        'error': "All address formats failed",
        'timestamp': datetime.now().isoformat()
    }
    save_geocoding_cache(cache)
    return None, None, None

def parse_address(address):
    """Parse address string into components."""
    if pd.isna(address) or not isinstance(address, str):
        return None, None, None, None
        
    # Remove any prefix like "Dairy Address: "
    address = address.replace(": ", "")
    
    # Split address into parts
    parts = address.split()
    if len(parts) < 3:
        return None, None, None, None
        
    # Extract components
    street_number = parts[0]
    street_name = ' '.join(parts[1:-2])  # Everything between number and city
    city = parts[-2]
    state_zip = parts[-1]
    
    # Extract zip code (assuming it's the last part after state)
    zip_code = state_zip[-5:] if len(state_zip) >= 5 else None
    
    # Try to extract county from address
    county = None
    address_lower = address.lower()
    if 'fresno' in address_lower or 'madera' in address_lower:
        county = 'Fresno/Madera'
    elif 'kern' in address_lower:
        county = 'Kern'
    elif 'kings' in address_lower:
        county = 'Kings'
    elif 'tulare' in address_lower:
        county = 'Tulare'
    elif 'sonoma' in address_lower:
        county = 'Sonoma'
    elif 'marin' in address_lower:
        county = 'Marin'
    elif 'napa' in address_lower:
        county = 'Napa'
    elif 'solano' in address_lower:
        county = 'Solano'
    elif 'contra costa' in address_lower:
        county = 'Contra Costa'
    elif 'alameda' in address_lower:
        county = 'Alameda'
    elif 'san francisco' in address_lower:
        county = 'San Francisco'
    elif 'san mateo' in address_lower:
        county = 'San Mateo'
    elif 'santa clara' in address_lower:
        county = 'Santa Clara'
    
    return street_number + ' ' + street_name, city, county, zip_code

def calculate_metrics(df):
    """Calculate metrics for each facility."""
    # Calculate manure factor
    df['Calculated Manure Factor'] = df.apply(
        lambda row: row['Total Manure Excreted (tons)'] / row['Total Herd Size'] 
        if row['Total Herd Size'] > 0 else None, 
        axis=1
    )
    
    # Calculate wastewater ratio
    df['Ratio of Wastewater to Milk (L/L)'] = df.apply(
        lambda row: row['Total Process Wastewater Generated (gals)'] / (row['Average Milk Production (lb per cow per day)'] * row['Average Milk Cows'] * 365 * 0.45359237)
        if row['Average Milk Production (lb per cow per day)'] > 0 and row['Average Milk Cows'] > 0 else None,
        axis=1
    )
    
    # Calculate nitrogen deviations
    df['USDA Nitrogen % Deviation'] = df.apply(
        lambda row: ((row['Total Dry Manure Generated N (lbs)'] - row['USDA Nitrogen Estimate (lbs)']) / row['USDA Nitrogen Estimate (lbs)'] * 100)
        if row['USDA Nitrogen Estimate (lbs)'] > 0 else None,
        axis=1
    )
    
    df['UCCE Nitrogen % Deviation'] = df.apply(
        lambda row: ((row['Total Dry Manure Generated N (lbs)'] - row['UCCE Nitrogen Estimate (lbs)']) / row['UCCE Nitrogen Estimate (lbs)'] * 100)
        if row['UCCE Nitrogen Estimate (lbs)'] > 0 else None,
        axis=1
    )
    
    return df

def calculate_consultant_metrics(df):
    """Calculate average under/over-reporting metrics for each consultant."""
    # First calculate the metrics for each facility
    df = calculate_metrics(df)
    
    # Group by consultant
    consultant_groups = df.groupby('Consultant')
    
    metrics = []
    for consultant, group in consultant_groups:
        # Calculate averages for each metric
        manure_avg = group['Calculated Manure Factor'].mean()
        manure_std = group['Calculated Manure Factor'].std()
        
        wastewater_avg = group['Ratio of Wastewater to Milk (L/L)'].mean()
        wastewater_std = group['Ratio of Wastewater to Milk (L/L)'].std()
        
        nitrogen_usda_avg = group['USDA Nitrogen % Deviation'].mean()
        nitrogen_usda_std = group['USDA Nitrogen % Deviation'].std()
        
        nitrogen_ucce_avg = group['UCCE Nitrogen % Deviation'].mean()
        nitrogen_ucce_std = group['UCCE Nitrogen % Deviation'].std()
        
        metrics.append({
            'Consultant': consultant,
            'Manure Factor Avg': manure_avg,
            'Manure Factor Std': manure_std,
            'Wastewater Ratio Avg': wastewater_avg,
            'Wastewater Ratio Std': wastewater_std,
            'USDA Nitrogen % Dev Avg': nitrogen_usda_avg,
            'USDA Nitrogen % Dev Std': nitrogen_usda_std,
            'UCCE Nitrogen % Dev Avg': nitrogen_ucce_avg,
            'UCCE Nitrogen % Dev Std': nitrogen_ucce_std,
            'Facility Count': len(group)
        })
    
    return pd.DataFrame(metrics)

try:
    with open(GEOCODING_CACHE_FILE, 'r') as f:
        geocoding_cache = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    geocoding_cache = {}

# Add R8 data to geocoding cache if not already present
r8_data_path = "data/2023/R8/all_r8/r8_csv/R8_animals.csv"
if os.path.exists(r8_data_path):
    print("Adding R8 facility data to geocoding cache...")
    r8_df = pd.read_csv(r8_data_path)
    for _, row in r8_df.iterrows():
        if pd.notna(row['Facility Address']) and pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            address = row['Facility Address']
            if address not in geocoding_cache:
                geocoding_cache[address] = {
                    'lat': float(row['Latitude']),
                    'lng': float(row['Longitude']),
                    'timestamp': datetime.now().isoformat(),
                    'successful_format': 'R8 direct data',
                    'geocoder': 'R8 CSV',
                    'address': address,
                    'county': row.get('County')  # If county is available in R8 data
                }
    save_geocoding_cache(geocoding_cache)
    print(f"Added {len(r8_df)} R8 facilities to geocoding cache")

# Process each year and region
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
            df = pd.read_csv(csv_file)
            
            # Add metadata columns
            df['Year'] = year
            df['Region'] = region
            df['filename'] = os.path.basename(csv_file)
            
            # Extract template from path
            path_parts = csv_file.split(os.sep)
            region_idx = path_parts.index(region)
            if region_idx + 2 < len(path_parts):
                df['Template'] = path_parts[region_idx + 2]
            
            dfs.append(df)
        if not dfs:
            continue
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.dropna(how='all')
        
        # Filter out empty rows
        metadata_cols = ['Year', 'Region', 'Template', 'filename']
        data_cols = [col for col in combined_df.columns if col not in metadata_cols]
        combined_df = combined_df[~combined_df.apply(
            lambda row: all(pd.isna(val) or val == 0 for val in row[data_cols]), 
            axis=1
        )]
        
        # Add consultant name based on template
        combined_df['Consultant'] = combined_df['Template'].map(consultant_mapping).fillna('Unknown')
        
        # Calculate total herd size robustly
        herd_cols = [
            'Average Milk Cows', 'Average Dry Cows', 'Average Bred Heifers',
            'Average Heifers', 'Average Calves (4-6 mo.)', 'Average Calves (0-3 mo.)',
        ]
        def calc_herd_size(row):
            vals = [row.get(col, 0) for col in herd_cols if col in row]
            vals = [v for v in vals if pd.notna(v)]
            return sum(vals) if vals else 0
        combined_df['Total Herd Size'] = combined_df.apply(calc_herd_size, axis=1)
        
        print("Geocoding addresses...")
        address_col = next((col for col in ['Dairy Address', 'Facility Address'] 
                            if col in combined_df.columns), None)
        
        if address_col:
            combined_df['Latitude'] = None
            combined_df['Longitude'] = None
            combined_df['Street Address'] = None
            combined_df['City'] = None
            combined_df['County'] = None
            combined_df['Zip'] = None
            
            unique_addresses = combined_df[address_col].dropna().unique()
            new_geocodes = 0
            
            for address in unique_addresses:
                # Get the county for this address
                county = combined_df[combined_df[address_col] == address]['County'].iloc[0] if not combined_df[combined_df[address_col] == address].empty else None
                
                lat, lng, geocoded_county = geocode_address(address, geocoding_cache, county=county, try_again=True)
                if lat is not None:
                    new_geocodes += 1
                
                # Parse address components
                street, city, parsed_county, zip_code = parse_address(address)
                
                # Use geocoded county if available, otherwise use parsed county
                final_county = geocoded_county or parsed_county
                
                mask = combined_df[address_col] == address
                combined_df.loc[mask, 'Latitude'] = lat
                combined_df.loc[mask, 'Longitude'] = lng
                combined_df.loc[mask, 'Street Address'] = street
                combined_df.loc[mask, 'City'] = city
                combined_df.loc[mask, 'County'] = final_county
                combined_df.loc[mask, 'Zip'] = zip_code
            
            print(f"Geocoding complete: {new_geocodes} addresses geocoded")
        
        # Add consultant metrics only for R5 and 2023
        if year == "2023" and region == "R5":
            consultant_metrics = calculate_consultant_metrics(combined_df)
            metrics_file = f"outputs/consolidated/{year}_{region}_consultant_metrics.csv"
            consultant_metrics.to_csv(metrics_file, index=False)
            print(f"Saved consultant metrics to {metrics_file}")
        
        output_file = f"outputs/consolidated/{year}_{region}_master.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"Saved consolidated data to {output_file}")
        print(f"Total records: {len(combined_df)}")