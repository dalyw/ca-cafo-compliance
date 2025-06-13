import os
import json
from datetime import datetime
from geopy.geocoders import ArcGIS

from ca_cafo_compliance.helper_functions.read_report_helpers import *

geolocator = ArcGIS(user_agent="ca_cafo_compliance")

def save_geocoding_cache(cache):
    """Save geocoded addresses to cache file."""
    os.makedirs(os.path.dirname("ca_cafo_compliance/outputs/geocoding_cache.json"), exist_ok=True)
    with open("ca_cafo_compliance/outputs/geocoding_cache.json", 'w') as f:
        json.dump(cache, f, indent=2)

def normalize_address(address):
    if pd.isna(address) or not isinstance(address, str):
        return None
    address = address.replace(": ", "")
    address = address.lower()
    address = re.sub(r'[.,]', '', address)
    for old, new in street_replacements.items():
        address = address.replace(old, new)
    address = re.sub(r'\b(ca|california)\b', '', address)
    address = re.sub(r'\b(inc|llc)\b', '', address) 
    return ' '.join(address.split())

def find_cached_address(address, cache):
    if pd.isna(address) or not isinstance(address, str):
        return None
    if address in cache:
        return address
    normalized = normalize_address(address)
    if not normalized:
        return None
        
    # Search through cache keys
    for cached_addr in cache.keys():
        if normalize_address(cached_addr) == normalized:
            return cached_addr
    return None

def geocode_address(address, cache, county=None, try_again=False):
    if pd.isna(address) or not isinstance(address, str):
        return None, None, None
    clean_address = address.replace(": ", "")
    cached_addr = find_cached_address(clean_address, cache)
    if cached_addr:
        cached_result = cache[cached_addr]
        if try_again and (cached_result['lat'] is None or cached_result['lng'] is None):
            print(f"Retrying previously failed address: {clean_address}")
        else:
            return cached_result['lat'], cached_result['lng'], cached_result.get('county')

    # Format address with county if provided
    formatted_address = clean_address
    if county and not pd.isna(county) and isinstance(county, str) and 'all_' not in county:
        formatted_address = f"{clean_address}, {county} county, CA"
    elif county and not pd.isna(county) and isinstance(county, str) and county == "all_r2":
        formatted_address = f"{clean_address}, California"

    try:
        location = geolocator.geocode(formatted_address)
        if location and location.address and ('California' in location.address or 'CA' in location.address):
            county_val = None
            if isinstance(geolocator, ArcGIS):
                try:
                    reverse = geolocator.reverse(f"{location.latitude}, {location.longitude}")
                    if reverse and reverse.raw:
                        address_components = reverse.raw.get('address', {})
                        county_val = address_components.get('county')
                except Exception as e:
                    print(f"Error getting county from ArcGIS: {e}")

            cache[clean_address] = {
                'lat': location.latitude,
                'lng': location.longitude,
                'timestamp': datetime.now().isoformat(),
                'successful_format': formatted_address,
                'geocoder': geolocator.__class__.__name__,
                'address': location.address,
                'county': county_val
            }
            save_geocoding_cache(cache)
            print(f"Successfully geocoded address: {formatted_address}")
            return location.latitude, location.longitude, county_val

    except Exception as e:
        print(f"Error geocoding address '{formatted_address}': {e}")

    cache[clean_address] = {
        'lat': None,
        'lng': None,
        'error': "Geocoding failed",
        'timestamp': datetime.now().isoformat()
    }
    save_geocoding_cache(cache)
    return None, None, None

def parse_address(address):
    """Parse an address string into its components."""
    if pd.isna(address) or not isinstance(address, str):
        return None, None, None, None
        
    address = address.replace(": ", "")
    parts = address.split()
    if len(parts) < 3:
        return None, None, None, None
        
    street_number = parts[0]
    street_name = ' '.join(parts[1:-2])
    city = parts[-2]
    state_zip = parts[-1]
    zip = ''.join(filter(str.isdigit, state_zip[-5:])) if len(state_zip) >= 5 else None
    
    # Find county from address
    address_lower = address.lower()
    county = next((COUNTY_MAPPING[county] for county in COUNTY_MAPPING if county in address_lower), None)
    
    return street_number + ' ' + street_name, city, county, zip
