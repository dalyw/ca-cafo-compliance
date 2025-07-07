import os
import json
import pandas as pd
from geopy.geocoders import ArcGIS
from datetime import datetime
import re


def save_geocoding_cache(cache):
    """Save geocoded addresses to cache file."""
    os.makedirs(
        os.path.dirname("ca_cafo_compliance/outputs/geocoding_cache.json"),
        exist_ok=True,
    )
    with open("ca_cafo_compliance/outputs/geocoding_cache.json", "w") as f:
        json.dump(cache, f, indent=2)


def normalize_address(address):
    if pd.isna(address) or not isinstance(address, str):
        return None
    address = address.replace(": ", "")
    address = address.lower()
    address = re.sub(r"[.,]", "", address)
    street_replacements = {
        "avenue": "ave",
        "street": "st",
        "road": "rd",
        "boulevard": "blvd",
        "highway": "hwy",
    }
    for old, new in street_replacements.items():
        address = address.replace(old, new)
    address = re.sub(r"\b(ca|california)\b", "", address)
    address = re.sub(r"\b(inc|llc)\b", "", address)
    return " ".join(address.split())


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


def get_region_counties(region):
    """Get list of counties for a given region from county_region.csv."""
    county_region_df = pd.read_csv("ca_cafo_compliance/data/county_region.csv")
    return county_region_df[county_region_df["region"] == region][
        "county_name"
    ].tolist()


def geocode_address(address, cache, county=None, try_again=False):
    if pd.isna(address) or not isinstance(address, str):
        return None, None
    clean_address = address.replace(": ", "")
    cached_addr = find_cached_address(clean_address, cache)
    if cached_addr:
        cached_result = cache[cached_addr]
        if try_again and (cached_result["lat"] is None or cached_result["lng"] is None):
            print(f"Retrying previously failed address: {clean_address}")
        else:
            return cached_result["lat"], cached_result["lng"]

    # Format address with county if provided
    formatted_address = clean_address
    location = None
    geolocator = ArcGIS(user_agent="ca_cafo_compliance")

    if county and not pd.isna(county) and isinstance(county, str):
        if "all_" in county:
            # Extract region from county (e.g., 'all_r2' -> 'R2')
            region = county.split("_")[1].upper()
            # Get counties for this region
            region_counties = get_region_counties(region)

            # Try each county in the region
            for region_county in region_counties:
                formatted_address = f"{clean_address}, {region_county} county, CA"
                location = geolocator.geocode(formatted_address)

                if (
                    location
                    and location.address
                    and ("California" in location.address or "CA" in location.address)
                ):
                    break

            # If no county worked, try without county
            if (
                not location
                or not location.address
                or (
                    "California" not in location.address
                    and "CA" not in location.address
                )
            ):
                formatted_address = f"{clean_address}, California"
                location = geolocator.geocode(formatted_address)
        else:
            formatted_address = f"{clean_address}, {county} county, CA"
            location = geolocator.geocode(formatted_address)
    else:
        formatted_address = f"{clean_address}, California"
        location = geolocator.geocode(formatted_address)

    if (
        location
        and location.address
        and ("California" in location.address or "CA" in location.address)
    ):
        cache[clean_address] = {
            "lat": location.latitude,
            "lng": location.longitude,
            "timestamp": datetime.now().isoformat(),
            "successful_format": formatted_address,
            "geocoder": geolocator.__class__.__name__,
            "address": location.address,
        }
        save_geocoding_cache(cache)
        print(f"Successfully geocoded address: {formatted_address}")
        return location.latitude, location.longitude

    cache[clean_address] = {
        "lat": None,
        "lng": None,
        "error": "Geocoding failed",
        "timestamp": datetime.now().isoformat(),
    }
    save_geocoding_cache(cache)
    return None, None


def extract_address_components(address):
    """Extract city, state, and county from address."""
    if pd.isna(address):
        return None, None, None
    parts = str(address).split(",")
    # print(parts)
    if len(parts) >= 3:
        zip_code = parts[-1].strip()
        state = parts[-2].strip()
        city = parts[-3].strip()
        return city, state, zip_code
    return None, None, None
