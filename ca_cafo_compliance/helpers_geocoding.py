import os
import json
import re
import pandas as pd
import requests
from datetime import datetime
from geopy.geocoders import ArcGIS


def save_geocoding_cache(cache):
    """Save geocoded addresses to cache file."""
    os.makedirs(
        os.path.dirname("ca_cafo_compliance/outputs/geocoding_cache.json"),
        exist_ok=True,
    )
    with open("ca_cafo_compliance/outputs/geocoding_cache.json", "w") as f:
        json.dump(cache, f, indent=2)


def load_geocoding_cache():
    """Load geocoded addresses from cache file."""
    # Load geocoding cache
    with open("ca_cafo_compliance/outputs/geocoding_cache.json", "r") as f:
        cache = json.load(f)
    return cache


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


# APN detection and parsing
def looks_like_parcel_number(text):
    """Return True if text looks like an assessor parcel number rather than a street address."""
    if not text or not isinstance(text, str):
        return False
    s = text.strip()
    if not s or len(s) < 3:
        return False
    # contains at least one hyphen or dot and only digits/hyphens/dots/spaces
    if "-" not in s and "." not in s:  # assume APNS given with some separator
        return False
    if not re.match(r"^[\d\s.\-]+$", s):
        return False
    # At least two digit groups separated by hyphen or dot
    return bool(re.search(r"\d+\s*[.\-]\s*\d+", s))


# digit groups with hyphens or dots (e.g. 006-0153-011-0000, 015.080.075, X142-0100-X031-Xxxx)
_PARCEL_RE = re.compile(
    r"(?:\(?\d*\)?\s*[Xx]?\s*)?([\dXx]{2,}\s*[.\-]\s*[\dXx]{2,}(?:\s*[.\-]\s*[\dXx]+)*)",
    re.IGNORECASE,
)


def parse_destination_address_and_parcel(value):
    """
    Split destination field into address and parcel number when both are present.
    Returns (address_part, parcel_part); either can be None. Parcel is normalized to digits-hyphens.
    """
    if not value or not isinstance(value, str):
        return None, None
    s = value.strip()
    if not s:
        return None, None
    # If whole string is parcel-only, return (None, normalized_parcel)
    if looks_like_parcel_number(s):
        parcel = re.sub(r"\s+", "", s)
        return None, parcel
    # Find parcel-like substrings (take last/longest match; APN often at end)
    matches = list(_PARCEL_RE.finditer(s))
    if not matches:
        return s, None
    # Prefer last match (parcel often at end of line)
    m = matches[-1]
    parcel_raw = m.group(1)
    # Normalize: remove spaces, keep digits/hyphens/dots (and X) as-is for storage
    parcel = re.sub(r"\s+", "", parcel_raw)
    before = s[: m.start()].strip()
    after = s[m.end() :].strip()
    # Rejoin remainder; drop trailing/leading junk like "X" or "(01)"
    rest = " ".join(filter(None, [before, after])).strip()
    # If remainder looks like an address (has letters or street-like content), use it
    if rest and len(rest) >= 5 and re.search(r"[A-Za-z]", rest):
        address = rest
    else:
        address = None
    return address, parcel


def get_region_counties(region):
    """Get list of counties for a given region from county_region.csv."""
    county_region_df = pd.read_csv("ca_cafo_compliance/data/county_region.csv")
    return county_region_df[county_region_df["region"] == region]["county_name"].tolist()


def is_state_or_county_only_geocode(geocoded_address):
    """Return True if geocoded address is only state, county, or city level only)."""
    if pd.isna(geocoded_address) or not isinstance(geocoded_address, str):
        return True
    addr = geocoded_address.strip()
    if not addr:
        return True
    if re.search(r"\d", addr):  # Has a street number
        return False
    parts = [p.strip() for p in addr.split(",")]
    if len(parts) <= 1:  # state-only
        return True
    if "County" in parts[0]:  # county-only (e.g., "Fresno County, California")
        return True
    # City/state check: if no digits and 2-4 parts, likely just city/county/state
    # e.g., "Fresno, California, USA" or "Modesto, Stanislaus County, California, USA"
    if len(parts) <= 4:
        return True
    return False


def geocode_address(address, cache, county=None, try_again=False):
    if pd.isna(address) or not isinstance(address, str):
        return None, None
    clean_address = address.replace(": ", "")
    cached_addr = find_cached_address(clean_address, cache)
    if cached_addr:
        cached_result = cache[cached_addr]
        # Check if this is a minimal entry (no lat/lng) - treat as county/state-only
        if "lat" not in cached_result or "lng" not in cached_result:
            return None, None
        if try_again and (cached_result["lat"] is None or cached_result["lng"] is None):
            print(f"Retrying previously failed address: {clean_address}")
        else:
            # Treat state/county-only matches as no geocode (fill as NULL)
            cached_addr_str = cached_result.get("address")
            if cached_addr_str is not None and is_state_or_county_only_geocode(cached_addr_str):
                return None, None
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
                or ("California" not in location.address and "CA" not in location.address)
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
        # If result is only county/state level, save minimal cache entry without lat/lng
        if is_state_or_county_only_geocode(location.address):
            cache[clean_address] = {
                "timestamp": datetime.now().isoformat(),
                "address": location.address,
                "geocoder": geolocator.__class__.__name__,
            }
            save_geocoding_cache(cache)
            print(f"Geocoded to county/state only (skipping): {location.address}")
            return None, None

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


# California DWR parcel geocoding (APN -> lat/lng)
DWR_PARCEL_GEOCODE_URL = (
    "https://gis.water.ca.gov/arcgis/rest/services/Location/Geocoding_Parcels_APN_TaxAPN/"
    "GeocodeServer/findAddressCandidates"
)


def _normalize_parcel_for_cache(parcel_number):
    """Normalize APN to digits and hyphens for cache key and API (X -> 0, dots -> hyphens)."""
    if not parcel_number or not isinstance(parcel_number, str):
        return None
    s = re.sub(r"\s+", "", parcel_number.strip())
    s = s.replace(".", "-")  # dot separators -> hyphen for API
    s = re.sub(r"[Xx]", "0", s)  # OCR placeholder X -> 0 for geocoding
    if not re.match(r"^[\d\-]+$", s):
        return None
    return s


def geocode_parcel(parcel_number, cache):
    """
    Geocode a California assessor parcel number (APN) via DWR GeocodeServer.
    Returns (lat, lng) or (None, None). Uses cache key 'parcel:APN'.
    """
    apn = _normalize_parcel_for_cache(parcel_number)
    if not apn:
        return None, None
    cache_key = f"parcel:{apn}"
    if cache_key in cache:
        entry = cache[cache_key]
        lat, lng = entry.get("lat"), entry.get("lng")
        if lat is not None and lng is not None:
            addr = entry.get("address")
            if addr is not None:
                if isinstance(addr, dict):
                    addr = addr.get("Match_addr") or str(addr)
                if is_state_or_county_only_geocode(addr):
                    return None, None
            return lat, lng
        return None, None
    try:
        r = requests.get(
            DWR_PARCEL_GEOCODE_URL,
            params={"SingleLine": apn, "f": "json", "outFields": "*"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            cache[cache_key] = {
                "lat": None,
                "lng": None,
                "error": "No parcel candidates",
                "timestamp": datetime.now().isoformat(),
            }
            save_geocoding_cache(cache)
            return None, None
        loc = candidates[0].get("location") or {}
        lat, lng = loc.get("y"), loc.get("x")
        if lat is None or lng is None:
            cache[cache_key] = {
                "lat": None,
                "lng": None,
                "error": "No coordinates in response",
                "timestamp": datetime.now().isoformat(),
            }
            save_geocoding_cache(cache)
            return None, None
        address = candidates[0].get("address")
        if isinstance(address, dict):
            address = address.get("Match_addr") or ""
        address = address or ""

        # If result is only county/state level, save minimal cache entry without lat/lng
        if is_state_or_county_only_geocode(address):
            cache[cache_key] = {
                "timestamp": datetime.now().isoformat(),
                "address": address,
                "geocoder": "DWR_Parcels_APN",
            }
            save_geocoding_cache(cache)
            return None, None

        cache[cache_key] = {
            "lat": lat,
            "lng": lng,
            "address": address,
            "timestamp": datetime.now().isoformat(),
            "geocoder": "DWR_Parcels_APN",
        }
        save_geocoding_cache(cache)
        return lat, lng
    except Exception as e:
        cache[cache_key] = {
            "lat": None,
            "lng": None,
            "error": str(e),
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
