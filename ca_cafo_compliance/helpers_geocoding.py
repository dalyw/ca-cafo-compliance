import json
import os
import re
import requests
import pandas as pd
from dotenv import load_dotenv
from geopy.geocoders import ArcGIS, GoogleV3
from geopy.extra.rate_limiter import RateLimiter
from postal.expand import expand_address
from postal.parser import parse_address

load_dotenv()

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "outputs", "geocode_cache.json")


class JsonCache:
    def __init__(self, path):
        self._path = path
        try:
            with open(path) as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    def _key(self, k):
        return "|".join(str(x) for x in k if x) if isinstance(k, tuple) else str(k)

    def __contains__(self, k):
        return self._key(k) in self._data

    def __getitem__(self, k):
        val = self._data[self._key(k)]
        return tuple(val) if isinstance(val, list) else val

    def __setitem__(self, k, v):
        self._data[self._key(k)] = v
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)


cache = JsonCache(_CACHE_PATH)

ZIP_TO_COUNTY = (
    pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "zipcode_to_county.csv"),
                usecols=["zip", "county_name"], dtype=str)
    .drop_duplicates(subset="zip")
    .set_index("zip")["county_name"]
    .to_dict()
)

_arcgis = RateLimiter(
    ArcGIS(user_agent="ca_cafo_compliance").geocode,
    min_delay_seconds=0.2,
    max_retries=2,
    error_wait_seconds=1.0,
    swallow_exceptions=True,
)
GOOGLE_API_KEY = os.environ.get("GOOGLE_GEOCODING_API_KEY", "")
if GOOGLE_API_KEY:
    _google = RateLimiter(
        GoogleV3(api_key=GOOGLE_API_KEY).geocode,
        min_delay_seconds=0.1,
        max_retries=2,
        error_wait_seconds=1.0,
        swallow_exceptions=True,
    )
    print(f"Google geocoding enabled")
else:
    _google = None
    print("Google geocoding disabled (no GOOGLE_GEOCODING_API_KEY)")

DWR_PARCEL_GEOCODE_URL = (
"https://gis.water.ca.gov/arcgis/rest/services/Location/Geocoding_Parcels_APN_TaxAPN/"
"GeocodeServer/findAddressCandidates"
)

def norm_addr(s: str) -> str | None:
    if not isinstance(s, str) or not (s := s.replace(": ", " ").strip().lower()):
        return None
    exps = expand_address(s)
    return exps[0] if exps else s


def has_street_level(s: str) -> bool:
    return (
        isinstance(s, str)
        and s.strip()
        and any(t in {"house_number", "road"} for _, t in parse_address(s))
    )


def geocode_address(address: str, county: str | None = None):
    na = norm_addr(address)
    if not na:
        return None, None, None

    key = (na, (county or "").strip().lower())
    if key in cache:
        return cache[key]

    q = f"{address}, {county} County, CA" if county else f"{address}, CA"

    # Try ArcGIS first
    loc = _arcgis(q)
    source = "arcgis"
    street = loc and loc.address and has_street_level(loc.address)

    # Fall back to Google if ArcGIS missed or returned non-street-level
    if _google and not street:
        g = _google(q, components={"country": "US", "administrative_area": "CA"})
        if g and g.address:
            loc, source = g, "google"
            street = has_street_level(g.address)

    if not loc or not loc.address:
        cache[key] = (None, None, {"source": "all_failed"})
    elif not street:
        cache[key] = (None, None, {"address": loc.address, "source": source})
    else:
        cache[key] = (loc.latitude, loc.longitude, {"address": loc.address, "source": source})

    return cache[key]


def county_from_zip(zip_code: str) -> str | None:
    z = str(zip_code).strip() if zip_code is not None else ""
    return ZIP_TO_COUNTY.get(z)


def enrich_address_columns(df: pd.DataFrame, address_col: str, prefix="", county_col_in: str | None = None):
    lat_col, lng_col = f"{prefix}Latitude", f"{prefix}Longitude"
    city_col, zip_col, county_col = f"{prefix}City", f"{prefix}Zip", f"{prefix}County"

    def enrich_one(row):
        addr = row[address_col]
        county = row.get(county_col_in) if county_col_in else None
        lat, lng, meta = geocode_address(addr, county=county)
        if lat is None:
            return pd.Series(
                [None] * 5, index=[lat_col, lng_col, city_col, zip_col, county_col]
            )

        formatted = (meta or {}).get("address") or ""
        parts = [p.strip() for p in formatted.split(",")]

        city = parts[-3] if len(parts) >= 3 else None
        zip_code = parts[-1].split()[-1] if parts else None

        return pd.Series(
            [lat, lng, city, zip_code, county_from_zip(zip_code)],
            index=[lat_col, lng_col, city_col, zip_col, county_col],
        )

    df[[lat_col, lng_col, city_col, zip_col, county_col]] = df.apply(
        enrich_one, axis=1
    )
    return df


# APN detection/parsing (mostly unchanged)
def looks_like_parcel_number(text):
    if not isinstance(text, str) or not (s := text.strip()) or len(s) < 3:
        return False
    return (
        (("-" in s) or ("." in s))
        and bool(re.fullmatch(r"[\d\s.\-]+", s))
        and bool(re.search(r"\d+\s*[.\-]\s*\d+", s))
    )


_PARCEL_RE = re.compile(
    r"(?:\(?\d*\)?\s*[Xx]?\s*)?([\dXx]{2,}\s*[.\-]\s*[\dXx]{2,}(?:\s*[.\-]\s*[\dXx]+)*)",
    re.IGNORECASE,
)


def parse_destination_address_and_parcel(value):
    if not isinstance(value, str) or not (s := value.strip()):
        return None, None

    if looks_like_parcel_number(s):
        return None, re.sub(r"\s+", "", s)

    matches = list(_PARCEL_RE.finditer(s))
    if not matches:
        return s, None

    m = matches[-1]
    parcel = re.sub(r"\s+", "", m.group(1))
    rest = " ".join(
        filter(None, [s[: m.start()].strip(), s[m.end() :].strip()])
    ).strip()
    address = (
        rest if (rest and len(rest) >= 5 and re.search(r"[A-Za-z]", rest)) else None
    )
    return address, parcel


def _normalize_apn(parcel_number):
    """Normalize APN: remove spaces, dots->hyphens, X->0."""
    if not isinstance(parcel_number, str) or not parcel_number.strip():
        return None
    s = re.sub(r"\s+", "", parcel_number.strip())
    s = s.replace(".", "-")
    s = re.sub(r"[Xx]", "0", s)
    return s if re.fullmatch(r"[\d\-]+", s) else None


def geocode_parcel(parcel_number):
    """Geocode a CA assessor parcel number via DWR GeocodeServer.
    Returns (lat, lng) or (None, None).
    """
    apn = _normalize_apn(parcel_number)
    if not apn:
        return None, None

    key = apn
    if key in cache:
        return cache[key]

    try:
        r = requests.get(
            DWR_PARCEL_GEOCODE_URL,
            params={"SingleLine": apn, "f": "json", "outFields": "*"},
            timeout=15,
        )
        r.raise_for_status()
        candidates = r.json().get("candidates") or []
        if not candidates:
            cache[key] = (None, None, {"source": "dwr_parcel"})
            return cache[key]

        loc = candidates[0].get("location") or {}
        lat, lng = loc.get("y"), loc.get("x")
        address = candidates[0].get("address")
        if isinstance(address, dict):
            address = address.get("Match_addr") or ""

        if lat is None or lng is None or not has_street_level(address or ""):
            cache[key] = (None, None, {"source": "dwr_parcel"})
        else:
            cache[key] = (lat, lng, {"source": "dwr_parcel"})

        return cache[key]
    except Exception:
        cache[key] = (None, None, {"source": "dwr_parcel"})
        return cache[key]