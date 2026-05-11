import json
import os
import re
from dotenv import load_dotenv
from geopy.geocoders import ArcGIS, GoogleV3
from geopy.extra.rate_limiter import RateLimiter

load_dotenv()

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "output_data", "geocode_cache.json")


class JsonCache:
    def __init__(self, path):
        self._path = path
        with open(path) as f:
            self._data = json.load(f)

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


_arcgis = RateLimiter(
    ArcGIS(user_agent="ca_cafo_compliance").geocode,
    min_delay_seconds=0.2,
    max_retries=2,
    error_wait_seconds=1.0,
    swallow_exceptions=True,
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_GEOCODING_API_KEY", "")
_google = (
    RateLimiter(
        GoogleV3(api_key=GOOGLE_API_KEY).geocode,
        min_delay_seconds=0.1,
        max_retries=2,
        error_wait_seconds=1.0,
        swallow_exceptions=True,
    )
    if GOOGLE_API_KEY
    else (print("Google geocoding disabled (no GOOGLE_GEOCODING_API_KEY)"), None)[1]
)


def normalize_apn(parcel_number):
    if not isinstance(parcel_number, str) or not parcel_number.strip():
        return None
    s = re.match(r"[\d\s.\-Xx]+", parcel_number.strip())
    if not s:
        return None
    s = re.sub(r"\s+", "", s.group()).replace(".", "-")
    s = re.sub(r"[Xx]", "0", s)
    return s if re.fullmatch(r"[\d\-]+", s) else None
