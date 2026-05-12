import os
import re
import requests
import json
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from postal.expand import expand_address
from postal.parser import parse_address
from dotenv import load_dotenv
from geopy.geocoders import ArcGIS, GoogleV3
from geopy.extra.rate_limiter import RateLimiter

from step2_extract_manifest_parameters import normalize_apn
from helpers import PARAMETERS_DF, PATH_TO_PDF_DATA, build_parameter_dicts, coerce_columns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "output_data")

MANUAL_PATH = os.path.join(OUTPUTS_DIR, "all_manifests_as_written_validated.csv")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "all_manifests_as_written_automatic.csv")

params = build_parameter_dicts(manifest_only=True)["key_to_name"]

SPECIFIC_COLS = {
    t: set(PARAMETERS_DF.loc[PARAMETERS_DF["manifest_type"] == t, "parameter_name"])
    for t in ["wastewater", "manure"]
}

METADATA_COLS = ["Source PDF", "Manifest Number", "Start Page", "End Page"]

DEST_TYPE_MAP = {
    "Composting Facility": ["compost", "kellogg", "hyponex", "fertilizer", "supply"],
    "Farmer": ["farm"],
    "Broker": ["broker"],
}

COORD_RE = re.compile(r"\s*\(?\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)?\s*$")
PO_BOX_RE = re.compile(r"\bP\.?O\.?\s*Box\b", re.IGNORECASE)
COUNTY_ALIASES = {
    "tulare_east": "Tulare",
    "tulare_west": "Tulare",
    "fresno_madera": "Fresno",
    "rancho_cordova": "Sacramento",
}
LOCALITY_TAGS = {"city", "state", "postcode", "state_district", "suburb"}


ZIP_TO_COUNTY = (
    pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "zipcode_to_county.csv"),
        usecols=["zip", "county_name"],
        dtype=str,
    )
    .drop_duplicates(subset="zip")
    .set_index("zip")["county_name"]
    .to_dict()
)


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


arcgis_geocoder = RateLimiter(
    ArcGIS(user_agent="ca_cafo_compliance").geocode,
    min_delay_seconds=0.2,
    max_retries=2,
    error_wait_seconds=1.0,
    swallow_exceptions=True,
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_GEOCODING_API_KEY", "")
if GOOGLE_API_KEY:
    google_geocoder = RateLimiter(
        GoogleV3(api_key=GOOGLE_API_KEY).geocode,
        min_delay_seconds=0.1,
        max_retries=2,
        error_wait_seconds=1.0,
        swallow_exceptions=True,
    )
else:
    print("Google geocoding disabled (no GOOGLE_GEOCODING_API_KEY)")
    google_geocoder = None


def geocode_if_valid(addr, geocode_fn, **kwargs):
    """Geocode address if valid, return (lat, lng) tuple or None."""
    if not isinstance(addr, str) or not addr.strip() or pd.isna(addr):
        return None
    res = geocode_fn(addr, **kwargs)
    if isinstance(res, (tuple, list)) and len(res) >= 2 and all(x is not None for x in res[:2]):
        return (res[0], res[1])
    return None


def geocode_parcel(parcel_number):
    apn = normalize_apn(parcel_number)
    if not apn:
        return None, None

    if apn in cache:
        return cache[apn]

    r = requests.get(
        "https://gis.water.ca.gov/arcgis/rest/services/Location/Geocoding_Parcels_APN_TaxAPN/"
        "GeocodeServer/findAddressCandidates",
        params={"SingleLine": apn, "f": "json", "outFields": "*"},
        timeout=15,
    )
    r.raise_for_status()
    candidates = r.json().get("candidates") or []

    if not candidates:
        cache[apn] = (None, None, {"source": "dwr_parcel"})
        return cache[apn]

    loc = candidates[0].get("location") or {}
    lat, lng = loc.get("y"), loc.get("x")
    address = candidates[0].get("address")
    if isinstance(address, dict):
        address = address.get("Match_addr") or ""

    result = (
        (lat, lng, {"source": "dwr_parcel"})
        if lat and lng and has_street_level(address or "")
        else (None, None, {"source": "dwr_parcel"})
    )
    cache[apn] = result
    return result


def norm_addr(s: str) -> str | None:
    if not isinstance(s, str) or not (s := s.replace(": ", " ").strip().lower()):
        return None
    if PO_BOX_RE.search(s):
        return None
    exps = expand_address(s, languages=["en"])
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

    county_norm = str(county).strip().lower() if pd.notna(county) and county else ""
    county_norm = "" if county_norm in ("nan", "none") else county_norm
    key = (na, county_norm)
    if key in cache:
        return cache[key]

    if not county_norm and not {t for _, t in parse_address(na)} & LOCALITY_TAGS:
        return None, None, None

    county_name = (
        COUNTY_ALIASES.get(county_norm, county_norm.title()) if county_norm else None
    )
    q = f"{address}, {county_name} County, CA" if county_name else f"{address}, CA"

    loc = arcgis_geocoder(q)
    source = "arcgis"
    street = loc and loc.address and has_street_level(loc.address)

    if google_geocoder and not street:
        g = google_geocoder(q, components={"country": "US", "administrative_area": "CA"})
        if g and g.address:
            loc, source = g, "google"
            street = has_street_level(g.address)

    if not loc or not loc.address:
        result = (None, None, {"source": "all_failed"})
    elif not street:
        result = (None, None, {"address": loc.address, "source": source})
    else:
        result = (loc.latitude, loc.longitude, {"address": loc.address, "source": source})

    cache[key] = result
    return result


def enrich_address_columns(
    df: pd.DataFrame, address_col: str, prefix="", county_col_in: str | None = None
):
    lat_col, lng_col = f"{prefix}Latitude", f"{prefix}Longitude"
    city_col, zip_col, county_col = f"{prefix}City", f"{prefix}Zip", f"{prefix}County"

    def enrich_one(row):
        addr = row[address_col]
        county = row.get(county_col_in) if county_col_in else None
        lat, lng, meta = geocode_address(addr, county=county)
        if lat is None:
            return pd.Series([None] * 5, index=[lat_col, lng_col, city_col, zip_col, county_col])

        address_str = (meta or {}).get("address") or ""
        parts = [p.strip() for p in address_str.split(",")]
        city = parts[-3] if len(parts) >= 3 else None
        zip_code = parts[-1].split()[-1] if parts else None

        return pd.Series(
            [
                lat,
                lng,
                city,
                zip_code,
                ZIP_TO_COUNTY.get(str(zip_code).strip() if zip_code is not None else ""),
            ],
            index=[lat_col, lng_col, city_col, zip_col, county_col],
        )

    df[[lat_col, lng_col, city_col, zip_col, county_col]] = df.apply(enrich_one, axis=1)
    return df


def merge_manifests():
    """Load and merge manual and extracted manifests."""
    manual_df = pd.read_csv(MANUAL_PATH, engine="python", on_bad_lines="warn")
    extracted_df = pd.read_csv(EXTRACTED_PATH)
    coerce_columns(manual_df)

    # Remove duplicates
    dupe_mask = manual_df.get("Is Duplicate", pd.Series()) == "x"
    if dupe_mask.any():
        manual_df = manual_df[~dupe_mask].reset_index(drop=True)

    # Merge in extracted columns
    key_cols = ["Source PDF", "Manifest Number"]
    cols_to_add = set(extracted_df.columns) - set(manual_df.columns) - set(key_cols)
    print(f"\nColumns to add from extracted_manifests: {sorted(cols_to_add)}")

    extracted_deduped = extracted_df.drop_duplicates(subset=key_cols, keep="first")
    merged = manual_df.merge(
        extracted_deduped[list(cols_to_add) + key_cols], on=key_cols, how="left"
    )

    for col in cols_to_add:
        manual_df[col] = merged[col]

    # Drop exact duplicates
    dup_subset = [
        c for c in manual_df.columns if c not in ["Manifest Number", "Start Page", "End Page"]
    ]
    manual_df = manual_df.drop_duplicates(subset=dup_subset)

    return manual_df


def resolve_destination_address(row, has_existing_coords, dest_geocoded):
    """Resolve final destination address and geocode it. Returns (val, src, geocoded)."""
    def get_str(k):
        v = row.get(params[k])
        return str(v).strip() if pd.notna(v) else ""

    parcel_county = get_str("destination_county") or None
    dest_address_present = False

    # Priority 1: Parcel number
    if is_valid_string(row.get(params["destination_parcel_number"])):
        dest_address_present = True
        raw_str = get_str("destination_parcel_number")
        if not has_existing_coords:
            parts = [p.strip() for p in raw_str.split(",") if p.strip()]
            hits = [r for p in parts if (r := geocode_if_valid(p, geocode_parcel))]
            if hits:
                return raw_str, params["destination_parcel_number"], hits[0]
        return raw_str, params["destination_parcel_number"], dest_geocoded

    # Priority 2: Cross street (if it looks like coordinates)
    if is_valid_string(row.get(params["destination_nearest_cross_street"])):
        dest_address_present = True
        raw_str = get_str("destination_nearest_cross_street")
        m = COORD_RE.match(raw_str)
        if m:
            if not has_existing_coords:
                dest_geocoded = (float(m.group(1)), float(m.group(2)))
            return raw_str, params["destination_nearest_cross_street"], dest_geocoded

    # Priority 3: Destination address (+cross street/county)
    if is_valid_string(row.get(params["destination_address"])):
        dest_address_present = True
        raw_str = get_str("destination_address")
        cross = row.get(params["destination_nearest_cross_street"])
        if is_valid_string(cross) and not COORD_RE.match(str(cross)):
            raw_str = f"{raw_str} {str(cross).strip()}"
        if parcel_county:
            raw_str = f"{raw_str} {parcel_county}"
        if not has_existing_coords:
            dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)
        return raw_str, params["destination_address"], dest_geocoded

    # Priority 4: Contact address (only if no destination address fields present)
    if not dest_address_present and is_valid_string(row.get(params["destination_contact_address"])):
        raw_str = get_str("destination_contact_address")
        if len(re.sub(r"[^a-zA-Z0-9]", "", raw_str)) >= 5:
            if not has_existing_coords:
                if g := geocode_if_valid(raw_str, geocode_address, county=parcel_county):
                    return raw_str, params["destination_contact_address"], g
            return raw_str, params["destination_contact_address"], dest_geocoded

    # Priority 5: Hauler address (only if all above empty, and looks like farm/compost/fertilizer)
    if not dest_address_present and is_valid_string(row.get(params["hauler_address"])):
        hauler_combined = " ".join(
            (str(v).lower() if pd.notna(v := row.get(params[k])) else "")
            for k in ["hauler_name", "hauler_address"]
        )
        if any(s in hauler_combined for s in ("farm", "compost", "fertilizer")):
            raw_str = get_str("hauler_address")
            if not has_existing_coords:
                dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)
            return raw_str, params["hauler_address"], dest_geocoded

    return None, None, None


def is_valid_string(val):
    """Check if value is a valid non-empty string."""
    return pd.notna(val) and str(val).strip()


def geocode_manifests(df):
    """Geocode origins and destinations."""
    print("\nResolving Destination Address Final + Geocoding")

    # Initialize columns
    df[[params["destination_address_final"], params["destination_address_final_source"]]] = None
    for key in ["origin_geo_lat", "origin_geo_lng", "destination_geo_lat", "destination_geo_lng"]:
        df[params[key]] = None

    for idx, row in df.iterrows():
        # Geocode origin
        addr = row[params["origin_dairy_address"]]
        county = row.get("County")
        if addr and (r := geocode_if_valid(addr, geocode_address, county=county)):
            df.at[idx, params["origin_geo_lat"]] = r[0]
            df.at[idx, params["origin_geo_lng"]] = r[1]

        # Check for existing manual coordinates (highest priority)
        existing_lat = pd.to_numeric(row.get(params["latitude"]), errors="coerce")
        existing_lng = pd.to_numeric(row.get(params["longitude"]), errors="coerce")
        has_existing_coords = pd.notna(existing_lat) and pd.notna(existing_lng)
        dest_geocoded = (existing_lat, existing_lng) if has_existing_coords else None

        val, src, dest_geocoded = resolve_destination_address(
            row, has_existing_coords, dest_geocoded
        )

        if val:
            df.at[idx, params["destination_address_final"]] = val
            df.at[idx, params["destination_address_final_source"]] = src
        if dest_geocoded:
            df.at[idx, params["destination_geo_lat"]] = dest_geocoded[0]
            df.at[idx, params["destination_geo_lng"]] = dest_geocoded[1]

    # Enrich address columns
    enrich_address_columns(
        df, params["origin_dairy_address"], prefix="Origin ", county_col_in="County"
    )
    enrich_address_columns(df, params["destination_address"], prefix="Destination ")

    return df


def backfill_columns(df):
    """Backfill mass, solids, and origin addresses."""
    # Backfill mass from volume
    backfill_rules = [
        (params["manure_amount"], params["manure_amount_yards"]),
        (params["manure_ton_per_haul"], params["manure_yard_per_haul"]),
    ]
    for mass_col, vol_col in backfill_rules:
        backfill = df[mass_col].isna() & df[vol_col].notna()
        df.loc[backfill, mass_col] = (
            df.loc[backfill, vol_col] * df.loc[backfill, params["manure_density"]]
        )
        print(f"  Backfilled {backfill.sum()} mass values for {mass_col}")

    # Remove invalid solids >100%
    solids_col = params["manure_solids_percent"]
    invalid_solids = df[solids_col] > 100
    if invalid_solids.any():
        print(f"  Nulled {invalid_solids.sum()} {solids_col} values >100")
        df.loc[invalid_solids, solids_col] = np.nan

    # Backfill solids from moisture
    backfill_solids = df[solids_col].isna() & df[params["manure_moisture_percent"]].notna()
    df.loc[backfill_solids, solids_col] = (
        100 - df.loc[backfill_solids, params["manure_moisture_percent"]]
    )
    print(f"  Backfilled {backfill_solids.sum()} values for {solids_col}")

    df.drop(
        columns=[src for _, src in backfill_rules]
        + [params["manure_density"], params["manure_moisture_percent"]],
        inplace=True,
    )

    # Backfill origin addresses from dairy summary
    dairy_summary_df = pd.read_csv(
        os.path.join(
            PATH_TO_PDF_DATA, "Dairy_Report_Summary_Region_5_2024_with_source_pdf.csv"
        )
    )
    origin_col = params["origin_dairy_address"]
    dairy_summary_df = dairy_summary_df.rename(columns={"Dairy Address": origin_col})
    dairy_summary_df["Source PDF"] = dairy_summary_df["Source PDF"].str.replace(
        r"\.pdf$", "", regex=True
    )
    dairy_name_to_addr = dairy_summary_df.drop_duplicates(subset="Dairy Name").set_index(
        "Dairy Name"
    )[origin_col]

    needs_backfill = df[origin_col].isna() | df["Origin Dairy Latitude (Geocoded)"].isna()
    print(f"{needs_backfill.sum()} need origin dairy address backfill")

    still_missing = needs_backfill & df[origin_col].isna()
    df.loc[still_missing, origin_col] = df.loc[still_missing, params["origin_dairy_name"]].map(
        dairy_name_to_addr
    )

    # Extract from PDF filename
    def addr_from_pdf_name(name):
        if re.match(r"^\d{4}[A-Z]", name):
            m = re.search(r"Dairy[^_]*_(.+)", name)
            return m.group(1).replace("_", " ").strip() if m else None
        m = re.match(r"(.+?)\s+\d{4}", name)
        return m.group(1).strip() if m else None

    still_missing = df[origin_col].isna()
    df.loc[still_missing, origin_col] = df.loc[still_missing, "Source PDF"].map(addr_from_pdf_name)

    newly_filled = still_missing & df[origin_col].notna()
    for idx in df.index[newly_filled]:
        addr = df.at[idx, origin_col]
        county = df.at[idx, "County"] if "County" in df.columns else None
        if r := geocode_if_valid(addr, geocode_address, county=county):
            df.at[idx, params["origin_geo_lat"]], df.at[idx, params["origin_geo_lng"]] = r
    print(f"Geocoded {newly_filled.sum()} addresses from PDF filename")

    return df


def add_haul_estimates(df, label, rate_col, haul_col, amount_col):
    """Add estimated number of hauls based on bins."""
    has_raw = df[rate_col].notna() & df[haul_col].notna()
    n_hauls = pd.to_numeric(df[haul_col], errors="coerce").round().astype("Int64")
    zero = pd.array([0] * len(df), dtype="Int64")

    if label == "Wastewater":
        cutoff = 8000
        bin_lo, bin_hi = "<8,000 gal", ">=8,000 gal"
        divisor_lo, divisor_hi = 2000.0, 10000.0
        rate = df[rate_col]
        in_lo_bin = rate < cutoff
        in_hi_bin = rate >= cutoff
    else:  # Manure
        lo1, hi1, lo2, hi2, bin_lo, bin_hi = 5, 15, 15, 25, "10-ton", "20-ton"
        divisor_lo, divisor_hi = 10.0, 20.0
        rate = df[rate_col]
        in_lo_bin = rate.between(lo1, hi1, inclusive="left")
        in_hi_bin = rate.between(lo2, hi2, inclusive="left")

    # Calculate proportions from existing data
    amount = df[amount_col]
    mass_lo = amount[in_lo_bin].sum()
    mass_hi = amount[in_hi_bin].sum()

    total = mass_lo + mass_hi
    p_lo = mass_lo / total if total > 0 else 0.5
    p_hi = 1.0 - p_lo

    # Estimate hauls for rows without raw data
    est_lo = (amount * p_lo / divisor_lo).round().astype("Int64")
    est_hi = (amount * p_hi / divisor_hi).round().astype("Int64")

    df[f"Estimated Number of {bin_lo} Hauls"] = est_lo.where(~has_raw)
    df[f"Estimated Number of {bin_hi} Hauls"] = est_hi.where(~has_raw)

    # For analysis: use actual when available, estimates otherwise
    in_lo = has_raw & in_lo_bin
    in_hi = has_raw & in_hi_bin

    df[f"Number of {bin_lo} Hauls for Analysis"] = est_lo.where(~in_lo, n_hauls).where(~in_hi, zero)
    df[f"Number of {bin_hi} Hauls for Analysis"] = est_hi.where(~in_hi, n_hauls).where(~in_lo, zero)


def calculate_distance(row):
    """Calculate miles between origin and destination."""
    try:
        o_lat = float(row.get(params["origin_geo_lat"], np.nan))
        o_lng = float(row.get(params["origin_geo_lng"], np.nan))
        d_lat = float(row.get(params["destination_geo_lat"], np.nan))
        d_lng = float(row.get(params["destination_geo_lng"], np.nan))
        if any(np.isnan([o_lat, o_lng, d_lat, d_lng])):
            return np.nan
        return geodesic((o_lat, o_lng), (d_lat, d_lng)).miles
    except (TypeError, ValueError):
        return np.nan


def save_manifest_type(df, label, specific_cols, output_data_dir, suffix=""):
    """Save processed manifest CSV with correct column ordering."""
    param_order = PARAMETERS_DF["parameter_name"].tolist()
    type_qty_cols = [
        c
        for c in param_order
        if c in specific_cols[label.lower()] and c in df.columns and not c.startswith("Method Used")
    ]
    estimated_cols = [c for c in df.columns if c.startswith("Number of")]
    extra_cols = ["origin_dest_miles"] if "origin_dest_miles" in df.columns else []

    cols = []
    for c in (
        METADATA_COLS
        + [
            params["origin_dairy_name"],
            params["origin_dairy_address"],
            params["origin_geo_lat"],
            params["origin_geo_lng"],
            params["destination_address_final"],
            params["destination_address_final_source"],
            params["destination_type_std"],
            params["destination_geo_lat"],
            params["destination_geo_lng"],
            params["haul_date_first"],
            params["haul_date_last"],
            params["is_pipeline"],
            params["is_trucked"],
        ]
        + type_qty_cols
        + estimated_cols
        + extra_cols
    ):
        if c in df.columns and c not in cols:
            cols.append(c)

    filename = f"processed_{label.lower()}_manifests{suffix}.csv"
    df[cols].to_csv(os.path.join(output_data_dir, filename), index=False)
    print(f"Saved {len(df)} rows to {filename}")


def main():
    # Merge and process manifests
    df = merge_manifests()
    df = geocode_manifests(df)

    # Backfill and standardize
    df = backfill_columns(df)

    # Standardize destination type
    def std_dest_type(val):
        if pd.isna(val) or not str(val).strip():
            return "Blank"
        vl = str(val).lower()
        matched = [c for c, kws in DEST_TYPE_MAP.items() if any(kw in vl for kw in kws)]
        if not matched:
            return "Other"
        return matched[0] if len(matched) == 1 else ", ".join(sorted(set(matched)))

    df[params["destination_type_std"]] = df[params["destination_type"]].apply(std_dest_type)

    # Split by manifest type
    manure_cols = [c for c in df.columns if c not in SPECIFIC_COLS["wastewater"]]
    wastewater_cols = [c for c in df.columns if c not in SPECIFIC_COLS["manure"]]

    df_manure = df.loc[df["Manifest Type"].isin(["manure", "both"]), manure_cols].copy()
    df_manure[params["is_trucked"]] = True
    df_ww = df.loc[df["Manifest Type"].isin(["wastewater", "both"]), wastewater_cols].copy()

    print(f"\nManure + both: {len(df_manure)} rows")
    print(f"Wastewater + both: {len(df_ww)} rows")

    # Calculate distance for wastewater
    df_ww["origin_dest_miles"] = df_ww.apply(calculate_distance, axis=1)

    # Add haul estimates
    add_haul_estimates(
        df_manure,
        "Manure",
        params["manure_ton_per_haul"],
        params["manure_number_hauls"],
        params["manure_amount"],
    )
    add_haul_estimates(
        df_ww,
        "Wastewater",
        params["wastewater_gallon_per_haul"],
        params["wastewater_number_hauls"],
        params["wastewater_amount"],
    )

    # Print unique wastewater methods
    methods = df_ww[params["wastewater_method"]].dropna().unique()
    print(f"\nUnique 'Method Used for Analysis' values in wastewater manifests ({len(methods)}):")
    for m in methods:
        print(f"  {m}")

    # Save output_data
    save_manifest_type(df_manure, "Manure", SPECIFIC_COLS, OUTPUTS_DIR)
    save_manifest_type(df_ww, "Wastewater", SPECIFIC_COLS, OUTPUTS_DIR)


if __name__ == "__main__":
    main()
