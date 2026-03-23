import os
import re
import numpy as np
import pandas as pd
from geopy.distance import geodesic

from helpers_geocoding import enrich_address_columns, geocode_address, geocode_parcel
from helpers_pdf_metrics import PARAMETERS_DF, GDRIVE_BASE, build_parameter_dicts, coerce_columns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "compiled_data")

MANUAL_PATH = os.path.join(OUTPUTS_DIR, "all_manifests_as_written_validated.csv")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "all_manifests_as_written_automatic.csv")

P = build_parameter_dicts(manifest_only=True)["key_to_name"]

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

_COORD_RE = re.compile(r"\s*\(?\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)?\s*$")


def geocode_if_valid(addr, geocode_fn, **kwargs):
    """Geocode address if valid, return (lat, lng) tuple or None."""
    if not isinstance(addr, str) or not addr.strip() or pd.isna(addr):
        return None
    res = geocode_fn(addr, **kwargs)
    if isinstance(res, (tuple, list)) and len(res) >= 2 and all(x is not None for x in res[:2]):
        return (res[0], res[1])
    return None


def weighted_avg(df, val_col, weight_col):
    """Calculate weighted average, dropping NA values."""
    valid = df.dropna(subset=[val_col, weight_col])
    return (valid[val_col] * valid[weight_col]).sum() / valid[weight_col].sum()


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
    merged = manual_df.merge(extracted_deduped[list(cols_to_add) + key_cols], on=key_cols, how="left")
    
    for col in cols_to_add:
        if col in merged.columns:
            manual_df[col] = merged[col]

    # Drop exact duplicates
    dup_subset = [c for c in manual_df.columns if c not in ["Manifest Number", "Start Page", "End Page"]]
    manual_df = manual_df.drop_duplicates(subset=dup_subset)
    
    return manual_df


def resolve_destination_address(row, has_existing_coords):
    """Resolve final destination address and geocode it."""
    parcel_county = str(row.get(P["destination_county"])).strip() if pd.notna(row.get(P["destination_county"])) else None
    dest_geocoded = None
    dest_address_present = False
    
    # Priority 1: Parcel number
    raw_parcel = row.get(P["destination_parcel_number"])
    if pd.notna(raw_parcel) and str(raw_parcel).strip():
        dest_address_present = True
        raw_str = str(raw_parcel).strip()
        if not has_existing_coords:
            parts = [p.strip() for p in raw_str.split(",") if p.strip()]
            hits = [r for p in parts if (r := geocode_if_valid(p, geocode_parcel))]
            if hits:
                return raw_str, P["destination_parcel_number"], hits[0]
        return raw_str, P["destination_parcel_number"], dest_geocoded

    # Priority 2: Cross street (if coordinates)
    raw_cross = row.get(P["destination_nearest_cross_street"])
    if pd.notna(raw_cross) and str(raw_cross).strip():
        dest_address_present = True
        raw_str = str(raw_cross).strip()
        m = _COORD_RE.match(raw_str)
        if m:
            if not has_existing_coords:
                dest_geocoded = (float(m.group(1)), float(m.group(2)))
            return raw_str, P["destination_nearest_cross_street"], dest_geocoded

    # Priority 3: Destination address
    raw_addr = row.get(P["destination_address"])
    if pd.notna(raw_addr) and str(raw_addr).strip():
        dest_address_present = True
        raw_str = str(raw_addr).strip()
        cross = row.get(P["destination_nearest_cross_street"])
        if pd.notna(cross) and str(cross).strip() and not _COORD_RE.match(str(cross)):
            raw_str = f"{raw_str} {str(cross).strip()}"
        if parcel_county:
            raw_str = f"{raw_str} {parcel_county}"
        if not has_existing_coords:
            dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)
        return raw_str, P["destination_address"], dest_geocoded

    # Priority 4: Contact address (only if no other destination fields)
    if not dest_address_present:
        raw_contact = row.get(P["destination_contact_address"])
        if pd.notna(raw_contact) and str(raw_contact).strip():
            raw_str = str(raw_contact).strip()
            if len(re.sub(r"[^a-zA-Z0-9]", "", raw_str)) >= 5:
                if not has_existing_coords:
                    g = geocode_if_valid(raw_str, geocode_address, county=parcel_county)
                    if g:
                        return raw_str, P["destination_contact_address"], g
                return raw_str, P["destination_contact_address"], dest_geocoded

        # Priority 5: Hauler address (must look like farm/compost/fertilizer)
        raw_hauler = row.get(P["hauler_address"])
        if pd.notna(raw_hauler) and str(raw_hauler).strip():
            hauler_combined = f"{str(row.get(P['hauler_name'], '') or '').lower()} {str(raw_hauler).lower()}"
            if any(s in hauler_combined for s in ("farm", "compost", "fertilizer")):
                raw_str = str(raw_hauler).strip()
                if not has_existing_coords:
                    dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)
                return raw_str, P["hauler_address"], dest_geocoded

    return None, None, None


def is_valid_string(val):
    """Check if value is a valid non-empty string."""
    return pd.notna(val) and str(val).strip()


def geocode_manifests(df):
    """Geocode origins and destinations."""
    print("\nResolving Destination Address Final + Geocoding")
    
    # Initialize columns
    df[[P["destination_address_final"], P["destination_address_final_source"]]] = None
    for col in ["origin_geo_lat", "origin_geo_lng", "destination_geo_lat", "destination_geo_lng"]:
        df[P[col]] = None

    source_counts = {}

    for idx, row in df.iterrows():
        # Geocode origin
        addr = row[P["origin_dairy_address"]]
        county = row.get("County")
        if addr and (r := geocode_if_valid(addr, geocode_address, county=county)):
            df.at[idx, P["origin_geo_lat"]] = r[0]
            df.at[idx, P["origin_geo_lng"]] = r[1]

        # Resolve destination
        parcel_county = str(row.get(P["destination_county"])).strip() if is_valid_string(row.get(P["destination_county"])) else None

        # Check for existing manual coordinates (highest priority)
        existing_lat = pd.to_numeric(row.get(P["latitude"]), errors="coerce")
        existing_lng = pd.to_numeric(row.get(P["longitude"]), errors="coerce")
        has_existing_coords = pd.notna(existing_lat) and pd.notna(existing_lng)

        val = src = None
        dest_geocoded = (existing_lat, existing_lng) if has_existing_coords else None
        dest_address_present = False

        # Priority 1: Parcel number
        if is_valid_string(row.get(P["destination_parcel_number"])):
            dest_address_present = True
            raw_str = str(row.get(P["destination_parcel_number"])).strip()
            if has_existing_coords:
                val, src = raw_str, P["destination_parcel_number"]
            else:
                parts = [x.strip() for x in raw_str.split(",") if x.strip()]
                hits = [r for p in parts if (r := geocode_if_valid(p, geocode_parcel))]
                if hits:
                    val, src, dest_geocoded = raw_str, P["destination_parcel_number"], hits[0]

        # Priority 2: Cross street (if it looks like coordinates)
        if not val and is_valid_string(row.get(P["destination_nearest_cross_street"])):
            dest_address_present = True
            raw_str = str(row.get(P["destination_nearest_cross_street"])).strip()
            m = _COORD_RE.match(raw_str)
            if m:
                val, src = raw_str, P["destination_nearest_cross_street"]
                if not has_existing_coords:
                    dest_geocoded = (float(m.group(1)), float(m.group(2)))

        # Priority 3: Destination address (+cross street/county)
        if not val and is_valid_string(row.get(P["destination_address"])):
            dest_address_present = True
            raw_str = str(row.get(P["destination_address"])).strip()
            cross = row.get(P["destination_nearest_cross_street"])
            if is_valid_string(cross) and not _COORD_RE.match(str(cross)):
                raw_str = f"{raw_str} {str(cross).strip()}"
            if parcel_county:
                raw_str = f"{raw_str} {parcel_county}"
            val, src = raw_str, P["destination_address"]
            if not has_existing_coords:
                dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)

        # Priority 4: Contact address (only if no destination address fields present)
        if not val and not dest_address_present and is_valid_string(row.get(P["destination_contact_address"])):
            raw_str = str(row.get(P["destination_contact_address"])).strip()
            if len(re.sub(r"[^a-zA-Z0-9]", "", raw_str)) >= 5:
                if has_existing_coords:
                    val, src = raw_str, P["destination_contact_address"]
                else:
                    if g := geocode_if_valid(raw_str, geocode_address, county=parcel_county):
                        val, src, dest_geocoded = raw_str, P["destination_contact_address"], g

        # Priority 5: Hauler address (only if all above empty, and looks like farm/compost/fertilizer)
        if not val and not dest_address_present and is_valid_string(row.get(P["hauler_address"])):
            hauler_combined = f"{str(row.get(P['hauler_name'], '') or '').lower()} {str(row.get(P['hauler_address'])).lower()}"
            if any(s in hauler_combined for s in ("farm", "compost", "fertilizer")):
                raw_str = str(row.get(P["hauler_address"])).strip()
                val, src = raw_str, P["hauler_address"]
                if not has_existing_coords:
                    dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)

        # Save results
        if val:
            df.at[idx, P["destination_address_final"]] = val
            df.at[idx, P["destination_address_final_source"]] = src
            source_counts[src] = source_counts.get(src, 0) + 1
        if dest_geocoded:
            df.at[idx, P["destination_geo_lat"]] = dest_geocoded[0]
            df.at[idx, P["destination_geo_lng"]] = dest_geocoded[1]

    # Enrich address columns
    enrich_address_columns(df, P["origin_dairy_address"], prefix="Origin ", county_col_in="County")
    enrich_address_columns(df, P["destination_address"], prefix="Destination ")

    return df


def backfill_columns(df):
    """Backfill mass, solids, and origin addresses."""
    # Backfill mass from volume
    backfill_rules = [
        (P["manure_amount"], P["manure_amount_yards"]),
        (P["manure_ton_per_haul"], P["manure_yard_per_haul"]),
    ]
    for mass_col, vol_col in backfill_rules:
        backfill = df[mass_col].isna() & df[vol_col].notna()
        df.loc[backfill, mass_col] = df.loc[backfill, vol_col] * df.loc[backfill, P["manure_density"]]
        print(f"  Backfilled {backfill.sum()} mass values for {mass_col}")

    # Backfill solids from moisture
    backfill_solids = df[P["manure_solids_percent"]].isna() & df[P["manure_moisture_percent"]].notna()
    df.loc[backfill_solids, P["manure_solids_percent"]] = 100 - df.loc[backfill_solids, P["manure_moisture_percent"]]
    print(f"  Backfilled {backfill_solids.sum()} values for {P['manure_solids_percent']}")

    df.drop(columns=[src for _, src in backfill_rules] + [P["manure_density"], P["manure_moisture_percent"]], 
            errors="ignore", inplace=True)

    # Backfill origin addresses from dairy summary
    dairy_summary_df = pd.read_csv(
        os.path.join(GDRIVE_BASE, "data/Dairy_Report_Summary_Region_5_2024_with_source_pdf.csv")
    )
    origin_col = P["origin_dairy_address"]
    dairy_summary_df = dairy_summary_df.rename(columns={"Dairy Address": origin_col})
    dairy_summary_df["Source PDF"] = dairy_summary_df["Source PDF"].str.replace(r"\.pdf$", "", regex=True)
    dairy_name_to_addr = dairy_summary_df.drop_duplicates(subset="Dairy Name").set_index("Dairy Name")[origin_col]

    needs_backfill = df[origin_col].isna() | df["Origin Dairy Latitude (Geocoded)"].isna()
    print(f"{needs_backfill.sum()} need origin dairy address backfill")
    
    still_missing = needs_backfill & df[origin_col].isna()
    df.loc[still_missing, origin_col] = df.loc[still_missing, P["origin_dairy_name"]].map(dairy_name_to_addr)

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
            df.at[idx, P["origin_geo_lat"]], df.at[idx, P["origin_geo_lng"]] = r
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
    if label == "Wastewater":
        mass_lo = amount[in_lo_bin].sum()
        mass_hi = amount[in_hi_bin].sum()
    else:
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
        o_lat = float(row.get(P["origin_geo_lat"], np.nan))
        o_lng = float(row.get(P["origin_geo_lng"], np.nan))
        d_lat = float(row.get(P["destination_geo_lat"], np.nan))
        d_lng = float(row.get(P["destination_geo_lng"], np.nan))
        if any(np.isnan([o_lat, o_lng, d_lat, d_lng])):
            return np.nan
        return geodesic((o_lat, o_lng), (d_lat, d_lng)).miles
    except:
        return np.nan


def save_manifest_type(df, label, specific_cols, compiled_data_dir, suffix=""):
    """Save processed manifest CSV with correct column ordering."""
    param_order = PARAMETERS_DF["parameter_name"].tolist()
    type_qty_cols = [
        c for c in param_order
        if c in specific_cols[label.lower()] and c in df.columns and not c.startswith("Method Used")
    ]
    estimated_cols = [c for c in df.columns if c.startswith("Number of")]
    extra_cols = ["origin_dest_miles"] if "origin_dest_miles" in df.columns else []
    
    cols = []
    for c in METADATA_COLS + [P["origin_dairy_name"], P["origin_dairy_address"], 
                               P["origin_geo_lat"], P["origin_geo_lng"],
                               P["destination_address_final"], P["destination_address_final_source"],
                               P["destination_type_std"], P["destination_geo_lat"], P["destination_geo_lng"],
                               P["haul_date_first"], P["haul_date_last"],
                               P["is_pipeline"], P["is_trucked"]] + type_qty_cols + estimated_cols + extra_cols:
        if c in df.columns and c not in cols:
            cols.append(c)
    
    filename = f"processed_{label.lower()}_manifests{suffix}.csv"
    df[cols].to_csv(os.path.join(compiled_data_dir, filename), index=False)
    print(f"Saved {len(df)} rows to {filename}")



def main():
    # Merge and process manifests
    df = merge_manifests()
    df = geocode_manifests(df)
    
    # Enrich addresses
    enrich_address_columns(df, P["origin_dairy_address"], prefix="Origin ", county_col_in="County")
    enrich_address_columns(df, P["destination_address"], prefix="Destination ")
    
    # Backfill and standardize
    df = backfill_columns(df)
    
    # Standardize destination type
    def std_dest_type(val):
        if pd.isna(val) or not str(val).strip():
            return "Blank"       
        vl = str(val).lower()
        matched = []
        for canonical, keywords in DEST_TYPE_MAP.items():
            if any(kw in vl for kw in keywords):
                matched.append(canonical)
        if not matched:
            return "Other"
        if len(matched) == 1:
            return matched[0]
        return ", ".join(sorted(set(matched)))
    
    df[P["destination_type_std"]] = df[P["destination_type"]].apply(std_dest_type)
    
    # Split by manifest type
    manure_cols = [c for c in df.columns if c not in SPECIFIC_COLS["wastewater"]]
    wastewater_cols = [c for c in df.columns if c not in SPECIFIC_COLS["manure"]]
    
    df_manure = df.loc[df["Manifest Type"].isin(["manure", "both"]), manure_cols].copy()
    df_manure[P["is_trucked"]] = True
    df_ww = df.loc[df["Manifest Type"].isin(["wastewater", "both"]), wastewater_cols].copy()
    
    print(f"\nManure + both: {len(df_manure)} rows")
    print(f"Wastewater + both: {len(df_ww)} rows")
    
    # Calculate distance for wastewater
    df_ww["origin_dest_miles"] = df_ww.apply(calculate_distance, axis=1)

    is_trucked = df_ww[P["is_trucked"]] == True if P["is_trucked"] in df_ww.columns else pd.Series([False]*len(df_ww))
    df_ww_trucked = df_ww[is_trucked].copy()

    # Add haul estimates
    add_haul_estimates(df_manure, "Manure", P["manure_ton_per_haul"],
                      P["manure_number_hauls"], P["manure_amount"])
    add_haul_estimates(df_ww, "Wastewater", P["wastewater_gallon_per_haul"],
                      P["wastewater_number_hauls"], P["wastewater_amount"])
    add_haul_estimates(df_ww_trucked, "Wastewater", P["wastewater_gallon_per_haul"],
                      P["wastewater_number_hauls"], P["wastewater_amount"])

    # Print unique wastewater methods
    methods = df_ww[P["wastewater_method"]].dropna().unique()
    print(f"\nUnique 'Method Used for Analysis' values in wastewater manifests ({len(methods)}):")
    for m in methods:
        print(f"  {m}")

    # Save compiled_data
    save_manifest_type(df_manure, "Manure", SPECIFIC_COLS, OUTPUTS_DIR)
    save_manifest_type(df_ww, "Wastewater", SPECIFIC_COLS, OUTPUTS_DIR)


if __name__ == "__main__":
    main()