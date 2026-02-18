#!/usr/bin/env python3
"""
Create 2024_manifests_manure.csv and 2024_manifests_wastewater.csv.

1. Start with 2024_manifests_manual.csv
2. Fill in columns NOT present in manifests_manual from 2024_manifests_raw.csv
3. Re-geocode dairy address, hauler address, and destination address
4. Split into manure (type=manure or both) and wastewater (type=wastewater or both)
"""

import os
import re
import pandas as pd
import plotly.express as px

from step2b_extract_manifest_parameters import to_numeric
from helpers_geocoding import (
    geocode_address,
    geocode_parcel,
    load_geocoding_cache,
    save_geocoding_cache,
)

GDRIVE_BASE = "/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/ca_cafo_manifests"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MANUAL_PATH = os.path.join(OUTPUTS_DIR, "2024_manifests_manual.csv")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "2024_manifests_raw.csv")
PARAMETERS_PATH = os.path.join(BASE_DIR, "data", "parameters.csv")

# Build sets of columns to EXCLUDE from each output based on parameters.csv manifest_type.
# Wastewater-only columns are excluded from manure output, and vice versa.
_params_df = pd.read_csv(PARAMETERS_PATH)
_WASTEWATER_ONLY_COLS = set(
    _params_df.loc[_params_df["manifest_type"] == "wastewater", "parameter_name"]
)
_MANURE_ONLY_COLS = set(
    _params_df.loc[_params_df["manifest_type"] == "manure", "parameter_name"]
)

# Geocoding column names
ORIGIN_GEOCODED_COL = "Origin Dairy Address (Geocoded)"
HAULER_GEOCODED_COL = "Hauler Address (Geocoded)"
DEST_GEOCODED_COL = "Destination Address (Geocoded)"

# Columns to exclude from filling (geocoding columns)
GEOCODING_COLS = {ORIGIN_GEOCODED_COL, HAULER_GEOCODED_COL, DEST_GEOCODED_COL}

# California bounding box
CA_LAT_MIN = 32.5
CA_LAT_MAX = 42.0
CA_LON_MIN = -124.5
CA_LON_MAX = -114.0


def _parse_geocoded(geocoded_str):
    """Parse geocoded string into list of (lat, lon) tuples.

    Handles single coords '(lat, lon)' and multi-parcel
    '[(lat1, lon1), (lat2, lon2)]'. Always returns a list.
    """
    if not geocoded_str or pd.isna(geocoded_str):
        return []
    s = str(geocoded_str).strip()
    # Multi-parcel: "[(lat, lon), (lat, lon), etc]
    if s.startswith("["):
        import ast
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [(float(lat), float(lon)) for lat, lon in parsed]
        except (ValueError, TypeError, SyntaxError):
            pass
        return []
    # Single: "(lat, lon)"
    try:
        s = s.strip("() ")
        parts = s.split(",")
        if len(parts) >= 2:
            return [(float(parts[0].strip()), float(parts[1].strip()))]
    except (ValueError, TypeError):
        pass
    return []


def _is_in_california(lat, lon):
    """Check if coordinates are within California bounding box."""
    if lat is None or lon is None:
        return True  # No geocoding = not flagged as out-of-CA
    return CA_LAT_MIN <= lat <= CA_LAT_MAX and CA_LON_MIN <= lon <= CA_LON_MAX


def _geocode_if_valid(addr, geocode_fn, cache):
    """Geocode address and return result if both coordinates are non-null."""
    if not addr or pd.isna(addr):
        return None
    geocoded = geocode_fn(addr, cache)
    return geocoded if geocoded and geocoded[0] is not None and geocoded[1] is not None else None


def _is_po_box(addr):
    """Check if address is a PO Box (not a physical location)."""
    if not addr or pd.isna(addr):
        return False
    return bool(re.search(r"\bP\.?O\.?\s*Box\b", str(addr), re.IGNORECASE))


def _is_valid_address(addr):
    """Check if address is non-empty, non-null, and not a PO Box."""
    if not addr or pd.isna(addr):
        return False
    return bool(str(addr).strip()) and not _is_po_box(addr)


DEST_FINAL_COL = "Destination Address Final"
DEST_FINAL_SOURCE_COL = "Destination Address Final Source"


def _get_month_weights(first_date_str, last_date_str):
    """Return {month_number: weight} distributing evenly across touched months.

    If a manifest spans e.g. March–May, returns {3: 1/3, 4: 1/3, 5: 1/3}.
    Single-date manifests return full weight on that month.
    """
    last = pd.to_datetime(last_date_str, format="mixed", dayfirst=False, errors="coerce")
    if pd.isna(last):
        return {}

    first = pd.to_datetime(first_date_str, format="mixed", dayfirst=False, errors="coerce")
    if pd.isna(first):
        return {last.month: 1.0}

    lo, hi = min(first.month, last.month), max(first.month, last.month)
    months = list(range(lo, hi + 1))
    weight = 1.0 / len(months)
    return {mo: weight for mo in months}


def _save_fig(fig, name):
    """Save a figure as PNG to both outputs and Google Drive."""
    png_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
    for dest in [OUTPUTS_DIR, GDRIVE_BASE + "/figures"]:
        with open(os.path.join(dest, f"{name}.png"), "wb") as f:
            f.write(png_bytes)
    print(f"  Saved {name}")


def _resolve_destination_address(row, cache):
    """Resolve the best available destination address using cascading fallback.

    Priority order:
      1. Assessor Parcel Number (most accurate; comma-separated → list of coords)
      2. Destination Address (if not a PO Box)
      3. Destination Contact Address (if not a PO Box)
      4. Destination Name (if geocodable, meaning it contains a real address)
      5. Hauler Address (last resort)

    Returns (address, source) tuple. For parcels, address may be a string
    representation of multiple parcel numbers.
    """
    # 1. Assessor Parcel Number (most accurate)
    dest_parcel = row.get("Destination Assessor Parcel Number")
    if dest_parcel and pd.notna(dest_parcel):
        parcels = [p.strip() for p in str(dest_parcel).split(",") if p.strip()]
        results = [_geocode_if_valid(p, geocode_parcel, cache) for p in parcels]
        results = [r for r in results if r]
        if results:
            return str(dest_parcel).strip(), "Assessor Parcel Number"

    # 2. Destination Address (+ cross street if available)
    dest_addr = row.get("Destination Address")
    dest_cross = row.get("Destination Nearest Cross Street")
    if _is_valid_address(dest_addr):
        addr = str(dest_addr).strip()
        if dest_cross and pd.notna(dest_cross) and str(dest_cross).strip():
            addr = f"{addr} {str(dest_cross).strip()}"
        return addr, "Destination Address"

    # 3. Destination Contact Address
    contact_addr = row.get("Destination Contact Address")
    if _is_valid_address(contact_addr):
        return str(contact_addr).strip(), "Destination Contact Address"

    # 4. Destination Name — check if it geocodes to a valid location
    #    Skip names that are too short or lack real address content (e.g. "Ln —")
    dest_name = row.get("Destination Name")
    if dest_name and pd.notna(dest_name):
        name_clean = str(dest_name).strip()
        alphanumeric = re.sub(r"[^a-zA-Z0-9]", "", name_clean)
        if len(alphanumeric) >= 5:
            result = _geocode_if_valid(name_clean, geocode_address, cache)
            if result:
                return name_clean, "Destination Name"

    # 5. Hauler Address (last resort)
    hauler_addr = row.get("Hauler Address")
    if _is_valid_address(hauler_addr):
        return str(hauler_addr).strip(), "Hauler Address"

    return None, None


def main():
    cache = load_geocoding_cache()

    # Load both files
    manual_df = pd.read_csv(MANUAL_PATH)
    extracted_df = pd.read_csv(EXTRACTED_PATH)

    print(f"Loaded manifests_manual: {len(manual_df)} rows")
    print(f"Loaded extracted_manifests: {len(extracted_df)} rows")

    # Print DONE statistics
    done_count = (manual_df["DONE"] == "x").sum()
    done_pct = 100 * done_count / len(manual_df) if len(manual_df) > 0 else 0
    print(f"Rows marked DONE: {done_count}/{len(manual_df)} ({done_pct:.1f}%)")

    # Key columns for matching
    key_cols = ["Source PDF", "Manifest Number"]

    # Find columns in extracted that are NOT in manual (excluding geocoding)
    manual_cols = set(manual_df.columns)
    extracted_cols = set(extracted_df.columns)
    cols_to_add = extracted_cols - manual_cols - GEOCODING_COLS - set(key_cols)

    print(f"\nColumns to add from extracted_manifests: {sorted(cols_to_add)}")

    # Add missing columns to manual_df
    for col in cols_to_add:
        manual_df[col] = None

    # Create lookup from extracted_manifests
    extracted_df["_key"] = (
        extracted_df["Source PDF"].astype(str) + "_" + extracted_df["Manifest Number"].astype(str)
    )
    extracted_lookup = extracted_df.set_index("_key")

    # Fill in missing column values from extracted
    filled_count = 0
    for idx, row in manual_df.iterrows():
        key = f"{row['Source PDF']}_{row['Manifest Number']}"

        if key in extracted_lookup.index:
            extracted_row = extracted_lookup.loc[key]
            if isinstance(extracted_row, pd.DataFrame):
                extracted_row = extracted_row.iloc[0]

            for col in cols_to_add:
                if col in extracted_row.index and pd.notna(extracted_row[col]):
                    manual_df.at[idx, col] = extracted_row[col]
            filled_count += 1

    print(f"Filled columns for {filled_count} rows")

    # Resolve Destination Address Final using cascading fallback
    print("\nResolving Destination Address Final")
    manual_df[DEST_FINAL_COL] = None
    manual_df[DEST_FINAL_SOURCE_COL] = None

    source_counts = {}
    for idx, row in manual_df.iterrows():
        addr, source = _resolve_destination_address(row, cache)
        if addr:
            manual_df.at[idx, DEST_FINAL_COL] = addr
            manual_df.at[idx, DEST_FINAL_SOURCE_COL] = source
            source_counts[source] = source_counts.get(source, 0) + 1

    resolved = manual_df[DEST_FINAL_COL].notna().sum()
    print(f"  Resolved {resolved}/{len(manual_df)} destination addresses:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {source}: {count}")

    # Add geocoding columns
    manual_df[ORIGIN_GEOCODED_COL] = None
    manual_df[HAULER_GEOCODED_COL] = None
    manual_df[DEST_GEOCODED_COL] = None

    # Geocode addresses
    print("\nGeocoding addresses")
    geocoded_origin = 0
    geocoded_hauler = 0
    geocoded_dest = 0

    for idx, row in manual_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing row {idx}/{len(manual_df)}")

        # Geocode origin dairy address
        origin_addr = row.get("Origin Dairy Address")
        if origin_addr and pd.notna(origin_addr):
            result = _geocode_if_valid(origin_addr, geocode_address, cache)
            if result:
                manual_df.at[idx, ORIGIN_GEOCODED_COL] = str(result)
                geocoded_origin += 1

        # Geocode hauler address
        hauler_addr = row.get("Hauler Address")
        if hauler_addr and pd.notna(hauler_addr):
            result = _geocode_if_valid(hauler_addr, geocode_address, cache)
            if result:
                manual_df.at[idx, HAULER_GEOCODED_COL] = str(result)
                geocoded_hauler += 1

        # Geocode destination: try parcel first, then Destination Address Final
        dest_final = row.get(DEST_FINAL_COL)
        dest_parcel = row.get("Destination Assessor Parcel Number")

        dest_geocoded = None

        # Try parcel first (most accurate); handle comma-separated parcels
        if dest_parcel and pd.notna(dest_parcel):
            parcels = [p.strip() for p in str(dest_parcel).split(",") if p.strip()]
            if len(parcels) > 1:
                results = []
                for p in parcels:
                    r = _geocode_if_valid(p, geocode_parcel, cache)
                    if r:
                        results.append(r)
                if results:
                    dest_geocoded = results if len(results) > 1 else results[0]
            else:
                dest_geocoded = _geocode_if_valid(dest_parcel, geocode_parcel, cache)

        # If no parcel geocoding, use Destination Address Final (already includes cross street)
        if not dest_geocoded and dest_final and pd.notna(dest_final):
            dest_geocoded = _geocode_if_valid(dest_final, geocode_address, cache)

        if dest_geocoded:
            manual_df.at[idx, DEST_GEOCODED_COL] = str(dest_geocoded)
            geocoded_dest += 1

    print(f"  Origin {geocoded_origin}/{len(manual_df)} addresses")
    print(f"  Hauler {geocoded_hauler}/{len(manual_df)} addresses")
    print(f"  Destination {geocoded_dest}/{len(manual_df)} addresses")

    # Save geocoding cache
    save_geocoding_cache(cache)

    # Check for out-of-CA geocoding
    out_of_ca_rows = []
    for idx, row in manual_df.iterrows():
        reasons = []

        for lat, lon in _parse_geocoded(row.get(ORIGIN_GEOCODED_COL)):
            if not _is_in_california(lat, lon):
                reasons.append(f"Origin: ({lat:.4f}, {lon:.4f})")

        for lat, lon in _parse_geocoded(row.get(DEST_GEOCODED_COL)):
            if not _is_in_california(lat, lon):
                reasons.append(f"Destination: ({lat:.4f}, {lon:.4f})")

        if reasons:
            row_copy = row.copy()
            row_copy["Out of CA Reason"] = "; ".join(reasons)
            out_of_ca_rows.append(row_copy)

    if out_of_ca_rows:
        df_out_of_ca = pd.DataFrame(out_of_ca_rows)
        out_of_ca_path = os.path.join(OUTPUTS_DIR, "2024_manifests_out_of_ca.csv")
        df_out_of_ca.to_csv(out_of_ca_path, index=False)
        df_out_of_ca.to_csv(os.path.join(GDRIVE_BASE, "2024_manifests_out_of_ca.csv"), index=False)

    # Split by manifest type and drop irrelevant columns
    manifest_type_col = "Manifest Type"
    manure_mask = manual_df[manifest_type_col].isin(["manure", "both"])
    manure_cols = [c for c in manual_df.columns if c not in _WASTEWATER_ONLY_COLS]
    df_manure = manual_df.loc[manure_mask, manure_cols].copy()

    wastewater_mask = manual_df[manifest_type_col].isin(["wastewater", "both"])
    wastewater_cols = [c for c in manual_df.columns if c not in _MANURE_ONLY_COLS]
    df_wastewater = manual_df.loc[wastewater_mask, wastewater_cols].copy()

    print(f"  Manure (manure + both): {len(df_manure)} rows, {len(manure_cols)} cols (excluded {len(_WASTEWATER_ONLY_COLS)} wastewater-only cols)")
    print(f"  Wastewater (wastewater + both): {len(df_wastewater)} rows, {len(wastewater_cols)} cols (excluded {len(_MANURE_ONLY_COLS)} manure-only cols)")

    # Summary statistics
    manure_col = "Total Manure Amount (tons)"
    wastewater_col = "Total Process Wastewater Exports (Gallons)"
    dest_type_col = "Destination Type (Standardized)"

    print("\nManure summary by destination type:")
    print(df_manure.groupby(dest_type_col)[manure_col].apply(lambda x: to_numeric(x).sum()))

    print("\nWastewater summary by destination type:")
    print(
        df_wastewater.groupby(dest_type_col)[wastewater_col].apply(lambda x: to_numeric(x).sum())
    )

    print("\nTemplates breakdown:")
    print(manual_df["Parameter Template"].value_counts())

    # Save output files
    manure_path = os.path.join(OUTPUTS_DIR, "2024_manifests_manure.csv")
    wastewater_path = os.path.join(OUTPUTS_DIR, "2024_manifests_wastewater.csv")

    df_manure.to_csv(manure_path, index=False)
    df_wastewater.to_csv(wastewater_path, index=False)

    # Also save to Google Drive
    df_manure.to_csv(os.path.join(GDRIVE_BASE, "2024_manifests_manure.csv"), index=False)
    df_wastewater.to_csv(os.path.join(GDRIVE_BASE, "2024_manifests_wastewater.csv"), index=False)

    print("Saved")

    dairy_rows = []
    dest_rows = []

    for idx, row in manual_df.iterrows():
        # Parse dairy address coordinates
        for dairy_lat, dairy_lon in _parse_geocoded(row.get(ORIGIN_GEOCODED_COL)):
            if _is_in_california(dairy_lat, dairy_lon):
                dairy_rows.append(
                    {
                        "Latitude": dairy_lat,
                        "Longitude": dairy_lon,
                        "Dairy Name": row.get("Origin Dairy Name", "Unknown"),
                        "Address": row.get("Origin Dairy Address", ""),
                        "Manifest Type": row.get("Manifest Type", "unknown"),
                    }
                )

        # Parse destination address coordinates (may have multiple parcels)
        for dest_lat, dest_lon in _parse_geocoded(row.get(DEST_GEOCODED_COL)):
            if _is_in_california(dest_lat, dest_lon):
                dest_rows.append(
                    {
                        "Latitude": dest_lat,
                        "Longitude": dest_lon,
                        "Destination Name": row.get("Destination Name", "Unknown"),
                        "Address": row.get("Destination Address", ""),
                        "Destination Type": row.get("Destination Type (Standardized)", "unknown"),
                        "Manifest Type": row.get("Manifest Type", "unknown"),
                    }
                )

    # California center and zoom
    ca_center = {"lat": 37.2719, "lon": -119.2702}

    # Create dairy address map using scatter_geo (works with kaleido PNG)
    if dairy_rows:
        dairy_df = pd.DataFrame(dairy_rows)
        dairy_fig = px.scatter_geo(
            dairy_df,
            lat="Latitude",
            lon="Longitude",
            color="Manifest Type",
            hover_name="Dairy Name",
            hover_data={"Address": True, "Manifest Type": True},
            title="Origin Dairy Addresses",
            scope="usa",
        )
        dairy_fig.update_geos(
            center=ca_center,
            projection_scale=4,
            lataxis_range=[CA_LAT_MIN, CA_LAT_MAX],
            lonaxis_range=[CA_LON_MIN, CA_LON_MAX],
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showcoastlines=True,
        )
        dairy_fig.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            height=800,
            width=1000,
        )

        # Save dairy map (HTML only — kaleido hangs on large scatter_geo PNGs)
        dairy_fig.write_html(os.path.join(OUTPUTS_DIR, "2024_dairy_addresses_map.html"))
        dairy_fig.write_html(os.path.join(GDRIVE_BASE + "/figures", "2024_dairy_addresses_map.html"))
        print("  Saved dairy addresses map")

    # Create destination address map using scatter_geo
    if dest_rows:
        dest_df = pd.DataFrame(dest_rows)
        dest_fig = px.scatter_geo(
            dest_df,
            lat="Latitude",
            lon="Longitude",
            color="Destination Type",
            hover_name="Destination Name",
            hover_data={
                "Address": True,
                "Destination Type": True,
                "Manifest Type": True,
            },
            title="Destination Addresses",
            scope="usa",
        )
        dest_fig.update_geos(
            center=ca_center,
            projection_scale=4,
            lataxis_range=[CA_LAT_MIN, CA_LAT_MAX],
            lonaxis_range=[CA_LON_MIN, CA_LON_MAX],
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showcoastlines=True,
        )
        dest_fig.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            height=800,
            width=1000,
        )

        # Save destination map
        dest_map_path = os.path.join(OUTPUTS_DIR, "2024_destination_addresses_map.html")
        dest_fig.write_html(dest_map_path)
        dest_fig.write_html(os.path.join(GDRIVE_BASE + "/figures", "2024_destination_addresses_map.html"))

    #  Pie charts: destination type breakdown 
    print("\nGenerating charts")

    for label, df, amount_col in [
        ("Manure", df_manure, "Total Manure Amount (tons)"),
        ("Wastewater", df_wastewater, "Total Process Wastewater Exports (Gallons)"),
    ]:
        type_counts = df[dest_type_col].value_counts()
        fig = px.pie(
            names=type_counts.index,
            values=type_counts.values,
        )
        _save_fig(fig, f"2024_{label.lower()}_destination_type_pie")

    #  Monthly charts: hauls per month, manure tons/month, wastewater gallons/month 
    first_col = "First Haul Date"
    last_col = "Last Haul Date"

    # Build monthly distributions for manure and wastewater
    for label, df, amount_col, unit in [
        ("Manure", df_manure, "Total Manure Amount (tons)", "tons"),
        ("Wastewater", df_wastewater, "Total Process Wastewater Exports (Gallons)", "gallons"),
    ]:
        monthly_hauls = pd.Series(0.0, index=range(1, 13))
        monthly_amount = pd.Series(0.0, index=range(1, 13))

        for _, row in df.iterrows():
            weights = _get_month_weights(row.get(first_col), row.get(last_col))
            for mo, w in weights.items():
                monthly_hauls[mo] += w
                amt = to_numeric(pd.Series([row.get(amount_col)])).iloc[0]
                if pd.notna(amt):
                    monthly_amount[mo] += amt * w

        month_labels = pd.to_datetime(monthly_hauls.index, format="%m").strftime("%b")

        # Hauls histogram
        fig = px.bar(
            x=month_labels, y=monthly_hauls.values,
            labels={"x": "Month", "y": "Number of Hauls"},
        )
        _save_fig(fig, f"2024_{label.lower()}_hauls_per_month")

        # Amount bar chart
        fig = px.bar(
            x=month_labels, y=monthly_amount.values,
            labels={"x": "Month", "y": f"Total {label} ({unit})"},
        )
        _save_fig(fig, f"2024_{label.lower()}_amount_per_month")


if __name__ == "__main__":
    main()
