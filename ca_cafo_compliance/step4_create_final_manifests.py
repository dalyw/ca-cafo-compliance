#!/usr/bin/env python3
"""
Create 2024_manifests_manure.csv and 2024_manifests_wastewater.csv.

1. Start with manifests_manual.csv
2. Fill in columns NOT present in manifests_manual from extracted_manifests.csv (except geocoding)
3. Re-geocode dairy address, hauler address, and destination address
4. Split into manure (type=manure or both) and wastewater (type=wastewater or both)
"""

import os
import pandas as pd
import plotly.express as px

from ca_cafo_compliance.step2b_extract_manifest_parameters import to_numeric
from ca_cafo_compliance.helpers_geocoding import (
    geocode_address,
    geocode_parcel,
    load_geocoding_cache,
    save_geocoding_cache,
)

GDRIVE_BASE = "/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/ca_cafo_manifests"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MANUAL_PATH = os.path.join(OUTPUTS_DIR, "manifests_manual.csv")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "extracted_manifests.csv")

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
    """Parse geocoded string like '(lat, lon)' into (lat, lon) tuple."""
    if not geocoded_str or pd.isna(geocoded_str):
        return None, None
    try:
        # Handle string format like "(37.285831, -120.596449)"
        s = str(geocoded_str).strip("() ")
        parts = s.split(",")
        if len(parts) >= 2:
            return float(parts[0].strip()), float(parts[1].strip())
    except (ValueError, TypeError):
        pass
    return None, None


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
    return (
        geocoded
        if geocoded and geocoded[0] is not None and geocoded[1] is not None
        else None
    )


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
        extracted_df["Source PDF"].astype(str)
        + "_"
        + extracted_df["Manifest Number"].astype(str)
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

    # Add geocoding columns
    manual_df[ORIGIN_GEOCODED_COL] = None
    manual_df[HAULER_GEOCODED_COL] = None
    manual_df[DEST_GEOCODED_COL] = None

    # Geocode addresses
    print("\nGeocoding addresses...")
    geocoded_origin = 0
    geocoded_hauler = 0
    geocoded_dest = 0

    for idx, row in manual_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing row {idx}/{len(manual_df)}...")

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

        # Geocode destination: try parcel first, then address + cross street
        dest_addr = row.get("Destination Address")
        dest_cross = row.get("Destination Nearest Cross Street")
        dest_parcel = row.get("Destination Assessor Parcel Number")

        dest_geocoded = None

        # Try parcel first (most accurate)
        if dest_parcel and pd.notna(dest_parcel):
            dest_geocoded = _geocode_if_valid(dest_parcel, geocode_parcel, cache)

        # If no parcel geocoding, try address + cross street
        if not dest_geocoded and dest_addr and pd.notna(dest_addr):
            addr_to_geocode = dest_addr
            if dest_cross and pd.notna(dest_cross) and str(dest_cross).strip():
                addr_to_geocode = f"{dest_addr} {dest_cross}"
            dest_geocoded = _geocode_if_valid(addr_to_geocode, geocode_address, cache)

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

        origin_lat, origin_lon = _parse_geocoded(row.get(ORIGIN_GEOCODED_COL))
        if origin_lat is not None and not _is_in_california(origin_lat, origin_lon):
            reasons.append(f"Origin: ({origin_lat:.4f}, {origin_lon:.4f})")

        dest_lat, dest_lon = _parse_geocoded(row.get(DEST_GEOCODED_COL))
        if dest_lat is not None and not _is_in_california(dest_lat, dest_lon):
            reasons.append(f"Destination: ({dest_lat:.4f}, {dest_lon:.4f})")

        if reasons:
            row_copy = row.copy()
            row_copy["Out of CA Reason"] = "; ".join(reasons)
            out_of_ca_rows.append(row_copy)

    if out_of_ca_rows:
        df_out_of_ca = pd.DataFrame(out_of_ca_rows)
        out_of_ca_path = os.path.join(OUTPUTS_DIR, "2024_manifests_out_of_ca.csv")
        df_out_of_ca.to_csv(out_of_ca_path, index=False)
        df_out_of_ca.to_csv(
            os.path.join(GDRIVE_BASE, "2024_manifests_out_of_ca.csv"), index=False
        )

    # Split by manifest type
    manifest_type_col = "Manifest Type"
    manure_mask = manual_df[manifest_type_col].isin(["manure", "both"])
    df_manure = manual_df[manure_mask].copy()
    wastewater_mask = manual_df[manifest_type_col].isin(["wastewater", "both"])
    df_wastewater = manual_df[wastewater_mask].copy()

    print(f"  Manure (manure + both): {len(df_manure)}")
    print(f"  Wastewater (wastewater + both): {len(df_wastewater)}")

    # Summary statistics
    manure_col = "Total Manure Amount (tons)"
    wastewater_col = "Total Process Wastewater Exports (Gallons)"
    dest_type_col = "Destination Type (Standardized)"

    print("\nManure summary by destination type:")
    print(
        df_manure.groupby(dest_type_col)[manure_col].apply(
            lambda x: to_numeric(x).sum()
        )
    )

    print("\nWastewater summary by destination type:")
    print(
        df_wastewater.groupby(dest_type_col)[wastewater_col].apply(
            lambda x: to_numeric(x).sum()
        )
    )

    print("\nTemplates breakdown:")
    print(manual_df["Parameter Template"].value_counts())

    # Save output files
    manure_path = os.path.join(OUTPUTS_DIR, "2024_manifests_manure.csv")
    wastewater_path = os.path.join(OUTPUTS_DIR, "2024_manifests_wastewater.csv")

    df_manure.to_csv(manure_path, index=False)
    df_wastewater.to_csv(wastewater_path, index=False)

    # Also save to Google Drive
    df_manure.to_csv(
        os.path.join(GDRIVE_BASE, "2024_manifests_manure.csv"), index=False
    )
    df_wastewater.to_csv(
        os.path.join(GDRIVE_BASE, "2024_manifests_wastewater.csv"), index=False
    )

    print("Saved")

    dairy_rows = []
    dest_rows = []

    for idx, row in manual_df.iterrows():
        # Parse dairy address coordinates
        dairy_lat, dairy_lon = _parse_geocoded(row.get(ORIGIN_GEOCODED_COL))
        if dairy_lat is not None and _is_in_california(dairy_lat, dairy_lon):
            dairy_rows.append(
                {
                    "Latitude": dairy_lat,
                    "Longitude": dairy_lon,
                    "Dairy Name": row.get("Origin Dairy Name", "Unknown"),
                    "Address": row.get("Origin Dairy Address", ""),
                    "Manifest Type": row.get("Manifest Type", "unknown"),
                }
            )

        # Parse destination address coordinates
        dest_lat, dest_lon = _parse_geocoded(row.get(DEST_GEOCODED_COL))
        if dest_lat is not None and _is_in_california(dest_lat, dest_lon):
            dest_rows.append(
                {
                    "Latitude": dest_lat,
                    "Longitude": dest_lon,
                    "Destination Name": row.get("Destination Name", "Unknown"),
                    "Address": row.get("Destination Address", ""),
                    "Destination Type": row.get(
                        "Destination Type (Standardized)", "unknown"
                    ),
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
            title="2024 Manifest Origin Dairy Addresses",
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

        # Save dairy map
        dairy_map_path = os.path.join(OUTPUTS_DIR, "2024_dairy_addresses_map.html")
        dairy_fig.write_html(dairy_map_path)
        dairy_fig.write_html(os.path.join(GDRIVE_BASE, "2024_dairy_addresses_map.html"))
        dairy_png_path = os.path.join(OUTPUTS_DIR, "2024_dairy_addresses_map.png")
        dairy_fig.write_image(dairy_png_path, width=1200, height=900, scale=2)
        dairy_fig.write_image(
            os.path.join(GDRIVE_BASE, "2024_dairy_addresses_map.png"),
            width=1200,
            height=900,
            scale=2,
        )
        print(f"  Dairy addresses map: {dairy_png_path}")

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
            title="2024 Manifest Destination Addresses",
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
        dest_fig.write_html(
            os.path.join(GDRIVE_BASE, "2024_destination_addresses_map.html")
        )
        dest_png_path = os.path.join(OUTPUTS_DIR, "2024_destination_addresses_map.png")
        dest_fig.write_image(dest_png_path, width=1200, height=900, scale=2)
        dest_fig.write_image(
            os.path.join(GDRIVE_BASE, "2024_destination_addresses_map.png"),
            width=1200,
            height=900,
            scale=2,
        )
        print(f"  Destination addresses map: {dest_png_path}")


if __name__ == "__main__":
    main()
