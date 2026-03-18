import os
import re
import numpy as np
import pandas as pd

from helpers_geocoding import enrich_address_columns, geocode_address, geocode_parcel, norm_addr
from helpers_pdf_metrics import PARAMETERS_DF, GDRIVE_BASE, build_parameter_dicts, coerce_columns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MANUAL_PATH = os.path.join(OUTPUTS_DIR, "all_manifests_as_written_validated.csv")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "all_manifests_as_written_automatic.csv")

# Column name mapping: parameter_key -> display name (e.g. P["origin_geo_lat"])
P = build_parameter_dicts(manifest_only=True)["key_to_name"]

# Columns exclusive to each manifest type (for splitting outputs)
specific_cols = {
    t: set(PARAMETERS_DF.loc[PARAMETERS_DF["manifest_type"] == t, "parameter_name"])
    for t in ["wastewater", "manure"]
}

METADATA_COLS = [
    "Source PDF",
    "Manifest Number",
    "Start Page",
    "End Page",
]

COLS_TO_KEEP = METADATA_COLS + [
    P["origin_dairy_name"],
    P["origin_dairy_address"],
    P["origin_geo_lat"],
    P["origin_geo_lng"],
    P["destination_address_final"],
    P["destination_address_final_source"],
    P["destination_type_std"],
    P["destination_geo_lat"],
    P["destination_geo_lng"],
    P["haul_date_first"],
    P["haul_date_last"],
    P["is_pipeline"],
    P["is_trucked"],
]

DEST_PRIORITY = [
    P["destination_parcel_number"],
    P["destination_nearest_cross_street"],
    P["destination_address"],
    P["destination_contact_address"],
    P["hauler_address"],
]


DEST_TYPE_MAP = {
    "Composting Facility": ["compost", "kellogg", "hyponex", "fertilizer", "supply"],
    "Farmer": ["farm"],
}

CA_MAP_LAYOUT = dict(
    # map_style="carto-positron",
    map_center={"lat": 37.2719, "lon": -119.2702},
    map_zoom=5,
)

_COORD_RE = re.compile(r"\s*\(?\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)?\s*$")


def geocode_if_valid(addr, geocode_fn, **kwargs):
    if not isinstance(addr, str) or not addr.strip() or pd.isna(addr):
        return None
    res = geocode_fn(addr, **kwargs)
    if not isinstance(res, (tuple, list)) or len(res) < 2:
        return None
    lat, lng = res[0], res[1]
    return (lat, lng) if (lat is not None and lng is not None) else None


def weighted_avg(df, val_col, weight_col):
    valid = df.dropna(subset=[val_col, weight_col])
    return (valid[val_col] * valid[weight_col]).sum() / valid[weight_col].sum()


def main():

    # Load and merge manual + extracted manifests
    manual_df = pd.read_csv(MANUAL_PATH, engine="python", on_bad_lines="warn")
    extracted_df = pd.read_csv(EXTRACTED_PATH)
    coerce_columns(manual_df)

    dupe_mask = manual_df.get("Is Duplicate", pd.Series()) == "x"
    if dupe_mask.any():
        manual_df = manual_df[~dupe_mask].reset_index(drop=True)

    key_cols = ["Source PDF", "Manifest Number"]
    cols_to_add = set(extracted_df.columns) - set(manual_df.columns) - set(key_cols)
    print(f"\nColumns to add from extracted_manifests: {sorted(cols_to_add)}")

    extracted_deduped = extracted_df.drop_duplicates(subset=key_cols, keep="first")
    merged = manual_df.merge(
        extracted_deduped[list(cols_to_add) + key_cols],
        on=key_cols,
        how="left",
        suffixes=("", "_ext"),
    )
    for col in cols_to_add:
        ext_col = f"{col}_ext"
        if ext_col in merged.columns:
            manual_df[col] = merged[col].combine_first(merged[ext_col])
        elif col in merged.columns:
            manual_df[col] = merged[col]

    # Drop rows that are EXACT duplicates across all columns except Manifest Number
    dup_subset = [c for c in manual_df.columns if c not in ["Manifest Number", "Start Page", "End Page"]]
    dupes = manual_df[manual_df.duplicated(subset=dup_subset, keep="first")]
    # for _, r in dupes[["Source PDF", "Manifest Number"]].iterrows():
    #     print(f" Duplicate Source PDF={r['Source PDF']}, Manifest {r['Manifest Number']}")
    manual_df = manual_df.drop_duplicates(subset=dup_subset)

    # Geocode origins and resolve destinations
    print("\nResolving Destination Address Final + Geocoding")
    manual_df[[P["destination_address_final"], P["destination_address_final_source"]]] = None

    latlong_col_keys = ["origin_geo_lat", "origin_geo_lng", "destination_geo_lat", "destination_geo_lng"]
    for latlong_col in latlong_col_keys:
        manual_df[P[latlong_col]] = None

    source_counts = {}
    n_origin_geo = n_dest_geo = 0

    for idx, row in manual_df.iterrows():
        addr = row[P["origin_dairy_address"]]
        county = row.get("County")
        if addr and (r := geocode_if_valid(addr, geocode_address, county=county)):
            manual_df.at[idx, P["origin_geo_lat"]] = r[0]
            manual_df.at[idx, P["origin_geo_lng"]] = r[1]
            n_origin_geo += 1

        raw_pc = row.get(P["destination_county"])
        parcel_county = str(raw_pc).strip() if raw_pc and pd.notna(raw_pc) else None

        # Existing manual lat/lng are highest-priority
        existing_lat = pd.to_numeric(row.get(P["latitude"]), errors="coerce")
        existing_lng = pd.to_numeric(row.get(P["longitude"]), errors="coerce")
        has_existing_coords = pd.notna(existing_lat) and pd.notna(existing_lng)

        val = src = None
        dest_geocoded = (existing_lat, existing_lng) if has_existing_coords else None

        # Track if any destination address field (parcel, cross street, address) was present
        dest_address_present = False

        # Try parcel number
        raw_parcel = row.get(P["destination_parcel_number"])
        if raw_parcel and not pd.isna(raw_parcel) and str(raw_parcel).strip():
            dest_address_present = True
            raw_str = str(raw_parcel).strip()
            if has_existing_coords:
                val, src = raw_str, P["destination_parcel_number"]
                dest_geocoded = (existing_lat, existing_lng)
            else:
                parts = [x.strip() for x in raw_str.split(",") if x.strip()]
                hits = [r for p in parts if (r := geocode_if_valid(p, geocode_parcel))]
                if hits:
                    val, src, dest_geocoded = raw_str, P["destination_parcel_number"], hits[0]

        # If parcel failed, try cross street (if it looks like coordinates)
        if not val:
            raw_cross = row.get(P["destination_nearest_cross_street"])
            if raw_cross and not pd.isna(raw_cross) and str(raw_cross).strip():
                dest_address_present = True
                raw_str = str(raw_cross).strip()
                m = _COORD_RE.match(raw_str)
                if m:
                    val, src = raw_str, P["destination_nearest_cross_street"]
                    if not has_existing_coords:
                        dest_geocoded = (float(m.group(1)), float(m.group(2)))

        # If still not found, try destination address (+cross street/county)
        if not val:
            raw_addr = row.get(P["destination_address"])
            if raw_addr and not pd.isna(raw_addr) and str(raw_addr).strip():
                dest_address_present = True
                raw_str = str(raw_addr).strip()
                cross = row.get(P["destination_nearest_cross_street"])
                if cross and pd.notna(cross) and str(cross).strip() and not _COORD_RE.match(str(cross)):
                    raw_str = f"{raw_str} {str(cross).strip()}"
                if parcel_county:
                    raw_str = f"{raw_str} {parcel_county}"
                val, src = raw_str, P["destination_address"]
                if not has_existing_coords:
                    dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)

        # Only if ALL destination address fields were empty, try contact address
        if not val and not dest_address_present:
            raw_contact = row.get(P["destination_contact_address"])
            if raw_contact and not pd.isna(raw_contact) and str(raw_contact).strip():
                raw_str = str(raw_contact).strip()
                if len(re.sub(r"[^a-zA-Z0-9]", "", raw_str)) >= 5:
                    if has_existing_coords:
                        val, src = raw_str, P["destination_contact_address"]
                        dest_geocoded = (existing_lat, existing_lng)
                    else:
                        g = geocode_if_valid(raw_str, geocode_address, county=parcel_county)
                        if g:
                            val, src, dest_geocoded = raw_str, P["destination_contact_address"], g

        # Only if ALL above are empty, try hauler address (must look like farm/compost/fertilizer)
        if not val and not dest_address_present:
            raw_hauler = row.get(P["hauler_address"])
            if raw_hauler and not pd.isna(raw_hauler) and str(raw_hauler).strip():
                hauler_combined = (
                    f"{str(row.get(P['hauler_name'], '') or '').lower()} {str(raw_hauler).lower()}"
                )
                if any(s in hauler_combined for s in ("farm", "compost", "fertilizer")):
                    raw_str = str(raw_hauler).strip()
                    val, src = raw_str, P["hauler_address"]
                    if not has_existing_coords:
                        dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)

        if val:
            manual_df.at[idx, P["destination_address_final"]] = val
            manual_df.at[idx, P["destination_address_final_source"]] = src
            source_counts[src] = source_counts.get(src, 0) + 1
        if dest_geocoded:
            manual_df.at[idx, P["destination_geo_lat"]] = dest_geocoded[0]
            manual_df.at[idx, P["destination_geo_lng"]] = dest_geocoded[1]
            n_dest_geo += 1

    # ...existing code...

    resolved = manual_df[P["destination_address_final"]].notna().sum()

    enrich_address_columns(
        manual_df, P["origin_dairy_address"], prefix="Origin ", county_col_in="County"
    )
    enrich_address_columns(manual_df, P["destination_address"], prefix="Destination ")

    # Backfill mass and solids columns
    BACKFILL_MASS_RULES = [
        (P["manure_amount"], P["manure_amount_yards"]),
        (P["manure_ton_per_haul"], P["manure_yard_per_haul"]),
    ]
    for mass_col, vol_col in BACKFILL_MASS_RULES:
        backfill = manual_df[mass_col].isna() & manual_df[vol_col].notna()
        manual_df.loc[backfill, mass_col] = (
            manual_df.loc[backfill, vol_col] * manual_df.loc[backfill, P["manure_density"]]
        )
        print(f"  Backfilled {backfill.sum()} mass values for {mass_col}")

    backfill_solids = (
        manual_df[P["manure_solids_percent"]].isna() & manual_df[P["manure_moisture_percent"]].notna()
    )
    manual_df.loc[backfill_solids, P["manure_solids_percent"]] = (
        1 - manual_df.loc[backfill_solids, P["manure_moisture_percent"]]
    )
    print(f"  Backfilled {backfill_solids.sum()} values for {P['manure_solids_percent']}")

    # Calculate avg manure density before dropping the column
    manure_density = manual_df.copy()[P["manure_density"]].dropna().astype(float)
    avg_manure_density = manure_density.mean()

    manual_df.drop(
        columns=[src for _, src in BACKFILL_MASS_RULES]
        + [P["manure_density"], P["manure_moisture_percent"]],
        errors="ignore",
        inplace=True,
    )

    # Compute destination_type_std from raw destination_type
    def std_dest_type(val):
        vl = str(val).lower()
        for canonical, keywords in DEST_TYPE_MAP.items():
            if any(kw in vl for kw in keywords):
                return canonical
        return "Other"

    manual_df[P["destination_type_std"]] = manual_df[P["destination_type"]].apply(std_dest_type)

    # Backfill missing origin dairy addresses from main report

    dairy_summary_df = pd.read_csv(
        os.path.join(GDRIVE_BASE, "Dairy_Report_Summary_Region_5_2024_with_source_pdf.csv")
    )
    origin_col = P["origin_dairy_address"]
    dairy_summary_df = dairy_summary_df.rename(columns={"Dairy Address": origin_col})
    dairy_summary_df["Source PDF"] = dairy_summary_df["Source PDF"].str.replace(
        r"\.pdf$", "", regex=True
    )

    dairy_name_to_addr = dairy_summary_df.drop_duplicates(subset="Dairy Name").set_index("Dairy Name")[
        origin_col
    ]

    needs_backfill = manual_df[origin_col].isna() | manual_df["Origin Dairy Latitude (Geocoded)"].isna()
    print(f"{needs_backfill.sum()} need origin dairy address backfill")
    num_before = manual_df.loc[needs_backfill, origin_col].isna().sum()

    still_missing = needs_backfill & manual_df[origin_col].isna()
    manual_df.loc[still_missing, origin_col] = manual_df.loc[still_missing, P["origin_dairy_name"]].map(
        dairy_name_to_addr
    )
    filled = num_before - manual_df.loc[needs_backfill, origin_col].isna().sum()
    print(f"Backfilled {filled} addresses")

    remaining = manual_df.loc[manual_df[origin_col].isna(), "Source PDF"].unique().tolist()
    print(f"Remaining rows with missing origin dairy address: {len(remaining)}")
    # for pdf in remaining:
    #     print(f"  {pdf}")

    # Extract address from Source PDF filename and geocode
    def addr_from_pdf_name(name):
        if re.match(r"^\d{4}[A-Z]", name):
            # "2024AR_Cream Top Dairy_13075 Ave 200_Tulare" → "13075 Ave 200 Tulare"
            m = re.search(r"Dairy[^_]*_(.+)", name)
            return m.group(1).replace("_", " ").strip() if m else None
        else:
            # "1007 S Hart Rd Modesto 2024 Dairy AR" → "1007 S Hart Rd Modesto"
            m = re.match(r"(.+?)\s+\d{4}", name)
            return m.group(1).strip() if m else None

    still_missing = manual_df[origin_col].isna()
    manual_df.loc[still_missing, origin_col] = manual_df.loc[still_missing, "Source PDF"].map(
        addr_from_pdf_name
    )
    newly_filled = still_missing & manual_df[origin_col].notna()
    lat_col, lng_col = P["origin_geo_lat"], P["origin_geo_lng"]
    for idx in manual_df.index[newly_filled]:
        addr = manual_df.at[idx, origin_col]
        county = manual_df.at[idx, "County"] if "County" in manual_df.columns else None
        if r := geocode_if_valid(addr, geocode_address, county=county):
            manual_df.at[idx, lat_col] = r[0]
            manual_df.at[idx, lng_col] = r[1]
    print(f"Geocoded {newly_filled.sum()} addresses from PDF filename")

    # Split by manifest type, compute stats, save CSVs
    manure_mask = manual_df["Manifest Type"].isin(["manure", "both"])
    manure_cols = [c for c in manual_df.columns if c not in specific_cols["wastewater"]]
    df_manure = manual_df.loc[manure_mask, manure_cols].copy()

    wastewater_mask = manual_df["Manifest Type"].isin(["wastewater", "both"])
    wastewater_cols = [c for c in manual_df.columns if c not in specific_cols["manure"]]
    df_ww = manual_df.loc[wastewater_mask, wastewater_cols].copy()

    print(f"  Manure + both: {len(df_manure)} rows")
    print(f"  Wastewater + both: {len(df_ww)} rows")

    ww_np = df_ww[df_ww[P["is_pipeline"]].ne(True)]  # non-pipeline only

    type_configs = [
        ("Manure", df_manure, P["manure_amount"], "tons"),
        ("Wastewater", df_ww, P["wastewater_amount"], "gallons"),
    ]
    # scale converts native rate units to tons/haul
    haul_cfg = [
        ("Manure", df_manure, P["manure_ton_per_haul"], P["manure_number_hauls"]),
        (
            "Wastewater",
            ww_np,
            P["wastewater_gallon_per_haul"],
            P["wastewater_number_hauls"],
        ),
    ]

    haul_stats = {}
    for label, df, rate_col, haul_col in haul_cfg:
        fac = (
            df.dropna(subset=[rate_col, haul_col])
            .groupby("Source PDF")
            .agg(avg_rate=(rate_col, "mean"), total_hauls=(haul_col, "sum"))
        )
        haul_stats[label] = dict(
            facility_hauls=fac,
            per_haul_series=fac["avg_rate"],
            avg_facility=(fac["avg_rate"]).mean(),
            avg_weighted=weighted_avg(df, rate_col, haul_col),
        )
    # Haul estimates: 10-ton / 20-ton bins for manure. 2,000 gal / 10,000 gal (cutoff 10k) for wastewater
    lo1, hi1, lo2, hi2, b1v, b2v, b1n, b2n = (5, 15, 15, 25, 10.0, 20.0, "10-ton", "20-ton")
    ww_gal_cutoff = 10000
    for (label, ref_df, rate_col, haul_col), (_, df, amount_col, unit) in zip(haul_cfg, type_configs):
        if label == "Wastewater":
            # Only calculate number of hauls for rows where is_trucked is True
            is_trucked = (
                ref_df[P["is_trucked"]] == True
                if P["is_trucked"] in ref_df.columns
                else pd.Series([True] * len(ref_df), index=ref_df.index)
            )
            rate_gal = ref_df.loc[is_trucked, P["wastewater_gallon_per_haul"]]
            amount_gal = ref_df.loc[is_trucked, P["wastewater_amount"]]
            haul_col_trucked = ref_df.loc[is_trucked, haul_col]

            mass_lo = amount_gal[rate_gal < ww_gal_cutoff].sum()
            mass_hi = amount_gal[rate_gal >= ww_gal_cutoff].sum()
            total = mass_lo + mass_hi
            p_lo = mass_lo / total if total > 0 else 0.5
            p_hi = 1.0 - p_lo
            print(f"Wastewater split: {p_lo:.1%} at <10,000 gal, {p_hi:.1%} at ≥10,000 gal")

            tons = amount_gal  # keep in gallons
            has_raw = rate_gal.notna() & haul_col_trucked.notna()
            est_lo = (tons * p_lo / ww_gal_cutoff).round().astype("Int64")
            est_hi = (tons * p_hi / ww_gal_cutoff).round().astype("Int64")

            # Get the indices in df that correspond to is_trucked True in ref_df
            trucked_idx = ref_df.index[is_trucked]

            df.loc[trucked_idx, "Estimated Number of <10,000 gal Hauls"] = est_lo.where(~has_raw).values
            df.loc[trucked_idx, "Estimated Number of ≥10,000 gal Hauls"] = est_hi.where(~has_raw).values

            # For analysis: actual hauls classified by bin if raw data present, else estimated
            rate_row = rate_gal
            n_hauls = pd.to_numeric(haul_col_trucked, errors="coerce").round().astype("Int64")
            in_lo = has_raw & (rate_row < ww_gal_cutoff)
            in_hi = has_raw & (rate_row >= ww_gal_cutoff)
            zero = pd.array([0] * len(rate_gal), dtype="Int64")
            df.loc[trucked_idx, "Number of <8,000 gal Hauls for Analysis"] = (
                est_lo.where(~in_lo, n_hauls).where(~in_hi, zero).values
            )
            df.loc[trucked_idx, "Number of >=8,000 gal Hauls for Analysis"] = (
                est_hi.where(~in_hi, n_hauls).where(~in_lo, zero).values
            )
        else:
            # Manure: keep as before
            rate_tons = ref_df[rate_col]
            amount_tons = ref_df[amount_col]
            mass_lo = amount_tons[rate_tons.between(lo1, hi1, inclusive="left")].sum()
            mass_hi = amount_tons[rate_tons.between(lo2, hi2, inclusive="left")].sum()
            total = mass_lo + mass_hi
            p_lo = mass_lo / total if total > 0 else 0.5
            p_hi = 1.0 - p_lo
            print(f"Manure split: {p_lo:.1%} at ~{b1n}, {p_hi:.1%} at ~{b2n}")

            tons = df[amount_col]
            has_raw = df[rate_col].notna() & df[haul_col].notna()
            est_lo = (tons * p_lo / b1v).round().astype("Int64")
            est_hi = (tons * p_hi / b2v).round().astype("Int64")

            # Estimated: only rows without real rate/haul data
            df[f"Estimated Number of {b1n} Hauls"] = est_lo.where(~has_raw)
            df[f"Estimated Number of {b2n} Hauls"] = est_hi.where(~has_raw)

            # For analysis: actual hauls classified by bin if raw data present, else estimated
            rate_row = df[rate_col]
            n_hauls = pd.to_numeric(df[haul_col], errors="coerce").round().astype("Int64")
            in_lo = has_raw & rate_row.between(lo1, hi1, inclusive="left")
            in_hi = has_raw & rate_row.between(lo2, hi2, inclusive="left")
            zero = pd.array([0] * len(df), dtype="Int64")
            df[f"Number of {b1n} Hauls for Analysis"] = est_lo.where(~in_lo, n_hauls).where(~in_hi, zero)
            df[f"Number of {b2n} Hauls for Analysis"] = est_hi.where(~in_hi, n_hauls).where(~in_lo, zero)
            print(f"Average manure haul: {haul_stats[label]['avg_weighted']:.2f} tons/haul")

    # print unique instances of "Method Used..." from the wastewater manifests
    methods = df_ww[P["wastewater_method"]].dropna().unique()
    print(f"\nUnique 'Method Used for Analysis' values in wastewater manifests ({len(methods)}):")
    for m in methods:
        print(f"  {m}")

    param_order = PARAMETERS_DF["parameter_name"].tolist()
    for label, df, amount_col, unit in type_configs:
        type_qty_cols = [
            c
            for c in param_order
            if c in specific_cols[label.lower()] and c in df.columns and not c.startswith("Method Used")
        ]
        estimated_cols = [c for c in df.columns if c.startswith("Number of")]
        cols = []
        for c in COLS_TO_KEEP + type_qty_cols + estimated_cols:
            if c in df.columns and c not in cols:
                cols.append(c)
        df[cols].to_csv(
            os.path.join(OUTPUTS_DIR, f"processed_{label.lower()}_manifests.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
