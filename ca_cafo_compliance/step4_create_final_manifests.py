import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helpers_geocoding import (
    enrich_address_columns,
    geocode_address,
    geocode_parcel,
    looks_like_parcel_number,
    parse_destination_address_and_parcel,
)
from helpers_pdf_metrics import (
    GDRIVE_BASE,
    PARAMETERS_DF,
    build_parameter_dicts,
    coerce_columns,
)
from helpers_plotting import MANIFEST_TYPE_COLORS, TYPE_COLOR_SEQ, manure_colors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MANUAL_PATH = os.path.join(OUTPUTS_DIR, "2024_manifests_manual.csv")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "2024_manifests_automatic.csv")

# Column name mapping: parameter_key -> display name (e.g. P["origin_geo_lat"])
P = build_parameter_dicts(manifest_only=True)["key_to_name"]

# Columns exclusive to each manifest type (for splitting outputs)
specific_cols = {
    t: set(PARAMETERS_DF.loc[PARAMETERS_DF["manifest_type"] == t, "parameter_name"])
    for t in ["wastewater", "manure"]
}

COLS_TO_GEOCODE = [P["origin_dairy_address"], P["destination_address"]]

COLS_TO_DROP = [
    "DONE",
    "Is Duplicate",
    "Street no",
    "Rest of PDF",
    "County",
    P["longitude"],
    P["latitude"],
    P["destination_type"],
    "Haul Date",
    "Unique Hauling Days Mentioned",
    "Parameter Template",
    "Method Used to Determine Volume of Manure",
    "Method Used to Determine Volume of Wastewater",
]

DEST_PRIORITY = [
    P["destination_parcel_number"],
    P["destination_nearest_cross_street"],
    P["destination_address"],
    P["destination_contact_address"],
    P["hauler_address"],
]

AMOUNT_COLS = [
    P["manure_amount"],
    P["wastewater_amount"],
    P["manure_ton_per_haul"],
    P["wastewater_gallon_per_haul"],
]

BACKFILL_MASS_RULES = [
    (P["manure_amount"], P["manure_amount_yards"]),
    (P["manure_ton_per_haul"], P["manure_yard_per_haul"]),
]

EARLY_DROP_COLS = [
    P["manure_amount_yards"],
    P["manure_yard_per_haul"],
    P["manure_density"],
    P["manure_moisture_percent"],
]

DEST_TYPE_MERGE_MAP = {
    "Composting Facility": ["Kelloggs", "Hyponex", "Fertilizer Company"],
    "Farmer": ["Spreader"],
    "Other": ["Garden", "Fenderup"],
}

CA_CENTER = {"lat": 37.2719, "lon": -119.2702}
CA_MAP_LAYOUT = dict(
    map_style="carto-positron",
    map_center=CA_CENTER,
    map_zoom=5,
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    height=800,
    width=1000,
)

_COORD_RE = re.compile(r"\s*\(?\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)?\s*$")


def _geocode_if_valid(addr, geocode_fn, **kwargs):
    if not isinstance(addr, str) or not addr.strip() or pd.isna(addr):
        return None
    res = geocode_fn(addr, **kwargs)
    if not isinstance(res, (tuple, list)) or len(res) < 2:
        return None
    lat, lng = res[0], res[1]
    return (lat, lng) if (lat is not None and lng is not None) else None


def _save_fig(fig, name):
    """Save a figure as PNG to outputs."""
    png_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
    with open(os.path.join(OUTPUTS_DIR, f"{name}.png"), "wb") as f:
        f.write(png_bytes)
    print(f"  Saved {name}")


def _parse_addr_from_pdf(pdf):
    if pd.isna(pdf):
        return None
    m = re.match(r"2024AR_.+?_(.+?)_(\w+)$", pdf)
    if m:
        return f"{m.group(1)}, {m.group(2)}"
    m = re.match(r"(.+?)\s+2024 Dairy AR$", pdf)
    return m.group(1) if m else None


# Load and merge manual + extracted manifests
manual_df = pd.read_csv(MANUAL_PATH, engine="python", on_bad_lines="warn")
extracted_df = pd.read_csv(EXTRACTED_PATH)
coerce_columns(manual_df)

done_count = (manual_df["DONE"] == "x").sum()
done_pct = 100 * done_count / len(manual_df) if len(manual_df) > 0 else 0
print(f"Rows marked DONE: {done_count}/{len(manual_df)} ({done_pct:.1f}%)")

dupe_mask = manual_df.get("Is Duplicate", pd.Series()) == "x"
n_dupes = dupe_mask.sum()
if n_dupes > 0:
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
duplicates_mask = manual_df.duplicated(subset=dup_subset, keep="first")
duplicates_df = manual_df.loc[duplicates_mask, ["Source PDF", "Manifest Number"]]

if not duplicates_df.empty:
    print("Dropping exact-duplicate rows")
    for _, r in duplicates_df.iterrows():
        print(f"  Source PDF={r['Source PDF']}, Manifest {r['Manifest Number']}")

manual_df = manual_df.drop_duplicates(subset=dup_subset)

# Geocode origins and resolve destinations
print("\nResolving Destination Address Final + Geocoding")
manual_df[[P["destination_address_final"], P["destination_address_final_source"]]] = None
manual_df[[P["origin_geo_lat"], P["origin_geo_lng"],
            P["destination_geo_lat"], P["destination_geo_lng"]]] = None
source_counts = {}
geo_counts = {col: 0 for col in COLS_TO_GEOCODE}

for idx, row in manual_df.iterrows():
    addr = row[P["origin_dairy_address"]]
    county = row.get("County")
    if addr and (r := _geocode_if_valid(addr, geocode_address, county=county)):
        manual_df.at[idx, P["origin_geo_lat"]] = r[0]
        manual_df.at[idx, P["origin_geo_lng"]] = r[1]
        geo_counts[P["origin_dairy_address"]] += 1

    parcel_county = row.get(P["destination_county"])
    parcel_county = (
        str(parcel_county).strip()
        if parcel_county and pd.notna(parcel_county)
        else None
    )

    # Existing manual lat/lng (are highest-priority
    existing_lat = pd.to_numeric(row.get(P["latitude"]), errors="coerce")
    existing_lng = pd.to_numeric(row.get(P["longitude"]), errors="coerce")
    has_existing_coords = pd.notna(existing_lat) and pd.notna(existing_lng)
    existing_coords = (existing_lat, existing_lng) if has_existing_coords else None

    val = src = None
    dest_geocoded = existing_coords
    for dest_col in DEST_PRIORITY:
        raw = row.get(dest_col)
        if not raw or pd.isna(raw):
            continue
        raw_str = str(raw).strip()
        if not raw_str:
            continue

        if dest_col == P["destination_parcel_number"]:
            if has_existing_coords:
                val, src = raw_str, dest_col
            else:
                parts = [x.strip() for x in raw_str.split(",") if x.strip()]
                hits = [r for p in parts if (r := _geocode_if_valid(p, geocode_parcel))]
                if hits:
                    val, src, dest_geocoded = raw_str, dest_col, hits[0]
                break

        elif dest_col == P["destination_nearest_cross_street"]:
            m = _COORD_RE.match(raw_str)
            if m:
                val, src = raw_str, dest_col
                if not has_existing_coords:
                    dest_geocoded = (float(m.group(1)), float(m.group(2)))
                break

        elif dest_col == P["destination_address"]:
            cross = row.get(P["destination_nearest_cross_street"])
            if cross and pd.notna(cross) and str(cross).strip() and not _COORD_RE.match(str(cross)):
                raw_str = f"{raw_str} {str(cross).strip()}"
            if parcel_county:
                raw_str = f"{raw_str} {parcel_county}"
            val, src = raw_str, dest_col
            if not has_existing_coords:
                dest_geocoded = _geocode_if_valid(raw_str, geocode_address, county=parcel_county)
            break

        elif dest_col == P["destination_contact_address"]:
            if len(re.sub(r"[^a-zA-Z0-9]", "", raw_str)) >= 5:
                if has_existing_coords:
                    val, src = raw_str, dest_col
                else:
                    g = _geocode_if_valid(raw_str, geocode_address, county=parcel_county)
                    if g:
                        val, src, dest_geocoded = raw_str, dest_col, g
                    break

        elif dest_col == P["hauler_address"]:
            strings_to_check = ["farm", "compost", "fertilizer"]
            # Only use hauler address if it looks like a farm/compost destination
            hauler_name_val = str(row.get(P["hauler_name"], "") or "").lower()
            combined = f"{hauler_name_val} {raw_str.lower()}"
            if not any(s in combined for s in strings_to_check):
                continue
            val, src = raw_str, dest_col
            if not has_existing_coords:
                dest_geocoded = _geocode_if_valid(raw_str, geocode_address, county=parcel_county)
            break

        else:
            val, src = raw_str, dest_col
            if not has_existing_coords:
                dest_geocoded = _geocode_if_valid(raw_str, geocode_address, county=parcel_county)
            break
    if val:
        manual_df.at[idx, P["destination_address_final"]] = val
        manual_df.at[idx, P["destination_address_final_source"]] = src
        source_counts[src] = source_counts.get(src, 0) + 1
    if dest_geocoded:
        manual_df.at[idx, P["destination_geo_lat"]] = dest_geocoded[0]
        manual_df.at[idx, P["destination_geo_lng"]] = dest_geocoded[1]
        geo_counts[P["destination_address"]] += 1

resolved = manual_df[P["destination_address_final"]].notna().sum()
print(f"  Resolved {resolved}/{len(manual_df)} destination addresses:")
for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"    {source}: {count}")
for col in COLS_TO_GEOCODE:
    print(f"{geo_counts[col]}/{len(manual_df)} {col}")

enrich_address_columns(manual_df, P["origin_dairy_address"], prefix="Origin ", county_col_in="County")
enrich_address_columns(manual_df, P["destination_address"], prefix="Destination ")

# Backfill mass and solids columns
for mass_col, vol_col in BACKFILL_MASS_RULES:
    backfill = manual_df[mass_col].isna() & manual_df[vol_col].notna()
    manual_df.loc[backfill, mass_col] = (
        manual_df.loc[backfill, vol_col] * manual_df.loc[backfill, P["manure_density"]]
    )
    print(f"  Backfilled {backfill.sum()} mass values for {mass_col}")

backfill_solids = (
    manual_df[P["manure_solids_percent"]].isna()
    & manual_df[P["manure_moisture_percent"]].notna()
)
manual_df.loc[backfill_solids, P["manure_solids_percent"]] = (
    1 - manual_df.loc[backfill_solids, P["manure_moisture_percent"]]
)
print(f"  Backfilled {backfill_solids.sum()} values for {P['manure_solids_percent']}")

manual_df = manual_df.drop(columns=[c for c in EARLY_DROP_COLS if c in manual_df.columns])

# Expand multi-parcel destinations into separate rows
new_rows = []
drop_idxs = []
for idx, row in manual_df.iterrows():
    apn_raw = row.get(P["destination_parcel_number"])
    if not apn_raw or pd.isna(apn_raw):
        continue
    parcels = []
    for p in (x.strip() for x in str(apn_raw).split(",") if x.strip()):
        _, parcel = parse_destination_address_and_parcel(p)
        if parcel and looks_like_parcel_number(parcel):
            parcels.append(parcel)
    if len(parcels) <= 1:
        continue

    n = len(parcels)
    base_manifest = str(row["Manifest Number"])
    for j, parcel in enumerate(parcels):
        new_row = row.copy()
        new_row["Manifest Number"] = f"{base_manifest}p{j+1}"
        new_row[P["destination_parcel_number"]] = parcel
        for col in AMOUNT_COLS:
            val = pd.to_numeric(new_row.get(col), errors="coerce")
            if pd.notna(val):
                new_row[col] = val / n
        r = _geocode_if_valid(parcel, geocode_parcel)
        if r:
            new_row[P["destination_geo_lat"]] = r[0]
            new_row[P["destination_geo_lng"]] = r[1]
        new_rows.append(new_row)
    drop_idxs.append(idx)

if new_rows:
    manual_df = manual_df.drop(drop_idxs)
    manual_df = pd.concat([manual_df, pd.DataFrame(new_rows)], ignore_index=True)
    print(f"  Expanded {len(drop_idxs)} multi-parcel rows into {len(new_rows)} rows")

#Standardize destination types
for new_type, old_types in DEST_TYPE_MERGE_MAP.items():
    manual_df[P["destination_type_std"]] = manual_df[P["destination_type_std"]].replace(
        {k: new_type for k in old_types}
    )

# Backfill missing origin dairy addresses
dairy_summary_df = pd.read_csv(
    "ca_cafo_compliance/local/Dairy_Data_and_Analysis/Data/Summary/"
    "Dairy_Report_Summary_Region_5_2024_pdf_merged.csv"
)
origin_col = P["origin_dairy_address"]
dairy_summary_df = dairy_summary_df.rename(columns={"Dairy Address": origin_col})
dairy_summary_df["Source PDF"] = dairy_summary_df["Source PDF"].str.replace(
    r"\.pdf$", "", regex=True
)

additional_origins_df = pd.read_csv(
    os.path.join(OUTPUTS_DIR, "2024_manifests_additional_origins.csv")
)

backfill_sources = [
    (
        "Source PDF (summary)",
        "Source PDF",
        dairy_summary_df.drop_duplicates(subset="Source PDF")
        .set_index("Source PDF")[origin_col],
    ),
    (
        "Origin Dairy Name (summary)",
        P["origin_dairy_name"],
        dairy_summary_df.drop_duplicates(subset="Dairy Name")
        .set_index("Dairy Name")[origin_col],
    ),
    (
        "additional origins CSV",
        "Source PDF",
        additional_origins_df.set_index("Source PDF")[origin_col],
    ),
    ("Source PDF filename", "Source PDF", _parse_addr_from_pdf),
]

needs_backfill = manual_df[origin_col].isna() | manual_df["Origin Dairy Latitude (Geocoded)"].isna()
print(f"{needs_backfill.sum()} need origin dairy address backfill")
num_before = manual_df.loc[needs_backfill, origin_col].isna().sum()

for label, col, source in backfill_sources:
    still_missing = needs_backfill & manual_df[origin_col].isna()
    if not still_missing.any():
        break
    before = still_missing.sum()
    if callable(source):
        manual_df.loc[still_missing, origin_col] = manual_df.loc[still_missing, col].apply(source)
    else:
        manual_df.loc[still_missing, origin_col] = manual_df.loc[still_missing, col].map(source)
    filled = before - (needs_backfill & manual_df[origin_col].isna()).sum()
    if filled:
        print(f"  {label}: {filled}")

print(f"Backfilled {num_before - manual_df.loc[needs_backfill, origin_col].isna().sum()} rows")

remaining = manual_df.loc[manual_df[origin_col].isna(), "Source PDF"].unique().tolist()
print(f"Remaining rows with missing origin dairy address: {len(remaining)}")
for pdf in remaining:
    print(f"  {pdf}")

#Split by manifest type, compute stats, save CSVs
manure_mask = manual_df["Manifest Type"].isin(["manure", "both"])
manure_cols = [c for c in manual_df.columns if c not in specific_cols["wastewater"]]
df_manure = manual_df.loc[manure_mask, manure_cols].copy()

wastewater_mask = manual_df["Manifest Type"].isin(["wastewater", "both"])
wastewater_cols = [c for c in manual_df.columns if c not in specific_cols["manure"]]
df_ww = manual_df.loc[wastewater_mask, wastewater_cols].copy()

print(f"  Manure + both: {len(df_manure)} rows")
print(f"  Wastewater + both: {len(df_ww)} rows")

# Per-facility averages and total hauls (for top-row histograms and overlays)
manure_facility_hauls = (
    df_manure.dropna(subset=[P["manure_ton_per_haul"], P["manure_number_hauls"]])
    .groupby("Source PDF")
    .agg(
        avg_tons_per_haul=(P["manure_ton_per_haul"], "mean"),
        total_hauls=(P["manure_number_hauls"], "sum"),
    )
)

ww_facility_hauls = (
    df_ww.dropna(subset=[P["wastewater_gallon_per_haul"], P["wastewater_number_hauls"]])
    .groupby("Source PDF")
    .agg(
        avg_gal_per_haul=(P["wastewater_gallon_per_haul"], "mean"),
        total_hauls=(P["wastewater_number_hauls"], "sum"),
    )
)

tons_per_haul_facility = manure_facility_hauls["avg_tons_per_haul"]
ww_per_haul_facility = ww_facility_hauls["avg_gal_per_haul"]

# Averages across facilities (simple mean of facility averages)
avg_tons_per_haul_facility = tons_per_haul_facility.mean()
avg_gal_per_haul_facility = ww_per_haul_facility.mean()

# Averages across hauls (weighted by number of hauls per manifest)
manure_valid = df_manure.dropna(
    subset=[P["manure_ton_per_haul"], P["manure_number_hauls"]]
).copy()
ww_valid = df_ww.dropna(
    subset=[P["wastewater_gallon_per_haul"], P["wastewater_number_hauls"]]
).copy()

avg_tons_per_haul_weighted = (
    (manure_valid[P["manure_ton_per_haul"]] * manure_valid[P["manure_number_hauls"]])
    .sum()
    / manure_valid[P["manure_number_hauls"]].sum()
    if len(manure_valid) > 0
    else float("nan")
)

avg_gal_per_haul_weighted = (
    (ww_valid[P["wastewater_gallon_per_haul"]] * ww_valid[P["wastewater_number_hauls"]])
    .sum()
    / ww_valid[P["wastewater_number_hauls"]].sum()
    if len(ww_valid) > 0
    else float("nan")
)

# Global average tons per haul (still used for estimated hauls)
avg_tons_per_haul = avg_tons_per_haul_weighted

# Simple-average based estimate (status quo)
df_manure["Estimated Number of Hauls (Based on Average)"] = (
    (df_manure[P["manure_amount"]] / avg_tons_per_haul)
    .round()
    .astype("Int64")
)

# Distribution-based estimates: split ALL manure into 10-ton and 20-ton
# hauls using the observed ratio of mass in the 5-15 vs 15-25 ton/haul bins.
tph = df_manure[P["manure_ton_per_haul"]]

mask_5_15 = tph.between(5, 15, inclusive="left")
mask_15_25 = tph.between(15, 25, inclusive="left")

mass_5_15 = df_manure.loc[mask_5_15, P["manure_amount"]].sum(skipna=True)
mass_15_25 = df_manure.loc[mask_15_25, P["manure_amount"]].sum(skipna=True)

# Normalize so p10 + p20 = 1.0 — all mass is allocated
mass_in_bins = mass_5_15 + mass_15_25
p10 = mass_5_15 / mass_in_bins if mass_in_bins > 0 else 0.5
p20 = 1.0 - p10

print(f"Manure distribution split: {p10:.1%} at ~10 ton, {p20:.1%} at ~20 ton")

# N10 = amount * p10 / 10,  N20 = amount * p20 / 20
# Mass check: N10*10 + N20*20 = amount*p10 + amount*p20 = amount
df_manure["Estimated Number of 10-ton Hauls (Based on Distribution)"] = (
    df_manure[P["manure_amount"]] * p10 / 10.0
).round().astype("Int64")

df_manure["Estimated Number of 20-ton Hauls (Based on Distribution)"] = (
    df_manure[P["manure_amount"]] * p20 / 20.0
).round().astype("Int64")

print(f"Average: {avg_tons_per_haul:.2f} tons/haul")
print(f"  Estimated hauls: {df_manure['Estimated Number of Hauls (Based on Average)'].sum():.0f}")
print(f"  Total tons: {df_manure[P['manure_amount']].sum():.0f}")

# Analogous estimates for wastewater
avg_gal_per_haul = avg_gal_per_haul_weighted

df_ww["Estimated Number of Hauls (Based on Average)"] = (
    (df_ww[P["wastewater_amount"]] / avg_gal_per_haul)
    .round()
    .astype("Int64")
)

wph = df_ww[P["wastewater_gallon_per_haul"]]

mask_5k_15k = wph.between(5_000, 15_000, inclusive="left")
mask_15k_25k = wph.between(15_000, 25_000, inclusive="left")

vol_5k_15k = df_ww.loc[mask_5k_15k, P["wastewater_amount"]].sum(skipna=True)
vol_15k_25k = df_ww.loc[mask_15k_25k, P["wastewater_amount"]].sum(skipna=True)

# Normalize so wp10 + wp20 = 1.0
vol_in_bins = vol_5k_15k + vol_15k_25k
wp10 = vol_5k_15k / vol_in_bins if vol_in_bins > 0 else 0.5
wp20 = 1.0 - wp10

print(f"Wastewater distribution split: {wp10:.1%} at ~10k gal, {wp20:.1%} at ~20k gal")

# N10k = amount * wp10 / 10000,  N20k = amount * wp20 / 20000
df_ww["Estimated Number of 10k-gallon Hauls (Based on Distribution)"] = (
    df_ww[P["wastewater_amount"]] * wp10 / 10_000.0
).round().astype("Int64")

df_ww["Estimated Number of 20k-gallon Hauls (Based on Distribution)"] = (
    df_ww[P["wastewater_amount"]] * wp20 / 20_000.0
).round().astype("Int64")

print(f"Average wastewater haul: {avg_gal_per_haul:.0f} gallons/haul")
print(
    f"  Estimated hauls (wastewater, simple average): "
    f"{df_ww['Estimated Number of Hauls (Based on Average)'].sum():.0f}"
)
print(f"  Total gallons: {df_ww[P['wastewater_amount']].sum():.0f}")

# Tons-per-haul & facility-level scatter subplot (manure + wastewater)

# Aggregate exports by facility (Source PDF), normalizing split manifests like '1p1'
def _facility_agg(df, amount_col):
    tmp = df.copy()
    tmp["_manifest_norm"] = tmp["Manifest Number"].astype(str).str.replace(
        r"p\d+$", "", regex=True
    )
    per_manifest = (
        tmp.groupby(["Source PDF", "_manifest_norm"])[amount_col]
        .sum()
        .reset_index()
    )
    summary = (
        per_manifest.groupby("Source PDF")
        .agg(
            total_amount=(amount_col, "sum"),
            manifest_count=("_manifest_norm", "nunique"),
        )
        .reset_index()
    )
    return summary

manure_facility = _facility_agg(df_manure, P["manure_amount"])
ww_facility = _facility_agg(df_ww, P["wastewater_amount"])

print("\nTop 5 facilities by total manure exported (tons):")
print(
    manure_facility.sort_values("total_amount", ascending=False)
    .head(5)[["Source PDF", "total_amount", "manifest_count"]]
)

print("\nTop 5 facilities by total wastewater exported (gallons):")
print(
    ww_facility.sort_values("total_amount", ascending=False)
    .head(5)[["Source PDF", "total_amount", "manifest_count"]]
)

fig_hauls = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"secondary_y": True}, {"secondary_y": True}], [{}, {}]],
)

def _binned_totals(x_vals, totals, nbins):
    if len(x_vals) == 0:
        return [], [], None

    # Explicit left-closed, right-open bins over a slightly expanded range
    x_min = float(x_vals.min())
    x_max = float(x_vals.max())
    span = x_max - x_min if x_max > x_min else 1.0
    margin = span * 1e-6
    bin_edges = np.linspace(x_min, x_max + margin, nbins + 1)

    counts = pd.cut(
        x_vals,
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        right=False,
    )

    bin_totals = []
    bin_centers = []
    for i in range(len(bin_edges) - 1):
        mask = counts == i
        if not mask.any():
            continue
        bin_totals.append(totals[mask].sum())
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
    return bin_centers, bin_totals, bin_edges

manure_bin_x, manure_bin_totals, manure_edges = _binned_totals(
    manure_facility_hauls["avg_tons_per_haul"],
    manure_facility_hauls["total_hauls"],
    nbins=10,
)
ww_bin_x, ww_bin_totals, ww_edges = _binned_totals(
    ww_facility_hauls["avg_gal_per_haul"],
    ww_facility_hauls["total_hauls"],
    nbins=10,
)

print("Manure bins (center, total hauls):", list(zip(manure_bin_x, manure_bin_totals)))
print("Wastewater bins (center, total hauls):", list(zip(ww_bin_x, ww_bin_totals)))

manure_max = max(manure_bin_totals) if manure_bin_totals else 0
ww_max = max(ww_bin_totals) if ww_bin_totals else 0

fig_hauls.update_yaxes(
    range=[0, manure_max * 1.1],
    row=1, col=1, secondary_y=True,
)
fig_hauls.update_yaxes(
    range=[0, ww_max],
    row=1, col=2, secondary_y=True,
)

# Using the same bin edges as the histograms
manure_counts, _ = np.histogram(tons_per_haul_facility, bins=manure_edges)
ww_counts, _ = np.histogram(ww_per_haul_facility, bins=ww_edges)

# Increase y-lim for the histogram subplots (facility count)
fig_hauls.update_yaxes(
    range=[0, manure_counts.max() * 1.1],
    row=1, col=1, secondary_y=False,
)
fig_hauls.update_yaxes(
    range=[0, ww_counts.max() * 1.1],
    row=1, col=2, secondary_y=False,
)

# Top row: facility-count histograms (primary y), forced to share bin edges with dots
if manure_edges is not None:
    manure_xbins = dict(
        start=float(manure_edges[0]),
        end=float(manure_edges[-1]),
        size=float(manure_edges[1] - manure_edges[0]),
    )
else:
    manure_xbins = None

if ww_edges is not None:
    ww_xbins = dict(
        start=float(ww_edges[0]),
        end=float(ww_edges[-1]),
        size=float(ww_edges[1] - ww_edges[0]),
    )
else:
    ww_xbins = None

fig_hauls.add_trace(
    go.Histogram(
        x=tons_per_haul_facility,
        xbins=manure_xbins,
        marker_color=manure_colors[0],
        name="Manure tons/haul (facilities)",
        showlegend=False,
    ),
    row=1,
    col=1,
    secondary_y=False,
)

fig_hauls.add_trace(
    go.Histogram(
        x=ww_per_haul_facility,
        xbins=ww_xbins,
        marker_color=MANIFEST_TYPE_COLORS.get("wastewater", "#1f77b4"),
        name="Wastewater gallons/haul (facilities)",
        showlegend=False,
    ),
    row=1,
    col=2,
    secondary_y=False,
)

# Overlay one dot per bin for total hauls (secondary y)

fig_hauls.add_trace(
    go.Scatter(
        x=manure_bin_x,
        y=manure_bin_totals,
        mode="markers",
        marker=dict(color="black", size=8, opacity=0.9, symbol="circle"),
        name="Manure total hauls (per bin)",
        showlegend=False,
    ),
    row=1,
    col=1,
    secondary_y=True,
)

fig_hauls.add_trace(
    go.Scatter(
        x=ww_bin_x,
        y=ww_bin_totals,
        mode="markers",
        marker=dict(
            color="black",
            size=8,
            opacity=0.9,
            symbol="circle",
        ),
        name="Wastewater total hauls (per bin)",
        showlegend=False,
    ),
    row=1,
    col=2,
    secondary_y=True,
)

# Add vertical lines for facility-mean and haul-weighted-mean
fig_hauls.add_vline(
    x=avg_tons_per_haul_facility,
    line_color="black",
    line_width=2,
    row=1,
    col=1,
    annotation_text=round(avg_tons_per_haul_facility, 1),
    annotation_position="top left",
)
fig_hauls.add_vline(
    x=avg_tons_per_haul_weighted,
    line_dash="dot",
    line_color="black",
    line_width=2,
    row=1,
    col=1,
    annotation_text=round(avg_tons_per_haul_weighted, 1),
    annotation_position="top right",
)

fig_hauls.add_vline(
    x=avg_gal_per_haul_facility,
    line_color="black",
    line_width=2,
    row=1,
    col=2,
    annotation_text=int(avg_gal_per_haul_facility),
    annotation_position="top right",
)
fig_hauls.add_vline(
    x=avg_gal_per_haul_weighted,
    line_dash="dot",
    line_color="black",
    line_width=2,
    row=1,
    col=2,
    annotation_text=int(avg_gal_per_haul_weighted),
    annotation_position="top left",
)

# Legend-only entries for shapes (no color encoding)
fig_hauls.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(color="black", size=10, symbol="square"),
        name="Facility Count",
        showlegend=True,
    )
)
fig_hauls.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(color="black", size=8, symbol="circle"),
        name="Total Hauls in Bin",
        showlegend=True,
    )
)
fig_hauls.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="black", width=2),
        name="Average by Facility",
        showlegend=True,
    )
)
fig_hauls.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="black", dash="dot", width=2),
        name="Average by Hauls",
        showlegend=True,
    )
)
fig_hauls.add_vline(
    x=avg_gal_per_haul_weighted,
    line_dash="dot",
    line_color="black",
    line_width=2,
    row=1,
    col=2,
)

# Bottom row: facility-level scatter plots
fig_hauls.add_trace(
    go.Scatter(
        x=manure_facility["total_amount"],
        y=manure_facility["manifest_count"],
        mode="markers",
        marker=dict(color=manure_colors[0]),
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig_hauls.add_trace(
    go.Scatter(
        x=ww_facility["total_amount"],
        y=ww_facility["manifest_count"],
        mode="markers",
        marker=dict(color=MANIFEST_TYPE_COLORS.get("wastewater", "#1f77b4")),
        showlegend=False,
    ),
    row=2,
    col=2,
)

fig_hauls.update_layout(
    showlegend=True,
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1000,
    height=700,
    font=dict(size=16),
    legend=dict(
        x=1.1,          # push legend to the right of plotting area
        y=1.0,           # align to top
        xanchor="left",  # anchor left edge of legend at x
        yanchor="top",   # anchor top of legend at y
        bordercolor="white",
        borderwidth=0,
    ),
)
fig_hauls.update_xaxes(title_text="Tons per haul", row=1, col=1)
fig_hauls.update_yaxes(title_text="Number of facilities", row=1, col=1, secondary_y=False)
fig_hauls.update_yaxes(title_text="Total hauls", row=1, col=1, secondary_y=True)
fig_hauls.update_xaxes(title_text="Gallons per haul", row=1, col=2)
fig_hauls.update_yaxes(title_text="Number of facilities", row=1, col=2, secondary_y=False)
fig_hauls.update_yaxes(title_text="Total hauls", row=1, col=2, secondary_y=True)
fig_hauls.update_xaxes(title_text="Total facility Exports (tons) in 2024", row=2, col=1)
fig_hauls.update_yaxes(title_text="Manifests per Facility", row=2, col=1)
fig_hauls.update_xaxes(title_text="Total Facility Exports (gallons) in 2024", row=2, col=2)
fig_hauls.update_yaxes(title_text="Manifests Per Facility", row=2, col=2)

# Add a small box around each individual subplot
fig_hauls.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
fig_hauls.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

# Column headers for manure / wastewater
fig_hauls.add_annotation(
    text="Manure",
    x=0,
    y=1.08,
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(size=16),
)
fig_hauls.add_annotation(
    text="Wastewater",
    x=0.6,
    y=1.08,
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(size=16),
)

_save_fig(fig_hauls, "2024_tons_per_haul")

_type_configs = [
    ("Manure", df_manure, P["manure_amount"], "tons"),
    ("Wastewater", df_ww, P["wastewater_amount"], "gallons"),
]

print("\nTemplates breakdown:")
print(manual_df["Parameter Template"].value_counts())

for label, df, amount_col, unit in _type_configs:
    print(f"\n{label} summary by destination type:")
    print(df.groupby(P["destination_type_std"])[amount_col].sum())
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    # Re-order columns to follow parameters.csv order where applicable
    param_order = PARAMETERS_DF["parameter_name"].tolist()
    ordered_param_cols = [c for c in param_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_param_cols]
    df = df[ordered_param_cols + remaining_cols]
    fname = f"2024_manifests_{label.lower()}.csv"
    df.to_csv(os.path.join(OUTPUTS_DIR, fname), index=False)

#Interactive maps
_map_configs = [
    (P["origin_dairy_address"], P["origin_geo_lat"], P["origin_geo_lng"]),
    (P["destination_address"], P["destination_geo_lat"], P["destination_geo_lng"]),
]
for col, lat_c, lng_c in _map_configs:
    has_geo = manual_df[lat_c].notna() & manual_df[lng_c].notna()
    if not has_geo.any():
        continue
    subset = manual_df.loc[has_geo].copy()
    subset["Geocoded Text"] = subset.get(
        P["destination_address_final"] if "Dest" in col else col, ""
    )
    subset["Address Source"] = subset.get(P["destination_address_final_source"], col)

    fig = px.scatter_map(
        subset,
        lat=lat_c,
        lon=lng_c,
        color="Manifest Type",
        color_discrete_map=MANIFEST_TYPE_COLORS,
        hover_name=P["origin_dairy_name"],
        hover_data={
            "Source PDF": True,
            P["destination_name"]: True,
            "Manifest Number": True,
            "Address Source": True,
            "Geocoded Text": True,
            "Manifest Type": False,
            lat_c: False,
            lng_c: False,
        },
        title=col,
    )
    fig.update_layout(**CA_MAP_LAYOUT)
    filename = f"2024_{col.lower().replace(' ', '_')}_map.html"
    fig.write_html(os.path.join(OUTPUTS_DIR, filename))
    print(f"  Saved {col} map")

#Combined 2x2 subplot: pie charts + monthly bar charts
month_labels = [
    pd.Timestamp(month=m, day=1, year=2024).strftime("%b") for m in range(1, 13)
]

fig_combined = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"type": "pie"}, {"type": "pie"}], [{"type": "bar"}, {"type": "bar"}]],
    subplot_titles=[f"{l} Destination Types" for l, *_ in _type_configs]
    + [f"{l} Hauls by Month" for l, *_ in _type_configs],
)

for col_idx, (label, df, amount_col, unit) in enumerate(_type_configs, start=1):
    colors = TYPE_COLOR_SEQ[label]

    # Build destination-type counts, splitting comma-separated types like
    # "Composting Facility, Farmer" and allocating equal weight to each.
    type_series = df[P["destination_type_std"]].dropna().astype(str)
    type_weights = {}
    for v in type_series:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        if not parts:
            continue
        w = 1.0 / len(parts)
        for p in parts:
            type_weights[p] = type_weights.get(p, 0.0) + w

    if type_weights:
        type_counts = pd.Series(type_weights).sort_values(ascending=False)
    else:
        type_counts = pd.Series([], dtype=float)

    # Reposition labels for Wastewater pie to reduce overlap
    if label == "Wastewater":
        pulls = [
            0.2 if name != "Farmer" else 0.0
            for name in type_counts.index
        ]
        pie = go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            marker_colors=colors,
            textposition="outside",
            textinfo="label+percent",
            textfont=dict(size=12),
            pull=pulls,
            rotation=90,
        )
    else:
        pie = go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            marker_colors=colors,
            textposition="auto",
            textinfo="label+percent",
            insidetextorientation="radial",
            pull=[
                0.05 if v / type_counts.sum() < 0.05 else 0
                for v in type_counts.values
            ],
        )

    fig_combined.add_trace(pie, row=1, col=col_idx)

    monthly_amount = pd.Series(0.0, index=range(1, 13))
    for _, row in df.iterrows():
        amt = pd.to_numeric(row.get(amount_col), errors="coerce")
        if pd.isna(amt):
            continue
        first = pd.to_datetime(
            row.get(P["haul_date_first"]), format="mixed", dayfirst=False, errors="coerce"
        )
        last = pd.to_datetime(
            row.get(P["haul_date_last"]), format="mixed", dayfirst=False, errors="coerce"
        )
        if pd.isna(last) and pd.isna(first):
            continue
        if pd.isna(first):
            first = last
        if first.month == 1 and last.month == 12:
            continue
        lo, hi = min(first.month, last.month), max(first.month, last.month)
        span = list(range(lo, hi + 1))
        weight = 1.0 / len(span)
        for mo in span:
            monthly_amount[mo] += amt * weight

    total = monthly_amount.sum()
    if total > 0:
        monthly_amount = monthly_amount / total

    fig_combined.add_trace(
        go.Bar(
            x=month_labels,
            y=monthly_amount.values,
            marker_color=colors[0],
            showlegend=False
        ),
        row=2,
        col=col_idx,
    )

    # Add a small box around each individual subplot
# Apply to all 4 subplots
for col_idx in [1, 2]:
    fig_combined.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True, row=1, col=col_idx)
    fig_combined.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True, row=1, col=col_idx)
    fig_combined.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True, row=2, col=col_idx)
    fig_combined.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True, row=2, col=col_idx)

fig_combined.update_layout(
    height=500,
    width=900,
    showlegend=False,
    margin=dict(t=40, b=30, l=40, r=20),
    plot_bgcolor="white",
)
for col_idx in [1, 2]:
    fig_combined.update_yaxes(range=[0, 0.15], dtick=0.05, row=2, col=col_idx)
_save_fig(fig_combined, "2024_manifest_summary")
