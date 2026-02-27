import os
import re
import pandas as pd
import plotly.express as px

from helpers_geocoding import (
    enrich_address_columns,
    geocode_address,
    geocode_parcel,
    looks_like_parcel_number,
    parse_destination_address_and_parcel,
)
from helpers_pdf_metrics import coerce_numeric_columns

GDRIVE_BASE = "/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/ca_cafo_manifests"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MANUAL_PATH = os.path.join(OUTPUTS_DIR, "2024_manifests_manual.csv")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "2024_manifests_automatic.csv")
PARAMETERS_PATH = os.path.join(BASE_DIR, "data", "parameters.csv")

# Columns to EXCLUDE from each output: wastewater-only cols excluded from manure, vice versa.
_params_df = pd.read_csv(PARAMETERS_PATH)
specific_cols = {
    t: set(_params_df.loc[_params_df["manifest_type"] == t, "parameter_name"])
    for t in ["wastewater", "manure"]
}


cols_to_geocode = ["Origin Dairy Address", "Destination Address"]
cols_to_drop = [
    "DONE",
    "Is Duplicate",
    "Street no",
    "Rest of PDF",
    "County",
    "Destination Longitude",
    "Destination Latitude",
    "Destination Type",
]

# California bounding box
CA_CENTER = {"lat": 37.2719, "lon": -119.2702}
CA_MAP_LAYOUT = dict(
    map_style="carto-positron",
    map_center=CA_CENTER,
    map_zoom=5,
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    height=800,
    width=1000,
)

# def _geocode_if_valid(addr, geocode_fn, **kwargs):
#     if not isinstance(addr, str) or not addr.strip():
#         return None
#     res = geocode_fn(addr, **kwargs)
#     lat, lng = (res[:2] if isinstance(res, (tuple, list)) and len(res) >= 2 else (None, None))
#     return (lat, lng) if (lat is not None and lng is not None) else None


def _geocode_if_valid(addr, geocode_fn, **kwargs):
    if not isinstance(addr, str) or not addr.strip() or pd.isna(addr):
        return None

    res = geocode_fn(addr, **kwargs)

    # geocode_address returns (lat, lng, meta); parcel geocoder often (lat, lng)
    if not isinstance(res, (tuple, list)) or len(res) < 2:
        return None

    lat, lng = res[0], res[1]
    return (lat, lng) if (lat is not None and lng is not None) else None


DEST_FINAL_COL = "Destination Address Final"
DEST_FINAL_SOURCE_COL = "Destination Address Final Source"


def _save_fig(fig, name):
    """Save a figure as PNG to outputs."""
    png_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
    with open(os.path.join(OUTPUTS_DIR, f"{name}.png"), "wb") as f:
        f.write(png_bytes)
    print(f"  Saved {name}")


_COORD_RE = re.compile(r"\s*\(?\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)?\s*$")

# Load both files
manual_df = pd.read_csv(MANUAL_PATH, engine="python", on_bad_lines="warn")
extracted_df = pd.read_csv(EXTRACTED_PATH)

# Coerce numeric columns (may have become strings after manual edits in step3)
coerce_numeric_columns(manual_df)

# Print DONE statistics
done_count = (manual_df["DONE"] == "x").sum()
done_pct = 100 * done_count / len(manual_df) if len(manual_df) > 0 else 0
print(f"Rows marked DONE: {done_count}/{len(manual_df)} ({done_pct:.1f}%)")

# Drop duplicate manifests marked manually in "Is Duplicate" column
dupe_mask = manual_df.get("Is Duplicate", pd.Series()) == "x"
n_dupes = dupe_mask.sum()
if n_dupes > 0:
    manual_df = manual_df[~dupe_mask].reset_index(drop=True)
    print(f"Dropped {n_dupes} duplicate manifests, {len(manual_df)} remaining")

# Key columns for matching
key_cols = ["Source PDF", "Manifest Number"]

# Find columns in extracted that are NOT in manual
cols_to_add = set(extracted_df.columns) - set(manual_df.columns) - set(key_cols)

print(f"\nColumns to add from extracted_manifests: {sorted(cols_to_add)}")

# Fill missing columns from extracted via merge (take first match per key)
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

print("\nResolving Destination Address Final + Geocoding")
manual_df[[DEST_FINAL_COL, DEST_FINAL_SOURCE_COL]] = None
source_counts = {}
geo_counts = {col: 0 for col in cols_to_geocode}

# Loop exits on first satisfied source
priority = [
    "Destination Assessor Parcel Number",
    "Destination Nearest Cross Street",
    "Destination Address",
    "Destination Contact Address",
    "Destination Name",
    "Hauler Address",
]

origin_address_col = "Origin Dairy Address"
origin_lat_col, origin_lng_col = "Origin Latitude", "Origin Longitude"
dest_lat_col, dest_lng_col = "Destination Latitude", "Destination Longitude"
manual_df[[origin_lat_col, origin_lng_col, dest_lat_col, dest_lng_col]] = None

for idx, row in manual_df.iterrows():
    addr = row[origin_address_col]
    county = row.get("County")
    if addr and (r := _geocode_if_valid(addr, geocode_address, county=county)):
        manual_df.at[idx, origin_lat_col] = r[0]
        manual_df.at[idx, origin_lng_col] = r[1]
        geo_counts[origin_address_col] += 1

    # Resolve destination address and geocode
    val = src = dest_geocoded = None
    for col in priority:
        raw = row.get(col)
        if not raw or pd.isna(raw):
            continue
        raw_str = str(raw).strip()
        if not raw_str or re.search(r"\bP\.?O\.?\s*Box\b", raw_str, re.IGNORECASE):
            continue

        if col == "Destination Assessor Parcel Number":
            parts = [x.strip() for x in raw_str.split(",") if x.strip()]
            hits = [r for p in parts if (r := _geocode_if_valid(p, geocode_parcel))]
            if hits:
                val, src = raw_str, col
                dest_geocoded = hits[0]  # use first parcel coords
                break
        elif col == "Destination Nearest Cross Street":
            m = _COORD_RE.match(raw_str)
            if m:
                dest_geocoded = (float(m.group(1)), float(m.group(2)))
                val, src = raw_str, col
                break
        elif col == "Destination Address":
            x = row.get("Destination Nearest Cross Street")
            if x and pd.notna(x) and str(x).strip() and not _COORD_RE.match(str(x)):
                raw_str = f"{raw_str} {str(x).strip()}"
            val, src = raw_str, col
            dest_geocoded = _geocode_if_valid(raw_str, geocode_address)
            break
        elif col == "Destination Name":
            if len(re.sub(r"[^a-zA-Z0-9]", "", raw_str)) >= 5:
                g = _geocode_if_valid(raw_str, geocode_address)
                if g:
                    val, src, dest_geocoded = raw_str, col, g
                    break
        else:
            val, src = raw_str, col
            dest_geocoded = _geocode_if_valid(raw_str, geocode_address)
            break

    if val:
        manual_df.at[idx, DEST_FINAL_COL] = val
        manual_df.at[idx, DEST_FINAL_SOURCE_COL] = src
        source_counts[src] = source_counts.get(src, 0) + 1
    if dest_geocoded:
        manual_df.at[idx, dest_lat_col] = dest_geocoded[0]
        manual_df.at[idx, dest_lng_col] = dest_geocoded[1]
        geo_counts["Destination Address"] += 1

resolved = manual_df[DEST_FINAL_COL].notna().sum()
print(f"  Resolved {resolved}/{len(manual_df)} destination addresses:")
for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"    {source}: {count}")
for col in cols_to_geocode:
    print(f"{geo_counts[col]}/{len(manual_df)} {col}")

# Enrich with city/zip/county from geocoded addresses
enrich_address_columns(manual_df, "Origin Dairy Address", prefix="Origin ", county_col_in="County")
enrich_address_columns(manual_df, "Destination Address", prefix="Destination ")

# Backfill missing mass columns: mass = volume * density
backfill_rules = [
    ("Total Manure Amount (tons)", "Total Manure Amount (cubic yards)"),
    ("Manure Mass per Haul (tons)", "Manure Volume per Haul (cubic yards)"),
]
for mass_col, vol_col in backfill_rules:
    backfill = manual_df[mass_col].isna() & manual_df[vol_col].notna()
    manual_df.loc[backfill, mass_col] = (
        manual_df.loc[backfill, vol_col] * manual_df.loc[backfill, "Manure Density"]
    )
    print(f"  Backfilled {backfill.sum()} mass values for {mass_col}")

# Backfill missing solids: solids = 1 - moisture
backfill_solids = (
    manual_df["Manure Solids (%)"].isna() & manual_df["Manure Moisture (%)"].notna()
)
manual_df.loc[backfill_solids, "Manure Solids (%)"] = (
    1 - manual_df.loc[backfill_solids, "Manure Moisture (%)"]
)
print(f"  Backfilled {backfill_solids.sum()} values for Manure Solids (%)")

# Drop columns now consolidated into tons/solids
early_drop = [
    "Total Manure Amount (cubic yards)",
    "Manure Volume per Haul (cubic yards)",
    "Manure Density",
    "Manure Moisture (%)",
]
manual_df = manual_df.drop(columns=[c for c in early_drop if c in manual_df.columns])

# Expand multi-parcel destinations into separate rows
amount_cols_to_split = [
    "Total Manure Amount (tons)",
    "Total Process Wastewater Exports (Gallons)",
    "Manure Mass per Haul (tons)",
    "Wastewater Volume per Haul (Gallons)",
]

new_rows = []
drop_idxs = []
for idx, row in manual_df.iterrows():
    apn_raw = row.get("Destination Assessor Parcel Number")
    if not apn_raw or pd.isna(apn_raw):
        continue
    # Validate each comma-separated part as a parcel number
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
        new_row["Destination Assessor Parcel Number"] = parcel
        for col in amount_cols_to_split:
            val = pd.to_numeric(new_row.get(col), errors="coerce")
            if pd.notna(val):
                new_row[col] = val / n
        # Geocode individual parcel
        r = _geocode_if_valid(parcel, geocode_parcel)
        if r:
            new_row[dest_lat_col] = r[0]
            new_row[dest_lng_col] = r[1]
        new_rows.append(new_row)
    drop_idxs.append(idx)

if new_rows:
    manual_df = manual_df.drop(drop_idxs)
    manual_df = pd.concat([manual_df, pd.DataFrame(new_rows)], ignore_index=True)
    print(f"  Expanded {len(drop_idxs)} multi-parcel rows into {len(new_rows)} rows")

# Aggregate destination types into broader categories
dest_type_col = "Destination Type (Standardized)"
merge_map = {
    "Composting Facility": ["Kelloggs", "Hyponex", "Fertilizer Company"],
    "Farmer": ["Spreader"],
    "Other": ["Garden"],
}
for new_type, old_types in merge_map.items():
    manual_df[dest_type_col] = manual_df[dest_type_col].replace(
        {k: new_type for k in old_types}
    )

# for rows with missing or non-geocoded Origin Dairy Address, backfill from main report outputs csv
# from LOCAL ca_cafo_compliance/local/Dairy_Data_and_Analysis/Data/Summary
needs_backfill = (
    manual_df["Origin Dairy Address"].isna() | manual_df["Origin Latitude"].isna()
)
print(f"{needs_backfill.sum()} need origin dairy address backfill")
dairy_summary_df = pd.read_csv(
    "ca_cafo_compliance/local/Dairy_Data_and_Analysis/Data/Summary/Dairy_Report_Summary_Region_5_2024_pdf_merged.csv"
)
dairy_summary_df = dairy_summary_df.rename(
    columns={"Dairy Address": "Origin Dairy Address"}
)
dairy_summary_df["Source PDF"] = dairy_summary_df["Source PDF"].str.replace(
    r"\.pdf$", "", regex=True
)
lookup = dairy_summary_df.drop_duplicates(subset="Source PDF").set_index("Source PDF")[
    "Origin Dairy Address"
]
manual_df.loc[needs_backfill, "Origin Dairy Address"] = manual_df.loc[
    needs_backfill, "Source PDF"
].map(lookup)
num_backfilled = (
    needs_backfill.sum()
    - manual_df.loc[needs_backfill, "Origin Dairy Address"].isna().sum()
)
print(f"Backfilled {num_backfilled} rows")

# print the name of remaining rows with missing origin dairy address
# print unique FACILITIES< not manifests
remaining_missing = (
    manual_df[manual_df["Origin Dairy Address"].isna()]
    .get("Source PDF", pd.Series())
    .loc[manual_df[manual_df["Origin Dairy Address"].isna()].index]
    .tolist()
)
remaining_missing = list(set(remaining_missing))
print(f"Remaining rows with missing origin dairy address: {len(remaining_missing)}")
if remaining_missing:
    print("Source PDFs with missing origin dairy address:")
    for pdf in remaining_missing:
        print(f"  {pdf}")

# Split by manifest type and drop irrelevant columns
manifest_type_col = "Manifest Type"
manure_mask = manual_df[manifest_type_col].isin(["manure", "both"])
manure_cols = [c for c in manual_df.columns if c not in specific_cols["wastewater"]]
df_manure = manual_df.loc[manure_mask, manure_cols].copy()

wastewater_mask = manual_df[manifest_type_col].isin(["wastewater", "both"])
wastewater_cols = [c for c in manual_df.columns if c not in specific_cols["manure"]]
df_ww = manual_df.loc[wastewater_mask, wastewater_cols].copy()

print(f"  Manure + both: {len(df_manure)} rows")
print(f"  Wastewater + both: {len(df_ww)} rows")


tons_per_haul = df_manure["Manure Mass per Haul (tons)"].dropna()
# Average per facility (PDF) so high-manifest facilities don't dominate
avg_tons_per_haul = (
    df_manure.dropna(subset=["Manure Mass per Haul (tons)"])
    .groupby("Source PDF")["Manure Mass per Haul (tons)"]
    .mean()
    .mean()
)

df_manure["Estimated Number of Hauls"] = (
    (df_manure["Total Manure Amount (tons)"] / avg_tons_per_haul)
    .round()
    .astype("Int64")
)

print(f"Average: {avg_tons_per_haul:.2f} tons/haul")
print(f"  Estimated hauls: {df_manure['Estimated Number of Hauls'].sum():.0f}")
print(f"  Total tons: {df_manure['Total Manure Amount (tons)'].sum():.0f}")

# Save tons-per-haul histogram (binned in 5-ton increments)
fig_hist = px.histogram(
    tons_per_haul,
    nbins=10,
    labels={"value": "Tons per Haul", "count": "Number of Manifests"},
    title="Distribution of Manure Tons per Haul",
)
fig_hist.update_layout(
    showlegend=False, xaxis_title="Tons per Haul", yaxis_title="Number of Manifests"
)
_save_fig(fig_hist, "2024_manure_tons_per_haul_histogram")

# Per-type config: (label, dataframe, amount column, unit)
_type_configs = [
    ("Manure", df_manure, "Total Manure Amount (tons)", "tons"),
    ("Wastewater", df_ww, "Total Process Wastewater Exports (Gallons)", "gallons"),
]

# Summary statistics, drop extra columns, save CSVs
print("\nTemplates breakdown:")
print(manual_df["Parameter Template"].value_counts())

for label, df, amount_col, unit in _type_configs:
    print(f"\n{label} summary by destination type:")
    print(df.groupby(dest_type_col)[amount_col].sum())
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    fname = f"2024_manifests_{label.lower()}.csv"
    df.to_csv(os.path.join(OUTPUTS_DIR, fname), index=False)
    df.to_csv(os.path.join(GDRIVE_BASE, fname), index=False)

# Interactive maps (HTML only — carto-positron basemap shows state/county boundaries)
_map_configs = [
    ("Origin Dairy Address", "Origin Latitude", "Origin Longitude"),
    ("Destination Address", "Destination Latitude", "Destination Longitude"),
]
for col, lat_c, lng_c in _map_configs:
    has_geo = manual_df[lat_c].notna() & manual_df[lng_c].notna()
    if not has_geo.any():
        continue
    subset = manual_df.loc[has_geo].copy()
    subset["Dairy Name"] = subset.get("Origin Dairy Name", "Unknown")
    subset["Address"] = subset[col]

    fig = px.scatter_map(
        subset,
        lat=lat_c,
        lon=lng_c,
        color="Manifest Type",
        hover_name="Dairy Name",
        hover_data={"Address": True, "Manifest Type": True},
        title=col,
    )
    fig.update_layout(**CA_MAP_LAYOUT)
    fig.write_html(os.path.join(OUTPUTS_DIR, f"2024 {col} map.html"))
    print(f"  Saved {col} map")

# Pie charts: destination type breakdown
for label, df, amount_col, unit in _type_configs:
    type_counts = df[dest_type_col].value_counts()
    fig = px.pie(names=type_counts.index, values=type_counts.values)
    _save_fig(fig, f"2024_{label.lower()}_destination_type_pie")

# Monthly charts: amount per month
first_col, last_col = "First Haul Date", "Last Haul Date"

for label, df, amount_col, unit in _type_configs:
    monthly_amount = pd.Series(0.0, index=range(1, 13))

    for _, row in df.iterrows():
        amt = pd.to_numeric(row.get(amount_col), errors="coerce")
        if pd.isna(amt):
            continue

        first = pd.to_datetime(
            row.get(first_col), format="mixed", dayfirst=False, errors="coerce"
        )
        last = pd.to_datetime(
            row.get(last_col), format="mixed", dayfirst=False, errors="coerce"
        )

        if pd.isna(last) and pd.isna(first):
            continue
        if pd.isna(first):
            first = last

        # Skip full-year reports (Jan–Dec)
        if first.month == 1 and last.month == 12:
            continue

        lo, hi = min(first.month, last.month), max(first.month, last.month)
        months = list(range(lo, hi + 1))
        weight = 1.0 / len(months)
        for mo in months:
            monthly_amount[mo] += amt * weight

    month_labels = [
        pd.Timestamp(month=m, day=1, year=2024).strftime("%b") for m in range(1, 13)
    ]

    # Normalize to proportions
    total = monthly_amount.sum()
    if total > 0:
        monthly_amount = monthly_amount / total

    fig = px.bar(
        x=month_labels,
        y=monthly_amount.values,
        labels={"y": f"Total {label} ({unit})"},
    )
    _save_fig(fig, f"2024_{label.lower()}_amount_per_month")
