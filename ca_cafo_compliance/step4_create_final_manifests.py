import os
import re
import pandas as pd
import plotly.express as px
from ast import literal_eval

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

# Columns to EXCLUDE from each output: wastewater-only cols excluded from manure, vice versa.
_params_df = pd.read_csv(PARAMETERS_PATH)
specific_cols = {
    t: set(_params_df.loc[_params_df["manifest_type"] == t, "parameter_name"])
    for t in ["wastewater", "manure"]
}


cols_to_geocode = ["Origin Dairy Address", "Destination Address"]
_geo = lambda col: f"{col} (Geocoded)"
_GEOCODING_COLS = {_geo(c) for c in cols_to_geocode}
cols_to_drop = [
    "Unnamed: 22",
    "DONE",
    "Is Duplicate",
    "Street no",
    "Rest of PDF",
    "County",
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


def _geocode_if_valid(addr, geocode_fn, cache):
    if not addr or pd.isna(addr):
        return None
    result = geocode_fn(addr, cache)
    return result if result and all(v is not None for v in result) else None


DEST_FINAL_COL = "Destination Address Final"
DEST_FINAL_SOURCE_COL = "Destination Address Final Source"


def _save_fig(fig, name):
    """Save a figure as PNG to outputs."""
    png_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
    with open(os.path.join(OUTPUTS_DIR, f"{name}.png"), "wb") as f:
        f.write(png_bytes)
    print(f"  Saved {name}")


_COORD_RE = re.compile(r"\s*\(?\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)?\s*$")


cache = load_geocoding_cache()

# Load both files
manual_df = pd.read_csv(MANUAL_PATH)
extracted_df = pd.read_csv(EXTRACTED_PATH)

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

# Find columns in extracted that are NOT in manual (excluding geocoding)
cols_to_add = (
    set(extracted_df.columns) - set(manual_df.columns) - _GEOCODING_COLS - set(key_cols)
)

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
manual_df[list(_GEOCODING_COLS)] = None
source_counts = {}
geo_counts = {col: 0 for col in _GEOCODING_COLS}

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
for idx, row in manual_df.iterrows():
    addr = row[origin_address_col]
    if addr and (r := _geocode_if_valid(addr, geocode_address, cache)):
        manual_df.at[idx, _geo(origin_address_col)] = str(r)
        geo_counts[_geo(origin_address_col)] += 1

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
            hits = [
                r for p in parts if (r := _geocode_if_valid(p, geocode_parcel, cache))
            ]
            if hits:
                val, src = raw_str, col
                dest_geocoded = hits if len(hits) > 1 else hits[0]
                break
        elif col == "Destination Nearest Cross Street":
            m = _COORD_RE.match(raw_str)
            if m:
                c = (float(m.group(1)), float(m.group(2)))
                val, src, dest_geocoded = str(c), col, c
                break
        elif col == "Destination Address":
            x = row.get("Destination Nearest Cross Street")
            if x and pd.notna(x) and str(x).strip() and not _COORD_RE.match(str(x)):
                raw_str = f"{raw_str} {str(x).strip()}"
            val, src = raw_str, col
            dest_geocoded = _geocode_if_valid(raw_str, geocode_address, cache)
            break
        elif col == "Destination Name":
            if len(re.sub(r"[^a-zA-Z0-9]", "", raw_str)) >= 5:
                g = _geocode_if_valid(raw_str, geocode_address, cache)
                if g:
                    val, src, dest_geocoded = str(g), col, g
                    break
        else:
            val, src = raw_str, col
            dest_geocoded = _geocode_if_valid(raw_str, geocode_address, cache)
            break

    if val:
        manual_df.at[idx, DEST_FINAL_COL] = val
        manual_df.at[idx, DEST_FINAL_SOURCE_COL] = src
        source_counts[src] = source_counts.get(src, 0) + 1
    if dest_geocoded:
        manual_df.at[idx, _geo("Destination Address")] = str(dest_geocoded)
        geo_counts[_geo("Destination Address")] += 1

resolved = manual_df[DEST_FINAL_COL].notna().sum()
print(f"  Resolved {resolved}/{len(manual_df)} destination addresses:")
for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"    {source}: {count}")
for col in cols_to_geocode:
    print(f"{geo_counts[_geo(col)]}/{len(manual_df)} {col}")
save_geocoding_cache(cache)

# Aggregate destination types into broader categories
dest_type_col = "Destination Type (Standardized)"
merge_with_composting = ["Kelloggs", "Hyponex", "Fertilizer Company"]
manual_df[dest_type_col] = manual_df[dest_type_col].replace(
    {k: "Composting Facility" for k in merge_with_composting}
)

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

# Coerce numeric columns and backfill tons from cubic yards * density where missing
# for col in [
#     "Manure Mass per Haul (tons)",
#     "Total Manure Amount (tons)",
#     "Total Manure Amount (cubic yards)",
#     "Manure Density",
# ]:
#     df_manure[col] = pd.to_numeric(df_manure[col], errors="coerce")

missing_tons = df_manure["Total Manure Amount (tons)"].isna()
has_cy = (
    df_manure["Total Manure Amount (cubic yards)"].notna()
    & df_manure["Manure Density"].notna()
)
backfill = missing_tons & has_cy
df_manure.loc[backfill, "Total Manure Amount (tons)"] = (
    df_manure.loc[backfill, "Total Manure Amount (cubic yards)"]
    * df_manure.loc[backfill, "Manure Density"]
)
print(f"  Backfilled {backfill.sum()} manure tons from cubic yards * density")

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
for col in cols_to_geocode:
    geo_col = _geo(col)
    has_geo = manual_df[geo_col].notna()
    if not has_geo.any():
        continue
    subset = manual_df.loc[has_geo].copy()
    # Parse geocoded strings into coord lists, explode multi-parcel rows
    subset["_coords"] = (
        subset[geo_col]
        .apply(lambda s: literal_eval(str(s).strip()))
        .apply(lambda p: p if isinstance(p, list) else [p])
    )
    subset = subset.explode("_coords")
    subset["Latitude"] = subset["_coords"].apply(lambda c: float(c[0]))
    subset["Longitude"] = subset["_coords"].apply(lambda c: float(c[1]))
    subset["Dairy Name"] = subset.get("Origin Dairy Name", "Unknown")
    subset["Address"] = subset[col]

    fig = px.scatter_map(
        subset,
        lat="Latitude",
        lon="Longitude",
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

        # Skip full-year reports (Jan–Dec) — not specific enough
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
