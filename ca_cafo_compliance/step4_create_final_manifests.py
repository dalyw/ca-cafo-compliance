import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helpers_geocoding import enrich_address_columns, geocode_address, geocode_parcel
from helpers_pdf_metrics import PARAMETERS_DF, build_parameter_dicts, coerce_columns
from helpers_plotting import MANIFEST_TYPE_COLORS, TYPE_COLOR_SEQ, manure_colors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MANUAL_PATH = os.path.join(OUTPUTS_DIR, "as_written_manifests_validated.csv")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "as_written_manifests_automatic.csv")

# Column name mapping: parameter_key -> display name (e.g. P["origin_geo_lat"])
P = build_parameter_dicts(manifest_only=True)["key_to_name"]

# Columns exclusive to each manifest type (for splitting outputs)
specific_cols = {
    t: set(PARAMETERS_DF.loc[PARAMETERS_DF["manifest_type"] == t, "parameter_name"])
    for t in ["wastewater", "manure"]
}

METATADA_COLS = [
    "Source PDF",
    "Manifest Number",
    "Start Page",
    "End Page",
]

COLS_TO_KEEP = [
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


def save_fig(fig, name):
    """Save a figure as PNG to outputs."""
    png_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
    with open(os.path.join(OUTPUTS_DIR, f"{name}.png"), "wb") as f:
        f.write(png_bytes)
    print(f"  Saved {name}")


def weighted_avg(df, val_col, weight_col):
    valid = df.dropna(subset=[val_col, weight_col])
    return (
        (valid[val_col] * valid[weight_col]).sum()
        / valid[weight_col].sum()
        # if not valid.empty
        # else float("nan")
    )


def _build_type_weights(series):
    weights = {}
    for v in series.dropna().astype(str):
        parts = [p.strip() for p in v.split(",") if p.strip()]
        w = 1.0 / len(parts) if parts else 0
        for p in parts:
            weights[p] = weights.get(p, 0.0) + w
    return (
        pd.Series(weights).sort_values(ascending=False) if weights else pd.Series([], dtype=float)
    )


def _monthly_allocation(df, amount_col, date_first_col, date_last_col):
    monthly = pd.Series(0.0, index=range(1, 13))
    for _, row in df.iterrows():
        amt = pd.to_numeric(row.get(amount_col), errors="coerce")
        if pd.isna(amt):
            continue
        first = pd.to_datetime(
            row.get(date_first_col), format="mixed", dayfirst=False, errors="coerce"
        )
        last = pd.to_datetime(
            row.get(date_last_col), format="mixed", dayfirst=False, errors="coerce"
        )
        if pd.isna(first) and pd.isna(last):
            continue
        if pd.isna(first):
            first = last
        if first.month == 1 and last.month == 12:
            continue
        lo, hi = min(first.month, last.month), max(first.month, last.month)
        for mo in range(lo, hi + 1):
            monthly[mo] += amt / (hi - lo + 1)
    total = monthly.sum()
    return monthly / total if total > 0 else monthly


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
dup_subset = [
    c for c in manual_df.columns if c not in ["Manifest Number", "Start Page", "End Page"]
]
dupes = manual_df[manual_df.duplicated(subset=dup_subset, keep="first")]
# if not dupes.empty:
print("Dropping exact-duplicate rows")
for _, r in dupes[["Source PDF", "Manifest Number"]].iterrows():
    print(f"  Source PDF={r['Source PDF']}, Manifest {r['Manifest Number']}")
manual_df = manual_df.drop_duplicates(subset=dup_subset)

# Geocode origins and resolve destinations
print("\nResolving Destination Address Final + Geocoding")
manual_df[[P["destination_address_final"], P["destination_address_final_source"]]] = None

latlong_col_keys = [
    "origin_geo_lat",
    "origin_geo_lng",
    "destination_geo_lat",
    "destination_geo_lng",
]
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
                hits = [r for p in parts if (r := geocode_if_valid(p, geocode_parcel))]
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
                dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)
            break

        elif dest_col == P["destination_contact_address"]:
            if len(re.sub(r"[^a-zA-Z0-9]", "", raw_str)) >= 5:
                if has_existing_coords:
                    val, src = raw_str, dest_col
                else:
                    g = geocode_if_valid(raw_str, geocode_address, county=parcel_county)
                    if g:
                        val, src, dest_geocoded = raw_str, dest_col, g
                    break

        elif dest_col == P["hauler_address"]:
            # Only use hauler address if it looks like a farm/compost destination
            hauler_combined = (
                f"{str(row.get(P['hauler_name'], '') or '').lower()} {raw_str.lower()}"
            )
            if not any(s in hauler_combined for s in ("farm", "compost", "fertilizer")):
                continue
            val, src = raw_str, dest_col
            if not has_existing_coords:
                dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)
            break

        else:
            val, src = raw_str, dest_col
            if not has_existing_coords:
                dest_geocoded = geocode_if_valid(raw_str, geocode_address, county=parcel_county)
            break
    if val:
        manual_df.at[idx, P["destination_address_final"]] = val
        manual_df.at[idx, P["destination_address_final_source"]] = src
        source_counts[src] = source_counts.get(src, 0) + 1
    if dest_geocoded:
        manual_df.at[idx, P["destination_geo_lat"]] = dest_geocoded[0]
        manual_df.at[idx, P["destination_geo_lng"]] = dest_geocoded[1]
        n_dest_geo += 1

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
print(f"Backfilled {filled} rows")

remaining = manual_df.loc[manual_df[origin_col].isna(), "Source PDF"].unique().tolist()
print(f"Remaining rows with missing origin dairy address: {len(remaining)}")
for pdf in remaining:
    print(f"  {pdf}")

# Split by manifest type, compute stats, save CSVs
manure_mask = manual_df["Manifest Type"].isin(["manure", "both"])
manure_cols = [c for c in manual_df.columns if c not in specific_cols["wastewater"]]
df_manure = manual_df.loc[manure_mask, manure_cols].copy()

wastewater_mask = manual_df["Manifest Type"].isin(["wastewater", "both"])
wastewater_cols = [c for c in manual_df.columns if c not in specific_cols["manure"]]
df_ww = manual_df.loc[wastewater_mask, wastewater_cols].copy()

print(f"  Manure + both: {len(df_manure)} rows")
print(f"  Wastewater + both: {len(df_ww)} rows")

type_configs = [
    ("Manure", df_manure, P["manure_amount"], "tons"),
    ("Wastewater", df_ww, P["wastewater_amount"], "gallons"),
]

# Per-facility averages and weighted averages across hauls
haul_cfg = [
    ("Manure", df_manure, P["manure_ton_per_haul"], P["manure_number_hauls"]),
    ("Wastewater", df_ww, P["wastewater_gallon_per_haul"], P["wastewater_number_hauls"]),
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
        avg_facility=fac["avg_rate"].mean(),
        avg_weighted=weighted_avg(df, rate_col, haul_col),
    )

# Haul estimates: distribution-based (manure + wastewater)
haul_bins = [
    (5, 15, 15, 25, 10.0, 20.0, "10-ton", "20-ton"),
    (5_000, 15_000, 15_000, 25_000, 10_000.0, 20_000.0, "10k-gallon", "20k-gallon"),
]
for (label, df, rate_col, _), (_, _, amount_col, unit), bins in zip(
    haul_cfg, type_configs, haul_bins
):
    lo1, hi1, lo2, hi2, b1v, b2v, b1n, b2n = bins
    avg_haul = haul_stats[label]["avg_weighted"]
    rate = df[rate_col]
    mass_lo = df.loc[rate.between(lo1, hi1, inclusive="left"), amount_col].sum()
    mass_hi = df.loc[rate.between(lo2, hi2, inclusive="left"), amount_col].sum()
    total = mass_lo + mass_hi
    p_lo = mass_lo / total if total > 0 else 0.5
    p_hi = 1.0 - p_lo
    print(f"{label} split: {p_lo:.1%} at ~{b1n}, {p_hi:.1%} at ~{b2n}")
    df[f"Estimated Number of {b1n} Hauls"] = (df[amount_col] * p_lo / b1v).round().astype("Int64")
    df[f"Estimated Number of {b2n} Hauls"] = (df[amount_col] * p_hi / b2v).round().astype("Int64")
    print(f"Average {label.lower()} haul: {avg_haul:.2f} {unit}/haul")

# Tons-per-haul & facility-level scatter subplot (manure + wastewater)


# Aggregate exports by facility (Source PDF), normalizing split manifests like '1p1'
def _facility_agg(df, amount_col):
    tmp = df.copy()
    tmp["_manifest_norm"] = tmp["Manifest Number"].astype(str).str.replace(r"p\d+$", "", regex=True)
    per_manifest = tmp.groupby(["Source PDF", "_manifest_norm"])[amount_col].sum().reset_index()
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

    counts = pd.cut(x_vals, bins=bin_edges, labels=False, include_lowest=True, right=False)

    bin_totals = []
    bin_centers = []
    for i in range(len(bin_edges) - 1):
        mask = counts == i
        if not mask.any():
            continue
        bin_totals.append(totals[mask].sum())
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
    return bin_centers, bin_totals, bin_edges


def edges_to_xbins(edges):
    if edges is None:
        return None
    return dict(start=float(edges[0]), end=float(edges[-1]), size=float(edges[1] - edges[0]))


for col_idx, (label, *rest) in enumerate(type_configs, start=1):
    hs = haul_stats[label]
    bin_x, bin_totals, edges = _binned_totals(
        hs["per_haul_series"], hs["facility_hauls"]["total_hauls"], nbins=10
    )
    print(f"{label} bins:", list(zip(bin_x, bin_totals)))
    counts, ignored = np.histogram(hs["per_haul_series"], bins=edges)
    fig_hauls.update_yaxes(
        range=[0, (max(bin_totals) if bin_totals else 0) * 1.1],
        row=1,
        col=col_idx,
        secondary_y=True,
    )
    fig_hauls.update_yaxes(
        range=[0, counts.max() * 1.1],
        row=1,
        col=col_idx,
        secondary_y=False,
    )
    hs.update(bin_x=bin_x, bin_totals=bin_totals, xbins=edges_to_xbins(edges))

col_configs = [
    dict(
        color=manure_colors[0],
        x_hist=haul_stats["Manure"]["per_haul_series"],
        xbins=haul_stats["Manure"]["xbins"],
        bin_x=haul_stats["Manure"]["bin_x"],
        bin_totals=haul_stats["Manure"]["bin_totals"],
        avg_facility=haul_stats["Manure"]["avg_facility"],
        avg_weighted=haul_stats["Manure"]["avg_weighted"],
        vline_fmt=lambda v: round(v, 1),
        facility_df=manure_facility,
    ),
    dict(
        color=MANIFEST_TYPE_COLORS.get("wastewater", "#1f77b4"),
        x_hist=haul_stats["Wastewater"]["per_haul_series"],
        xbins=haul_stats["Wastewater"]["xbins"],
        bin_x=haul_stats["Wastewater"]["bin_x"],
        bin_totals=haul_stats["Wastewater"]["bin_totals"],
        avg_facility=haul_stats["Wastewater"]["avg_facility"],
        avg_weighted=haul_stats["Wastewater"]["avg_weighted"],
        vline_fmt=int,
        facility_df=ww_facility,
    ),
]

for col, (cfg, (label, _, _, unit)) in enumerate(zip(col_configs, type_configs), start=1):
    color, fmt, fac_df = cfg["color"], cfg["vline_fmt"], cfg["facility_df"]
    avg_fac, avg_w = cfg["avg_facility"], cfg["avg_weighted"]
    pos_fac, pos_w = ("top left", "top right") if col == 1 else ("top right", "top left")

    # Top-row histogram (facility count) + total-hauls scatter overlay
    fig_hauls.add_trace(
        go.Histogram(x=cfg["x_hist"], xbins=cfg["xbins"], marker_color=color, showlegend=False),
        row=1,
        col=col,
        secondary_y=False,
    )
    fig_hauls.add_trace(
        go.Scatter(
            x=cfg["bin_x"],
            y=cfg["bin_totals"],
            mode="markers",
            marker=dict(color="black", size=8, opacity=0.9, symbol="circle"),
            showlegend=False,
        ),
        row=1,
        col=col,
        secondary_y=True,
    )
    # Vertical lines: facility mean (solid) and haul-weighted mean (dotted)
    fig_hauls.add_vline(
        x=avg_fac,
        line_color="black",
        line_width=2,
        row=1,
        col=col,
        annotation_text=fmt(avg_fac),
        annotation_position=pos_fac,
    )
    fig_hauls.add_vline(
        x=avg_w,
        line_dash="dot",
        line_color="black",
        line_width=2,
        row=1,
        col=col,
        annotation_text=fmt(avg_w),
        annotation_position=pos_w,
    )
    # Bottom-row scatter: facility exports vs manifest count
    fig_hauls.add_trace(
        go.Scatter(
            x=fac_df["total_amount"],
            y=fac_df["manifest_count"],
            mode="markers",
            marker=dict(color=color),
            showlegend=False,
        ),
        row=2,
        col=col,
    )
    fig_hauls.update_xaxes(title_text=f"{unit.capitalize()} per haul", row=1, col=col)
    fig_hauls.update_yaxes(title_text="Number of facilities", row=1, col=col, secondary_y=False)
    fig_hauls.update_yaxes(title_text="Total hauls", row=1, col=col, secondary_y=True)
    fig_hauls.update_xaxes(title_text=f"Total Facility Exports ({unit}) in 2024", row=2, col=col)
    fig_hauls.update_yaxes(title_text="Manifests per Facility", row=2, col=col)
    fig_hauls.add_annotation(
        text=label,
        x=0 if col == 1 else 0.6,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )

# Legend-only entries for shapes (no color encoding)
for mode, props, name in [
    ("markers", dict(color="black", size=10, symbol="square"), "Facility Count"),
    ("markers", dict(color="black", size=8, symbol="circle"), "Total Hauls in Bin"),
    ("lines", dict(color="black", width=2), "Average by Facility"),
    ("lines", dict(color="black", width=2, dash="dot"), "Average by Hauls"),
]:
    kw = dict(marker=props) if mode == "markers" else dict(line=props)
    fig_hauls.add_trace(go.Scatter(x=[None], y=[None], mode=mode, name=name, **kw))

fig_hauls.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1000,
    height=700,
    font=dict(size=16),
    legend=dict(x=1.1, y=1.0, xanchor="left", yanchor="top"),
)
fig_hauls.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
fig_hauls.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

save_fig(fig_hauls, "2024_tons_per_haul")

print("\nTemplates breakdown:")
print(manual_df["Parameter Template"].value_counts())

param_order = PARAMETERS_DF["parameter_name"].tolist()
for label, df, amount_col, unit in type_configs:
    print(f"\n{label} summary by destination type:")
    print(df.groupby(P["destination_type_std"])[amount_col].sum())
    type_qty_cols = [
        c
        for c in param_order
        if c in specific_cols[label.lower()] and c in df.columns and not c.startswith("Method Used")
    ]
    estimated_cols = [c for c in df.columns if c.startswith("Estimated")]
    cols = [c for c in COLS_TO_KEEP + type_qty_cols + estimated_cols if c in df.columns]
    df[cols].to_csv(
        os.path.join(OUTPUTS_DIR, f"processed_{label.lower()}_manifests.csv"),
        index=False,
    )

# Interactive maps
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

# Combined 2x2 subplot: pie charts + monthly bar charts
month_labels = [pd.Timestamp(month=m, day=1, year=2024).strftime("%b") for m in range(1, 13)]

fig_combined = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"type": "pie"}, {"type": "pie"}], [{"type": "bar"}, {"type": "bar"}]],
    subplot_titles=[f"{l} Destination Types" for l, *_ in type_configs]
    + [f"{l} Hauls by Month" for l, *_ in type_configs],
)

for col_idx, (label, df, amount_col, unit) in enumerate(type_configs, start=1):
    colors = TYPE_COLOR_SEQ[label]

    type_counts = _build_type_weights(df[P["destination_type_std"]])
    n_blank = df[P["destination_type_std"]].isna().sum()
    type_counts["N/A"] = n_blank
    pie_colors = (list(colors) + ["#cccccc"])[: len(type_counts)]

    # Reposition labels for Wastewater pie to reduce overlap
    if label == "Wastewater":
        pulls = [0.2 if name not in ("Farmer", "N/A") else 0.0 for name in type_counts.index]
    else:
        pulls = [0.0] * len(type_counts)
    pie = go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        marker_colors=pie_colors,
        textposition="outside",
        textinfo="label+percent",
        textfont=dict(size=12),
        pull=pulls,
        rotation=330,
    )

    fig_combined.add_trace(pie, row=1, col=col_idx)

    monthly_amount = _monthly_allocation(df, amount_col, P["haul_date_first"], P["haul_date_last"])

    fig_combined.add_trace(
        go.Bar(x=month_labels, y=monthly_amount.values, marker_color=colors[0], showlegend=False),
        row=2,
        col=col_idx,
    )

fig_combined.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
fig_combined.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

fig_combined.update_layout(
    height=500,
    width=900,
    showlegend=False,
    margin=dict(t=40, b=30, l=40, r=20),
    plot_bgcolor="white",
)
for col_idx in [1, 2]:
    fig_combined.update_yaxes(range=[0, 0.15], dtick=0.05, row=2, col=col_idx)
save_fig(fig_combined, "2024_manifest_summary")
