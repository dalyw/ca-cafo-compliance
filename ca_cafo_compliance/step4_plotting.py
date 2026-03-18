import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from helpers_geocoding import norm_addr, normalize_apn
from helpers_pdf_metrics import PARAMETERS_DF, build_parameter_dicts
from helpers_plotting import MANIFEST_TYPE_COLORS, PALETTE, TYPE_COLOR_SEQ, manure_colors, save_fig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "as_written_manifests_automatic.csv")
P = build_parameter_dicts(manifest_only=True)["key_to_name"]


# --- Load data ---
df_manure = pd.read_csv(os.path.join(OUTPUTS_DIR, "processed_manure_manifests.csv"))
df_ww = pd.read_csv(os.path.join(OUTPUTS_DIR, "processed_wastewater_manifests.csv"))
extracted_df = pd.read_csv(EXTRACTED_PATH)
manual_src = pd.read_csv(
    os.path.join(OUTPUTS_DIR, "as_written_manifests_validated.csv"), engine="python", on_bad_lines="warn"
)


WATER_DENSITY = 8.34 / 2_000  # tons per gallon
ww_np = df_ww[df_ww[P["is_pipeline"]].ne(True)]  # non-pipeline only

type_configs = [
    ("Manure", df_manure, P["manure_amount"], "tons"),
    ("Wastewater", df_ww, P["wastewater_amount"], "gallons"),
]
haul_cfg = [
    ("Manure", df_manure, P["manure_ton_per_haul"], P["manure_number_hauls"], 1.0),
    ("Wastewater", ww_np, P["wastewater_gallon_per_haul"], P["wastewater_number_hauls"], WATER_DENSITY),
]


# --- Helpers ---
def weighted_avg(df, val_col, weight_col):
    valid = df.dropna(subset=[val_col, weight_col])
    return (valid[val_col] * valid[weight_col]).sum() / valid[weight_col].sum()


def build_type_weights(series):
    weights = {}
    for v in series.dropna().astype(str):
        parts = [p.strip() for p in v.split(",") if p.strip()]
        w = 1.0 / len(parts) if parts else 0
        for p in parts:
            weights[p] = weights.get(p, 0.0) + w
    return pd.Series(weights).sort_values(ascending=False) if weights else pd.Series([], dtype=float)


def monthly_allocation(df, amount_col, date_first_col, date_last_col):
    monthly = pd.Series(0.0, index=range(1, 13))
    for _, row in df.iterrows():
        amt = pd.to_numeric(row.get(amount_col), errors="coerce")
        if pd.isna(amt):
            continue
        first = pd.to_datetime(row.get(date_first_col), format="mixed", dayfirst=False, errors="coerce")
        last = pd.to_datetime(row.get(date_last_col), format="mixed", dayfirst=False, errors="coerce")
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


def facility_agg(df, amount_col):
    tmp = df.copy()
    tmp["_manifest_norm"] = tmp["Manifest Number"].astype(str).str.replace(r"p\d+$", "", regex=True)
    per_manifest = tmp.groupby(["Source PDF", "_manifest_norm"])[amount_col].sum().reset_index()
    return (
        per_manifest.groupby("Source PDF")
        .agg(total_amount=(amount_col, "sum"), manifest_count=("_manifest_norm", "nunique"))
        .reset_index()
    )


def haul_bins(x_vals, totals, nbins=10):
    """Bin per-facility rates; return (centers, bin_totals, counts, xbins)."""
    if len(x_vals) == 0:
        return [], [], np.array([0]), None
    counts, edges = np.histogram(x_vals, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    cuts = pd.cut(x_vals, bins=edges, labels=False, include_lowest=True)
    bin_totals = [totals[cuts == i].sum() for i in range(len(edges) - 1)]
    xbins = dict(start=float(edges[0]), end=float(edges[-1]), size=float(edges[1] - edges[0]))
    return centers.tolist(), bin_totals, counts, xbins


# --- Haul stats (rates converted to tons/haul via scale) ---
haul_stats = {}
for label, df, rate_col, haul_col, scale in haul_cfg:
    fac = (
        df.dropna(subset=[rate_col, haul_col])
        .groupby("Source PDF")
        .agg(avg_rate=(rate_col, "mean"), total_hauls=(haul_col, "sum"))
    )
    haul_stats[label] = dict(
        facility_hauls=fac,
        per_haul_series=fac["avg_rate"] * scale,
        avg_facility=(fac["avg_rate"] * scale).mean(),
        avg_weighted=weighted_avg(df, rate_col, haul_col) * scale,
    )

manure_facility = facility_agg(df_manure, P["manure_amount"])
ww_facility = facility_agg(df_ww, P["wastewater_amount"])

# --- Hauls subplot ---
fig_hauls = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"secondary_y": True}, {"secondary_y": True}], [{}, {}]],
)

for col_idx, (label, *_) in enumerate(type_configs, start=1):
    hs = haul_stats[label]
    bin_x, bin_totals, counts, xbins = haul_bins(
        hs["per_haul_series"], hs["facility_hauls"]["total_hauls"]
    )
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
    hs.update(bin_x=bin_x, bin_totals=bin_totals, xbins=xbins)

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


# --- Interactive maps ---
map_configs = [
    (P["origin_dairy_address"], P["origin_geo_lat"], P["origin_geo_lng"]),
    (P["destination_address"], P["destination_geo_lat"], P["destination_geo_lng"]),
]
for col, lat_c, lng_c in map_configs:
    fig = go.Figure()
    for label, df, *_ in type_configs:
        subset = df[df[lat_c].notna() & df[lng_c].notna()].copy()
        if subset.empty:
            continue
        subset["Geocoded Text"] = subset.get(
            P["destination_address_final"] if "Dest" in col else col, ""
        )
        subset["Address Source"] = subset.get(P["destination_address_final_source"], col)
        sub_fig = px.scatter_map(
            subset, lat=lat_c, lon=lng_c,
            color_discrete_sequence=[MANIFEST_TYPE_COLORS.get(label.lower(), "#888")],
            hover_name=P["origin_dairy_name"],
            hover_data={
                "Source PDF": True, "Manifest Number": True,
                "Address Source": True, "Geocoded Text": True,
                lat_c: False, lng_c: False,
            },
        )
        for trace in sub_fig.data:
            trace.name = label
            fig.add_trace(trace)
    fig.update_layout(
        map_center={"lat": 37.2719, "lon": -119.2702},
        title=col)
    filename = f"2024_{col.lower().replace(' ', '_')}_map.html"
    fig.write_html(os.path.join(OUTPUTS_DIR, filename))
    print(f"  Saved {col} map")


# --- Combined pie + monthly bar chart ---
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
    type_counts = build_type_weights(df[P["destination_type_std"]])
    type_counts["N/A"] = df[P["destination_type_std"]].isna().sum()
    pie_colors = (list(colors) + ["#cccccc"])[: len(type_counts)]
    pulls = (
        [0.2 if n not in ("Farmer", "N/A") else 0.0 for n in type_counts.index]
        if label == "Wastewater"
        else [0.0] * len(type_counts)
    )
    fig_combined.add_trace(
        go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            marker_colors=pie_colors,
            textposition="outside",
            textinfo="label+percent",
            textfont=dict(size=12),
            pull=pulls,
            rotation=330,
        ),
        row=1,
        col=col_idx,
    )
    monthly_amount = monthly_allocation(df, amount_col, P["haul_date_first"], P["haul_date_last"])
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


# --- Manual vs extracted accuracy comparison ---
params_to_compare = [
    "origin_dairy_address",
    "hauler_address",
    "destination_contact_address",
    "destination_address",
    "destination_nearest_cross_street",
    "destination_parcel_number",
    # "haul_date_first",
    # "haul_date_last",
    "is_pipeline",
    "manure_ton_per_haul",
    "manure_yard_per_haul",
    "manure_number_hauls",
    "manure_amount",
    "manure_density",
    "manure_solids_percent",
    "manure_moisture_percent",
]

plt_values = []
key_to_name = PARAMETERS_DF.set_index("parameter_key")["parameter_name"]
for col in [key_to_name[k] for k in params_to_compare if k in key_to_name]:
    if col not in manual_src.columns or col not in extracted_df.columns:
        continue
    manual_count = manual_src[col].notna().sum()
    extracted_count = extracted_df[col].notna().sum()
    merged_cmp = pd.merge(
        manual_src[["Source PDF", "Manifest Number", col]],
        extracted_df[["Source PDF", "Manifest Number", col]],
        on=["Source PDF", "Manifest Number"],
        how="inner",
        suffixes=("_manual", "_extracted"),
    )
    has_value = merged_cmp[f"{col}_manual"].notna() | merged_cmp[f"{col}_extracted"].notna()
    comparable = merged_cmp[has_value]
    if len(comparable) == 0:
        plt_values.append((col, manual_count, extracted_count, 0))
        continue
    m_col, e_col = f"{col}_manual", f"{col}_extracted"
    if "address" in col.lower():
        matches = (
            comparable[m_col].apply(lambda x: norm_addr(x) if isinstance(x, str) else x)
            == comparable[e_col].apply(lambda x: norm_addr(x) if isinstance(x, str) else x)
        ).fillna(False)
    elif "parcel" in col.lower():

        def norm_apns(x):
            if not isinstance(x, str):
                return x
            parts = [normalize_apn(p.strip()) for p in x.replace(",", " ").split() if p.strip()]
            return ",".join(sorted(p for p in parts if p)) or None

        matches = (comparable[m_col].apply(norm_apns) == comparable[e_col].apply(norm_apns)).fillna(
            False
        )
    else:
        matches = (comparable[m_col] == comparable[e_col]).fillna(False)
    plt_values.append((col, manual_count, extracted_count, matches.mean() * 100))

plt_values.sort(key=lambda x: x[3], reverse=True)
indices = range(len(plt_values))
bar_width = 0.35
fig_acc, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()
bars1 = ax1.bar(
    [i - bar_width / 2 for i in indices],
    [v[1] for v in plt_values],
    width=bar_width,
    label="Manual",
    alpha=0.7,
    color=PALETTE["green"],
)
bars2 = ax1.bar(
    [i + bar_width / 2 for i in indices],
    [v[2] for v in plt_values],
    width=bar_width,
    label="Extracted",
    alpha=0.7,
    color=PALETTE["orange"],
)
dots = ax2.plot(
    list(indices),
    [v[3] for v in plt_values],
    "o",
    color=PALETTE["blue"],
    markersize=8,
    label="Accuracy (%)",
    zorder=5,
)
ax1.set_xlabel("Parameters")
ax1.set_ylabel("Count of Extracted Values")
ax2.set_ylabel("Accuracy (%)")
ax1.set_xticks(list(indices))
ax1.set_xticklabels([v[0] for v in plt_values], rotation=45, ha="right")
ax1.legend(
    [bars1, bars2, dots[0]],
    ["Manual Count", "Automatic Count", "Accuracy (%)"],
    loc="upper right",
)
ax2.set_ylim(0, 100)
plt.tight_layout()
fig_acc.savefig(os.path.join(OUTPUTS_DIR, "manual_vs_extracted_comparison.png"))
for col, manual_count, extracted_count, accuracy in plt_values:
    print(
        f"  {col}: {manual_count}/{len(manual_src)} manual, "
        f"{extracted_count}/{len(extracted_df)} extracted, {accuracy:.2f}% accuracy"
    )
