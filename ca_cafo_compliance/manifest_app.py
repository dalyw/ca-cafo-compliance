import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

from app import load_data_from_source

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

st.set_page_config(
    page_title="Manifest Destination Maps",
    layout="wide",
)


def load_manifests() -> pd.DataFrame:
    """Load combined manure + wastewater manifests from local CSVs or GitHub."""
    base_local = "ca_cafo_compliance/outputs"
    base_github = (
        "https://raw.githubusercontent.com/dalywettermark/ca-cafo-compliance/"
        "main/outputs"
    )

    files = [
        ("2024_manifests_manure.csv",),
        ("2024_manifests_wastewater.csv",),
    ]

    dfs = []
    for (name,) in files:
        local_path = os.path.join(base_local, name)
        github_url = f"{base_github}/{name}"
        df = load_data_from_source(local_path, github_url)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        st.error("Could not load manifest CSVs from local files or GitHub.")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def make_origin_map(df: pd.DataFrame):
    if not {"Origin Latitude", "Origin Longitude"}.issubset(df.columns):
        st.error("Origin latitude/longitude columns not found in manifest data.")
        return

    fig = px.scatter_mapbox(
        df,
        lat="Origin Latitude",
        lon="Origin Longitude",
        color="Manifest Type",
        hover_name="Origin Dairy Name",
        hover_data=[
            "Origin Dairy Address",
            "Source PDF",
            "Manifest Number",
        ],
        zoom=5,
        center={"lat": 37.0, "lon": -120.0},
        height=700,
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Manifest Type",
    )
    st.subheader("Origin Dairy Address")
    st.plotly_chart(fig, use_container_width=True)


def make_destination_map(df: pd.DataFrame):
    if not {"Destination Latitude (Geocoded)", "Destination Longitude (Geocoded)"}.issubset(
        df.columns
    ):
        st.error("Destination latitude/longitude columns not found in manifest data.")
        return

    fig = px.scatter_mapbox(
        df,
        lat="Destination Latitude (Geocoded)",
        lon="Destination Longitude (Geocoded)",
        color="Manifest Type",
        hover_name="Destination Name",
        hover_data=[
            "Destination Address Final",
            "Destination County",
            "Source PDF",
            "Manifest Number",
        ],
        zoom=5,
        center={"lat": 37.0, "lon": -120.0},
        height=700,
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Manifest Type",
    )
    st.subheader("Destination Address")
    st.plotly_chart(fig, use_container_width=True)


def main():
    df = load_manifests()
    if df.empty:
        return

    make_origin_map(df)
    make_destination_map(df)


if __name__ == "__main__":
    main()
