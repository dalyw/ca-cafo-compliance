import streamlit as st
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import requests
from io import StringIO
import sys
from helpers_pdf_metrics import YEARS, REGIONS, cf, TEMPLATE_KEY_TO_NAME

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

st.set_page_config(
    page_title="Heaping Piles of Fraud: CA CAFO Annual Report Data Exploration",
    layout="centered",
)

with open(
    "ca_cafo_compliance/data/images/vecteezy_steaming-pile-of-manure-on-farm-field-in-dutch-countryside_8336504.jpg",
    "rb",
) as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode()


# Colorblind-friendly palette
# from ColorBrewer
PALETTE = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "yellow": "#bcbd22",
    "teal": "#17becf",
}

# Nitrogen-related colors
NITROGEN_COLOR = PALETTE["blue"]
NITROGEN_EST_COLOR = "rgba(31, 119, 180, 0.5)"  # blue 0.5 opacity

# Wastewater-related colors
WASTEWATER_COLOR = PALETTE["teal"]
WASTEWATER_EST_COLOR = "rgba(23, 190, 207, 0.5)"  # tea 0.5 opacity

# Manure-related colors
MANURE_COLOR = PALETTE["orange"]
MANURE_EST_COLOR = "rgba(255, 127, 14, 0.5)"  # orange 0.5 opacity

# Region colors
REGION_COLORS = {
    "Region 1": PALETTE["blue"],
    "Region 2": PALETTE["orange"],
    "Region 5": PALETTE["green"],
    "Region 7": PALETTE["red"],
    "Region 8": PALETTE["purple"],
}

# Chart-specific colors
CHART_COLORS = {
    "acquired": PALETTE["blue"],
    "not_acquired": PALETTE["yellow"],
    "perfect_match": PALETTE["gray"],
    "herd_breakdown": PALETTE["gray"],
    "under_reporting": PALETTE["red"],
}

# watermark background
st.markdown(
    f"""
    <div style="position: relative; width: 100%; height: 240px; margin-bottom: -250px;">
        <img src="data:image/jpeg;base64,{img_base64}"
             style="position: absolute; top: 0; left: 0; width: 100%; height: 190px;
                    object-fit: cover; opacity: 0.4; z-index: 0;" />
        <div style="position: absolute; top: 20px; left: 0; width: 100%;
                    text-align: right; z-index: 1;">
            <span style="background: rgba(255,255,255,0.7); padding: 0.2em 1em;
                        border-radius: 8px; font-size: 1em; color: #444;">
                Image: <a href='https://www.vecteezy.com/photo/8336504-steaming-pile-of-manure-on-farm-field-in-dutch-countryside'
                          target='_blank'>Vecteezy</a>
            </span>
        </div>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)


def load_data_from_source(local_path, github_url, encoding="utf-8"):
    """
    Helper function to load data from either local file or GitHub.

    Args:
        local_path (str): Path to the local file
        github_url (str): URL to the GitHub raw file
        encoding (str): File encoding (default: 'utf-8')

    Returns:
        pd.DataFrame: Loaded dataframe or empty dataframe if loading fails
    """
    if os.path.exists(local_path):
        return pd.read_csv(local_path, encoding=encoding)
    else:
        response = requests.get(github_url)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text), encoding=encoding)
        else:
            st.warning(f"Could not load {os.path.basename(local_path)} from local or GitHub.")
            return pd.DataFrame()


def load_data():
    """Load data from CSV files in the outputs/consolidated directory,
    or from GitHub."""
    # Try local files first
    csv_files = glob.glob("ca_cafo_compliance/outputs/consolidated/*.csv")

    dfs = []
    if csv_files:
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
    else:
        print("No local CSV files found. Attempting to load from GitHub...")
        base_url = (
            "https://raw.githubusercontent.com/dalywettermark/ca-cafo-compliance/"
            "main/outputs/consolidated"
        )
        files_to_load = [f"{year}_{region}_master.csv" for year in YEARS for region in REGIONS]
        for file in files_to_load:
            local_path = f"ca_cafo_compliance/outputs/consolidated/{file}"
            github_url = f"{base_url}/{file}"
            df = load_data_from_source(local_path, github_url)
            if not df.empty:
                dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)

    # Clean up Year column: drop NaN, convert to int then str, and filter out invalid years
    if "Year" in combined_df.columns:
        combined_df = combined_df[combined_df["Year"].notna()]

        def year_to_str(x):
            try:
                return str(int(float(x)))
            except Exception:
                return None

        combined_df["Year"] = combined_df["Year"].apply(year_to_str)
        combined_df = combined_df[combined_df["Year"].notna()]

    return combined_df


def add_histogram_trace(fig, data, name, color, nbinsx=50, clip_range=None):
    """Helper function to add a histogram trace to a figure."""
    if clip_range:
        data = data.clip(clip_range[0], clip_range[1])
    fig.add_trace(go.Histogram(x=data, nbinsx=nbinsx, name=name, marker_color=color, opacity=0.7))


def create_comparison_plots(df):
    """Create comparison plots between estimated and reported values."""

    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()

    # Calculate Milk Production Source
    milk_col = "Average Milk Production (lb per cow per day)"
    if milk_col in df.columns:
        df.loc[:, "Milk Production Source"] = df[milk_col].apply(
            lambda x: "Reported" if pd.notna(x) and x > 0 else "Estimated"
        )
    else:
        print(f"\nWARNING: Milk production column '{milk_col}' not found")
        df.loc[:, "Milk Production Source"] = "Estimated"

    # --- Nitrogen Deviation Plot ---
    usda_col = "USDA Nitrogen % Deviation"
    ucce_col = "UCCE Nitrogen % Deviation"

    # 1. Nitrogen Generation - Percentage Deviation Histograms
    nitrogen_fig = go.Figure()

    # Filter data for USDA nitrogen deviations
    usda_nitrogen_data = df[usda_col].dropna() if usda_col in df.columns else pd.Series()
    add_histogram_trace(
        nitrogen_fig,
        usda_nitrogen_data,
        "USDA Estimate",
        NITROGEN_EST_COLOR,
        clip_range=(-100, 100),
    )

    # Filter data for UCCE nitrogen deviations
    ucce_nitrogen_data = df[ucce_col].dropna() if ucce_col in df.columns else pd.Series()
    add_histogram_trace(
        nitrogen_fig,
        ucce_nitrogen_data,
        "UCCE Estimate",
        NITROGEN_COLOR,
        clip_range=(-100, 100),
    )

    # Add vertical line at 0% deviation if we have any data
    if not usda_nitrogen_data.empty or not ucce_nitrogen_data.empty:
        max_count = max(
            (usda_nitrogen_data.value_counts().max() if not usda_nitrogen_data.empty else 0),
            (ucce_nitrogen_data.value_counts().max() if not ucce_nitrogen_data.empty else 0),
        )

        nitrogen_fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, max_count],
                mode="lines",
                name="Perfect Match",
                line=dict(color=CHART_COLORS["perfect_match"], width=2, dash="dash"),
            )
        )

    # Update layout for nitrogen plot
    nitrogen_fig.update_layout(
        title="Nitrogen Generation - Percentage Deviation from Estimates",
        xaxis_title="Percentage Deviation",
        yaxis_title="Number of Facilities",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # 2. Wastewater Generation - Ratio to Milk Production
    wastewater_fig = go.Figure()

    # Filter data for reported wastewater ratios
    reported_mask = df["Milk Production Source"] == "Reported"
    reported_wastewater_data = (
        df.loc[reported_mask, "Wastewater to Reported Milk Ratio"].dropna()
        if "Wastewater to Reported Milk Ratio" in df.columns
        else pd.Series()
    )
    add_histogram_trace(
        wastewater_fig,
        reported_wastewater_data,
        "Based on Reported Milk",
        WASTEWATER_COLOR,
    )

    # Filter data for estimated wastewater ratios
    estimated_mask = df["Milk Production Source"] == "Estimated"
    estimated_wastewater_data = (
        df.loc[estimated_mask, "Wastewater to Estimated Milk Ratio"].dropna()
        if "Wastewater to Estimated Milk Ratio" in df.columns
        else pd.Series()
    )
    add_histogram_trace(
        wastewater_fig,
        estimated_wastewater_data,
        "Based on Estimated Milk",
        WASTEWATER_EST_COLOR,
    )

    # Add green rectangle for expected range
    if not reported_wastewater_data.empty or not estimated_wastewater_data.empty:
        wastewater_fig.add_vrect(
            x0=0,
            x1=cf["L_WW_PER_L_MILK_LOW"],
            fillcolor=CHART_COLORS["under_reporting"],
            layer="below",
            line_width=0,
            annotation_text="Likely<br>Under-Reporting",
            annotation_position="top",
        )

    # Update layout for wastewater plot
    wastewater_fig.update_layout(
        title="Wastewater Generation - Ratio to Milk Production",
        xaxis_title="Liters Wastewater per Liter Milk",
        yaxis_title="Number of Facilities",
        showlegend=True,
        barmode="stack",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=1),
    )

    return nitrogen_fig, wastewater_fig


def filter_tab2(df, selected_year):
    available_regions = ["R5", "R7"]
    selected_regions = st.multiselect(
        "Select Regions", available_regions, default=available_regions
    )

    if not selected_regions:
        return pd.DataFrame(), [], [], []

    # Use startswith to include subregions like R5,F, R5,S, etc.
    region_mask = df["Region"].astype(str).str.startswith(tuple(selected_regions))
    available_counties = sorted(df[region_mask]["County"].dropna().unique())
    selected_counties = st.multiselect(
        "Select Counties", available_counties, default=available_counties
    )

    available_consultants = sorted(df[region_mask]["Template"].dropna().unique())
    selected_consultants = st.multiselect(
        "Select Consultants", available_consultants, default=available_consultants
    )

    # filtering logic
    year_mask = df["Year"] == selected_year
    region_mask = df["Region"].astype(str).str.startswith(tuple(selected_regions))
    county_mask = df["County"].isin(selected_counties)
    if selected_consultants:
        consultant_mask = df["Template"].isin(selected_consultants)
    else:
        consultant_mask = True  # Do not filter by consultant if none selected

    filtered_df = df[year_mask & region_mask & county_mask & consultant_mask]

    return filtered_df, selected_regions, selected_counties, selected_consultants


def manure_scatter_from_df(xlim=20000):
    csv_files = glob.glob("ca_cafo_compliance/outputs/consolidated/*.csv")
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Total Herd Size"],
            y=df["Total Manure Excreted (tons)"],
            mode="markers",
            marker=dict(size=10, color=MANURE_COLOR, opacity=0.7),
            name="Facilities",
        )
    )
    max_herd = min(df["Total Herd Size"].max(), xlim)
    fig.add_trace(
        go.Scatter(
            x=[0, max_herd],
            y=[0, max_herd * 20],
            mode="lines",
            line=dict(color=CHART_COLORS["perfect_match"], width=2, dash="dash"),
            name="20 tons/cow/year",
        )
    )
    df["expected_manure"] = df["Total Herd Size"] * 12
    df["deviation"] = (df["Total Manure Excreted (tons)"] - df["expected_manure"]) / df[
        "expected_manure"
    ]
    bottom_points = df[df["Total Herd Size"] >= 5000].nsmallest(0, "deviation")
    for _, row in bottom_points.iterrows():
        fig.add_annotation(
            x=row["Total Herd Size"],
            y=row["Total Manure Excreted (tons)"],
            text=row["Dairy Name"],
            showarrow=True,
            arrowhead=1,
            ax=-100,  # left of the point
            ay=-40,  # above the point
        )
    fig.update_layout(
        title="Total Manure Excreted vs Herd Size",
        xaxis_title="Total Herd Size",
        yaxis_title="Total Manure Excreted (tons)",
        font=dict(size=20),
        title_font=dict(size=28),
        xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20), range=[0, xlim]),
        yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=18)),
    )
    return fig


def main():
    st.title("Heaping Piles of Fraud")
    st.markdown("""
    ### Revealing Dairy CAFO Compliance and Data Discrepancies
    This dashboard presents the first public analysis of annual reports from Dairy CAFOs (Concentrated Animal Feeding Operations) and reveals concerns about manure and wastewater-related reporting.
    This data shows what local community members have long known: that CAFO dairies are lying and not prioritizing public health.
    """)

    df = load_data()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Herd Size",
            "Manure Manifests",
            "Nutrients and Wastewater",
            "Enforcement",
            "Data Availability & Sources",
        ]
    )

    with tab1:
        st.write("""
        This section includes maps showing the geographic distribution
        of CAFO facilities and their herd sizes.
        """)

        # Years filter (single select)
        years = sorted(df["Year"].unique())
        default_year_index = years.index("2023") if "2023" in years else len(years) - 1
        selected_year = st.selectbox(
            "Select Year", years, index=default_year_index, key="map_year"
        )

        # Filter data for map (only by year)
        map_df = df[df["Year"] == selected_year].copy()

        # Display map
        st.subheader("Facility Locations")
        if not map_df.empty and "Latitude" in map_df.columns:
            st.metric("Total Animals", f"{map_df['Total Herd Size'].sum():,.0f}")
            # Filter for year and valid coordinates and herd size
            year_df = df[df["Year"].astype(str) == str(selected_year)].copy()
            map_df = year_df[
                year_df["Latitude"].notna()
                & year_df["Longitude"].notna()
                & year_df["Total Herd Size"].notna()
            ].copy()

            map_fig = px.scatter_map(
                map_df,
                lat="Latitude",
                lon="Longitude",
                size="Total Herd Size",
                color="Region",
                color_discrete_map=REGION_COLORS,
                hover_name="Dairy Name",
                hover_data={"Total Herd Size": True},
                title=f"CAFO Facilities in California ({selected_year})",
                size_max=20,
                zoom=4.0,
                center={"lat": 37.2719, "lon": -119.2702},
            )

            st.plotly_chart(map_fig, use_container_width=True, height=1000)
        else:
            st.warning("No location data available for the selected year.")

        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)

        # Facility search and comparison
        st.subheader("Facility Search")
        st.write("""
        Search for specific facilities to examine their reporting patterns in detail.
        This tool helps identify individual cases of potential noncompliance.
        """)

        # county filter for facility search
        facility_counties = sorted(map_df["County"].dropna().unique())
        selected_facility_county = st.selectbox(
            "Filter by County",
            ["All Counties"] + list(facility_counties),
            key="facility_county_tab1",
        )
        # Filter facilities by selected county
        if selected_facility_county != "All Counties":
            facility_df = map_df[map_df["County"] == selected_facility_county].copy()
        else:
            facility_df = map_df.copy()

        # Convert Dairy Name to string and handle NaN values
        facility_df.loc[:, "Dairy Name"] = facility_df["Dairy Name"].fillna("Unknown").astype(str)
        facility_names = sorted(facility_df["Dairy Name"].unique())
        selected_facility = st.selectbox(
            "Select a Facility",
            facility_names,
            index=(
                facility_names.index("AJ Slenders Dairy")
                if "AJ Slenders Dairy" in facility_names
                else 0
            ),
            key="facility_name_tab1",
        )
        if selected_facility:  # Get facility details
            facility_data = facility_df[facility_df["Dairy Name"] == selected_facility].iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                st.write(
                    f"Address: {facility_data['Dairy Address']} {facility_data['City']}, CA {facility_data['Zip']}"
                )
            with col2:
                template_key = facility_data["Template"]
                template_name = TEMPLATE_KEY_TO_NAME.get(template_key, template_key)
                st.write(f"Report prepared by: {template_name}")

            herd_cols = [
                "Average Milk Cows",
                "Average Dry Cows",
                "Average Bred Heifers",
                "Average Heifers",
                "Average Calves (4-6 mo.)",
                "Average Calves (0-3 mo.)",
                "Average Other",
            ]

            facility_data = facility_df[facility_df["Dairy Name"] == selected_facility].iloc[0]
            facility_fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Herd Breakdown",
                    "Nitrogen Generation",
                    "Wastewater Generation",
                    "Manure Generation",
                ),
            )

            # 1. Herd Breakdown
            herd_data = []
            for col in herd_cols:
                if col in facility_data and pd.notna(facility_data[col]):
                    herd_data.append({"name": col, "value": facility_data[col]})
            if herd_data:
                facility_fig.add_trace(
                    go.Bar(
                        x=[d["name"].replace("Average ", "") for d in herd_data],
                        y=[d["value"] for d in herd_data],
                        name="Herd Breakdown",
                        marker_color="gray",
                        text=[f"{d['value']:,.0f}" for d in herd_data],
                        textposition="auto",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

            # 2. Nitrogen Generation
            n_reported = "Total Reported N (lbs)"
            n_usda = "USDA N Estimate (lbs)"
            n_ucce = "UCCE N Estimate (lbs)"
            n_reported_val = (
                facility_data[n_reported]
                if n_reported in facility_data and pd.notna(facility_data[n_reported])
                else None
            )
            n_usda_val = (
                facility_data[n_usda]
                if n_usda in facility_data and pd.notna(facility_data[n_usda])
                else None
            )
            n_ucce_val = (
                facility_data[n_ucce]
                if n_ucce in facility_data and pd.notna(facility_data[n_ucce])
                else None
            )

            def _add_bar_trace(x, y, color, shape="", text=None, row=None, col=None):
                """Helper function to add a bar trace to a figure."""
                legendgroup = "Reported" if "report" in x.lower() else "Estimated"
                facility_fig.add_trace(
                    go.Bar(
                        x=[x],
                        y=[y],
                        name=x,
                        marker_color=color,
                        marker_pattern_shape=shape,
                        text=[text] if text else None,
                        legendgroup=legendgroup,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            _add_bar_trace(
                "Reported",
                n_reported_val,
                NITROGEN_COLOR,
                text=f"{n_reported_val:,.0f} lbs" if n_reported_val else None,
                row=1,
                col=2,
            )
            _add_bar_trace(
                "Estimated (USDA)",
                n_usda_val,
                NITROGEN_EST_COLOR,
                shape="/",
                text=f"{n_usda_val:,.0f} lbs" if n_usda_val else None,
                row=1,
                col=2,
            )
            _add_bar_trace(
                "Estimated (UCCE)",
                n_ucce_val,
                NITROGEN_EST_COLOR,
                shape="/",
                text=f"{n_ucce_val:,.0f} lbs" if n_ucce_val else None,
                row=1,
                col=2,
            )

            # 3. Wastewater Generation
            ww_reported = "Total Wastewater Generated (L)"
            ww_reported_val = (
                facility_data[ww_reported]
                if ww_reported in facility_data and pd.notna(facility_data[ww_reported])
                else None
            )
            ww_estimated_val = facility_data["Estimated Total Wastewater Generated (L)"]

            _add_bar_trace(
                "Reported",
                ww_reported_val,
                WASTEWATER_COLOR,
                text=f"{ww_reported_val:,.0f} L" if ww_reported_val else None,
                row=2,
                col=1,
            )
            _add_bar_trace(
                "Estimated",
                ww_estimated_val,
                WASTEWATER_EST_COLOR,
                shape="/",
                text=f"{ww_estimated_val:,.0f} L" if ww_estimated_val else None,
                row=2,
                col=1,
            )

            # 4. Manure Generation
            manure_reported = "Total Manure Excreted (tons)"
            herd_size_col = "Total Herd Size"
            manure_reported_val = (
                facility_data[manure_reported]
                if manure_reported in facility_data and pd.notna(facility_data[manure_reported])
                else None
            )
            manure_estimated_val = (
                cf["MANURE_FACTOR_AVERAGE"] * facility_data[herd_size_col]
                if herd_size_col in facility_data and pd.notna(facility_data[herd_size_col])
                else None
            )

            _add_bar_trace(
                "Reported",
                manure_reported_val,
                MANURE_COLOR,
                text=(f"{manure_reported_val:,.0f} tons" if manure_reported_val else None),
                row=2,
                col=2,
            )
            _add_bar_trace(
                "Estimated",
                manure_estimated_val,
                MANURE_EST_COLOR,
                shape="/",
                text=(f"{manure_estimated_val:,.0f} tons" if manure_estimated_val else None),
                row=2,
                col=2,
            )

            facility_fig.update_layout(
                height=700,
                font=dict(size=22),
            )
            facility_fig.update_xaxes(title_font=dict(size=22), tickfont=dict(size=20))
            facility_fig.update_yaxes(title_font=dict(size=22), tickfont=dict(size=20))
            facility_fig.update_xaxes(tickangle=30, row=1, col=1)

            if facility_fig is not None:
                st.plotly_chart(facility_fig, use_container_width=True)
            else:
                st.warning(f"No data available for {selected_facility}")

        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)

    with tab2:
        st.write("""
        This section visualizes the movement of manure exports throughout the Central Valley region,
        revealing the flow of nutrients and potential environmental impacts beyond facility boundaries.
        """)

        st.image(
            "ca_cafo_compliance/data/images/manifest_placeholder.png",
            caption="Manure manifest showing export destinations and volumes (from Sophia)",
        )

        st.subheader("CAFO Density around Elementary Schools - example embed")
        st.components.v1.iframe(
            "https://www.arcgis.com/apps/webappviewer/index.html?id=a247a569c9854bb89689bebb01f5eee4",
            height=600,
            scrolling=True,
        )

        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)

    with tab3:
        st.write("""
        This section focuses on the reported manure, nitrogen and wastewater production in Regions 5 and 7, where we've identified significant variations between reported and estimated values.
        """)

        # Years filter
        years = sorted(df["Year"].unique())
        default_year_index = years.index("2023") if "2023" in years else len(years) - 1
        selected_year_tab3 = st.selectbox(
            "Select Year", years, index=default_year_index, key="plot_year"
        )

        # Filter data for plots
        (
            filtered_df,
            selected_regions,
            selected_counties,
            selected_consultants,
        ) = filter_tab2(df, selected_year_tab3)

        if filtered_df.empty:
            st.warning("Please select at least one region (R5 or R7) to view comparison plots.")
            return

        # Comparison plots with explanations
        st.subheader("Estimated vs Actual Comparisons")

        # Nitrogen Generation Plot
        st.markdown("""
        ### Nitrogen Generation Comparison
        Values above 0% indicate facilities reporting less nitrogen than estimated
        We compare reported nitrogen generation to two estimated metris. The USDA estimate is based on nitrogen per unit of manure generation. The UCCE estimate is based on nitrogen per animal unit.
        """)
        nitrogen_fig, wastewater_fig = create_comparison_plots(filtered_df)
        st.plotly_chart(nitrogen_fig, use_container_width=True)

        # Wastewater to Milk Ratio Plot
        st.markdown("""
        ### Wastewater to Milk Ratio
        Unusually low ratios may indicate under-reporting of wastewater usage.
        The ratio is calculated as: Total Process Wastewater (L) / Annual Milk Production (L). Milk production is either reported or estimated (using 68 lb/cow/day default)
        """)
        st.plotly_chart(wastewater_fig, use_container_width=True)

        # Manure Factor Plot
        st.markdown("""
        ### Manure Generation""")
        manure_fig_consolidated = manure_scatter_from_df()
        st.plotly_chart(manure_fig_consolidated, use_container_width=True)

        # Add consultant comparison plots
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("Consultant Comparison")
        st.write("""
        Many facilities in region 5 use consultatns to prepare their reports. This section assesses reporting patterns across different consultants and self-reported facilities, to understand if there are any systematic issues with certain consultants.
        Each bar represents a consultant's average value, with error bars showing the standard deviation.
        """)

        metrics_path = "ca_cafo_compliance/outputs/consolidated/2023_R5_consultant_metrics.csv"
        df = pd.read_csv(metrics_path)
        consultants = df["Template"]

        # Each entry: (metric_prefix, legend_name, color, subplot_col, fmt, y_axis_title)
        panels = [
            ("USDA Nitrogen % Deviation", "USDA Estimate", "blue", 1, ".0f", "%"),
            ("UCCE Nitrogen % Deviation", "UCCE Estimate", "lightblue", 1, ".0f", None),
            (
                "Wastewater Ratio",
                "Based on Reported Milk",
                "red",
                2,
                ".2f",
                "Liters per Liter Milk",
            ),
            ("Manure Factor", "Annual Production", "green", 3, ".0f", "Manure per Cow"),
        ]

        consultant_fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Nitrogen Deviation",
                "Wastewater to Milk",
                "Manure Factor",
            ),
            horizontal_spacing=0.15,
        )

        for metric, name, color, col, fmt, y_title in panels:
            y = df[f"{metric} Average"]
            y_std = df[f"{metric} Standard Deviation"]
            error_y = (
                None
                if y_std.isnull().all()
                else dict(type="data", array=np.nan_to_num(y_std).tolist(), visible=True)
            )
            consultant_fig.add_trace(
                go.Bar(
                    x=consultants,
                    y=y,
                    name=name,
                    marker_color=color,
                    error_y=error_y,
                    text=[format(v, fmt) if not np.isnan(v) else "" for v in y],
                    textposition="auto",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            if y_title:
                consultant_fig.update_yaxes(title_text=y_title, row=1, col=col)

        consultant_fig.update_layout(
            showlegend=False, height=500, margin=dict(l=40, r=40, t=80, b=40)
        )
        st.plotly_chart(consultant_fig, use_container_width=True)

        # Raw data
        st.subheader("Raw Data")
        st.write("""
        View and downloada the complete dataset for detailed analysis.
        Questions on the data can be directed to (insert email)
        """)

        display_df = filtered_df.copy()
        # Ensure Zip is always a string for display and export
        if "Zip" in display_df.columns:
            display_df["Zip"] = display_df["Zip"].astype(str)
        st.dataframe(display_df)

        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="filtered_cafo_data.csv",
            mime="text/csv",
        )

    with tab4:  # Violation Summary
        st.markdown("""
        **Summary of Violations by Region and Type**
        Most violations issued by the Water Boards are for paperwork and reporting issues
        (such as late or missing reports), not for actual non-compliance with nutrient
        management or environmental protection. This is despite the fact that many submitted
        reports show clear evidence of over-application of manure and nitrogen, which can
        lead to water quality violations and environmental harm.
        The table and chart below summarize the types of violations recorded in the enforcement data.
        """)

        # Load and summarize violation data
        violations_path = "ca_cafo_compliance/data/Detailed_Violation_Report.csv"
        github_url = "https://raw.githubusercontent.com/dalywettermark/ca-cafo-compliance/main/data/Detailed_Violation_Report.csv"
        vdf = load_data_from_source(violations_path, github_url, encoding="latin1")

        # Clean up region and type columns, and map RB for display
        vdf["RB"] = vdf["RB"].astype(str)
        vdf["Violation Type"] = vdf["Violation Type"].astype(str)
        rb_map = {
            "5F": "R5-F",
            "5S": "R5-S",
            "5R": "R5-R",
            "1": "R1",
            "2": "R2",
            "3": "R3",
            "6V": "R6V",
            "7": "R7",
            "8": "R8",
            "9": "R9",
            "6B": "R6B",
        }
        vdf["Region"] = vdf["RB"].map(rb_map).fillna(vdf["RB"])
        summary = vdf.groupby(["Region", "Violation Type"]).size().reset_index(name="Count")
        summary_pivot = (
            summary.pivot(index="Region", columns="Violation Type", values="Count")
            .fillna(0)
            .astype(int)
        )
        summary_pivot = summary_pivot.reindex(
            sorted(
                summary_pivot.index,
                key=lambda x: (x not in ["R5-F", "R5-S", "R5-R"], x),
            )
        )
        st.dataframe(summary_pivot)

        # Bar chart of violation types by region (show R5-F, R5-S, R5-R as separate bars)
        bar_df = summary.copy()
        bar_df["Region"] = pd.Categorical(
            bar_df["Region"],
            categories=[
                "R5-F",
                "R5-S",
                "R5-R",
                "R1",
                "R2",
                "R3",
                "R6V",
                "R6B",
                "R7",
                "R8",
                "R9",
            ],
            ordered=True,
        )
        fig = px.bar(
            bar_df,
            x="Region",
            y="Count",
            color="Violation Type",
            barmode="stack",
            title="Violations by Region and Type",
            labels={"Region": "Region", "Count": "Number of Violations"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.write("""
        This section provides information about the data availability and types for each region,
        based on the provided text. This analysis is inherently limited by the accessibility
        and consistency of the source data, including issues like inconsistent regional
        reporting formats and levels of detail, the requirement to visit in-person to get
        data in some regions, and different data collection periods
        """)

        # Reporting Requirements Table
        def reporting_requirements_table():
            data = [
                ["1", "❌", "❌", "❌", "❌", "❌", "❌", "❌"],
                ["2", "❌", "❌", "❌", "✅", "❌", "❌", "❌"],
                ["5", "✅", "✅", "✅", "❌", "❌", "✅", "❌"],
                ["7", "✅", "✅", "❌", "❌", "🟡", "❌", "❌"],
                ["8", "❌", "✅", "❌", "❌", "🟡", "✅", "✅"],
            ]
            columns = [
                "Region",
                "Wastewater Production Reporting",
                "Manure Production Reporting",
                "Nutrient Content Reporting",
                "Groundwater Sampling Required",
                "Public access to NMPs?",
                "Require Manure Tracking Manifests?",
                "Provides Tabular Data?",
            ]
            df = pd.DataFrame(data, columns=columns)

            def style_cell(val):
                if val == "✅":
                    return '<span style="color:green;font-size:1.5em;">&#x2705;</span>'
                elif val == "❌":
                    return '<span style="color:red;font-size:1.5em;">&#10060;</span>'
                elif val == "🟡":
                    return "<b>Limited</b>"
                else:
                    return val

            styled_df = df.style.format(style_cell, escape="html")
            st.markdown("### CAFO Reporting Requirements by Region")
            st.write(
                "The table below summarizes reporting and public access requirements for each region."
            )
            st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

        # Show the table instead of the image
        reporting_requirements_table()

        csv_path = "ca_cafo_compliance/data/reports_available.csv"
        github_url = "https://raw.githubusercontent.com/dalywettermark/ca-cafo-compliance/main/data/reports_available.csv"
        reports_df = load_data_from_source(csv_path, github_url)

        # Region/county mapping for labels, accounting for sub-regions of R5
        def get_region_label(row):
            region = str(row.get("region", ""))
            county = str(row.get("county", "")).lower() if "county" in row else ""
            if region == "5":
                if county == "kern":
                    return "R5-F"
                elif county in ["fresno_madera", "kings", "tulare_west"]:
                    return "R5-S"
                else:
                    return "R5-R"
            region_map = {
                "R1": "R1",
                "R2": "R2",
                "R3": "R3",
                "R6V": "R6V",
                "R7": "R7",
                "R8": "R8",
                "R9": "R9",
            }
            return region_map.get(region, region)

        # If county column is not present, infer from region key (for legacy CSVs)
        if "county" not in reports_df.columns:
            # For legacy, use region key directly for 5F, 5S, 5R
            def legacy_label(region):
                if region == "5F":
                    return "R5-F"
                elif region == "5S":
                    return "R5-S"
                elif region == "5R":
                    return "R5-R"
                region_map = {
                    "1": "R1",
                    "2": "R2",
                    "3": "R3",
                    "6V": "R6V",
                    "7": "R7",
                    "8": "R8",
                    "9": "R9",
                }
                return region_map.get(region, region)

            reports_df["region_label"] = reports_df["region"].apply(legacy_label)
        else:
            reports_df["region_label"] = reports_df.apply(get_region_label, axis=1)

        available_regions = reports_df["region_label"].unique().tolist()
        selected_regions = st.multiselect(
            "Select Regions to Display",
            available_regions,
            default=available_regions,
        )
        filtered_df = reports_df[reports_df["region_label"].isin(selected_regions)]

        # Calculate totals for pie chart
        acquired = pd.to_numeric(filtered_df["acquired"], errors="coerce").fillna(0).sum()
        total = pd.to_numeric(filtered_df["total"], errors="coerce").fillna(0).sum()
        not_acquired = total - acquired

        # Pie chart
        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Acquired", "Not Acquired"],
                    values=[acquired, not_acquired],
                    marker=dict(
                        colors=[
                            CHART_COLORS["acquired"],
                            CHART_COLORS["not_acquired"],
                        ]
                    ),
                    textinfo="label+percent",
                    hole=0.2,
                )
            ]
        )
        pie_fig.update_traces(textfont_size=16)
        pie_fig.update_layout(title="Annual Reports Acquired Across Selected Regions")
        st.plotly_chart(pie_fig, use_container_width=True)

        st.subheader("R-1 North Coast")
        st.markdown("""
        Annual reports under Order No. R1-2019-0001 requested by emailing the R2 Water Board and transferred via email.
        - Waste discharge requirements documentation
        """)

        st.subheader("R-2 San Francisco Bay")
        st.markdown("""
        Annual reports under Order R2-2016-0031 requested by emailing the R2 Water Board and transferred via email.
        - Facility information and animal counts
        - Certification of facility monitoring programs, waste management plans, grazing management plans, and nutrient management plans
        - Pre-rainy season pollution prevention inspection documentation
        - Groundwater sampling data (when provided) or indication of group monitoring program participation
        """)

        st.subheader("R-5 Central Valley")
        st.markdown("""
        Annual reports under General Order No. R5-2007-0035 requested by emailing the Central Valley Water Board and transferred through their Transfer Portal.
        - Animal counts
        - Manure production with nutrient breakdown
        - Wastewater production
        - Groundwater reporting
        - Stormwater reporting
        - Manure tracking manifests
        - Laboratory analyses of discharges
        """)

        st.subheader("R-7 Colorado River Basin")
        st.markdown("""
        Annual reports under Order R7-2021-0029 requested by emailing the R7 Water Board and transferred through their Transfer Portal.
        - Animal counts
        - Composting inventory
        - Land application of manure, litter, and process wastewater report
        - Groundwater monitoring report
        - Certification
        """)

        st.subheader("R-8 Santa Ana")
        st.markdown("""
        Annual reports under Order No. R8-2018-0001 requested by emailing the R8 Water Board and transferred through their Transfer Portal.
        The reports are still available for download as of May 2025.
        https://ftp.waterboards.ca.gov/WebInterface/login.html?path=/CAFO%202023%20Annual%20Reports/
        Username: rb8download
        Password: Region8_public
        - Summary Report of Weekly Storm Water Management Structure Inspections (Form 2)
        - Annual Report Form (Form 3) with facility information and animal population data
        - Manure Tracking Manifests (Form 4)
        - CSV files with farm population and total manure hauled data
        """)


if __name__ == "__main__":
    main()
