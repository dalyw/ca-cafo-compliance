import streamlit as st
import base64
st.set_page_config(page_title="Heaping Piles of Fraud: CA CAFO Annual Report Data Exploration", layout="centered")
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import geopandas as gpd
from conversion_factors import *

# Streamlit generated and modified with prompts to Claude 3.7, 

with open("ca_cafo_compliance/vecteezy_steaming-pile-of-manure-on-farm-field-in-dutch-countryside_8336504.jpg", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode()

# Add watermark background at the top of the app (no <h1> here)
st.markdown(
    f'''
    <div style="position: relative; width: 100%; height: 260px; margin-bottom: -120px;">
        <img src="data:image/jpeg;base64,{img_base64}" 
             style="position: absolute; top: 0; left: 0; width: 100%; height: 260px; object-fit: cover; opacity: 0.4; filter: blur(1px); z-index: 0;" />
        <div style="position: absolute; top: 20px; left: 0; width: 100%; text-align: center; z-index: 1;">
            <span style="background: rgba(255,255,255,0.7); padding: 0.2em 1em; border-radius: 8px; font-size: 1em; color: #444;">Image: <a href='https://www.vecteezy.com/photo/8336504-steaming-pile-of-manure-on-farm-field-in-dutch-countryside' target='_blank'>Vecteezy</a></span>
        </div>
    </div>
    <br>
    ''',
    unsafe_allow_html=True
)

def convert_columns(df):
    categorical_cols = ['Year', 'Region', 'County', 'Template', 'Consultant', 'Dairy Name', 'Dairy Address', 'filename']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    numeric_cols = [col for col in df.columns if col not in categorical_cols]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_data():
    print("\n=== Debug: Loading Data ===")
    all_files = glob.glob("outputs/consolidated/*.csv")
    dfs = []
    if not all_files:
        try:
            import requests
            from io import StringIO
            print('getting data from github')
            base_url = "https://raw.githubusercontent.com/dalywettermark/ca-cafo-compliance/main/outputs/consolidated"
            files_to_load = [f"{year}_{region}_master.csv" for year in YEARS for region in REGIONS]
            for file in files_to_load:
                try:
                    url = f"{base_url}/{file}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        df = pd.read_csv(StringIO(response.text))
                        df['Year'] = file.split('_')[0]
                        dfs.append(convert_columns(df))
                except Exception as e:
                    st.warning(f"Error loading {file} from GitHub: {str(e)}")
            if not dfs:
                st.error("No data could be loaded from GitHub. Please check the repository URL and file paths.")
                st.stop()
        except ImportError:
            st.error("Could not load data from GitHub.")
            st.stop()
    else:
        for file in all_files:
            try:
                print(f"Loading file: {file}")
                df = pd.read_csv(file)
                df['Year'] = os.path.basename(file).split('_')[0]
                dfs.append(convert_columns(df))
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
        if not dfs:
            st.error("No data could be loaded.")
            st.stop()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows after combining: {len(combined_df)}")
    
    if 'Consultant' not in combined_df.columns:
        combined_df['Consultant'] = combined_df['Template'].map(consultant_mapping).fillna('Unknown')
    else:
        mask = combined_df['Consultant'].isna() | (combined_df['Consultant'] == 'nan')
        combined_df.loc[mask, 'Consultant'] = combined_df.loc[mask, 'Template'].map(consultant_mapping).fillna('Unknown')
    
    # Ensure consistent column names
    column_mapping = {
        'usda_nitrogen_pct_deviation': 'USDA Nitrogen % Deviation',
        'ucce_nitrogen_pct_deviation': 'UCCE Nitrogen % Deviation',
        'ratio_ww_to_milk_l_per_l': 'Ratio of Wastewater to Milk (L/L)',
        'calculated_manure_factor': 'Calculated Manure Factor'
    }
    
    # Print original column names
    print("\nOriginal columns before renaming:")
    for old_col, new_col in column_mapping.items():
        if old_col in combined_df.columns:
            print(f"{old_col}: {combined_df[old_col].notna().sum()} non-null values")
    
    # Rename columns if they exist in the original format
    for old_col, new_col in column_mapping.items():
        if old_col in combined_df.columns:
            combined_df[new_col] = combined_df[old_col]
            print(f"Renamed {old_col} to {new_col}")
    
    # Print final column names and their non-null counts
    print("\nFinal columns after processing:")
    for col in ['USDA Nitrogen % Dev Avg', 'UCCE Nitrogen % Dev Avg', 'Wastewater Ratio Avg', 'Manure Factor Avg']:
        if col in combined_df.columns:
            print(f"{col}: {combined_df[col].notna().sum()} non-null values")
            if combined_df[col].notna().sum() > 0:
                print(f"Sample values: {combined_df[col].dropna().head().tolist()}")
    
    combined_df['Year'] = combined_df['Year'].astype(str)
    print("\n=== End Debug: Loading Data ===\n")
    return combined_df

def create_map(df, selected_year):
    """Create a map visualization of facilities with regional board boundaries."""
    # Convert both to strings for comparison
    year_df = df[df['Year'].astype(str) == str(selected_year)]
    
    # Filter out rows with NaN coordinates or herd size
    map_df = year_df[
        year_df['Latitude'].notna() & 
        year_df['Longitude'].notna() & 
        year_df['Total Herd Size'].notna()
    ]
    
    if map_df.empty:
        st.warning("No valid location data available for the selected year.")
        return None
    
    # Define a distinct color palette for regions
    color_discrete_map = {
        'Region 1': '#1f77b4',  # Blue
        'Region 2': '#ff7f0e',  # Orange
        'Region 5': '#2ca02c',  # Green
        'Region 7': '#d62728',  # Red
        'Region 8': '#9467bd'   # Purple
    }
    
    # Create the base map with facilities
    fig = px.scatter_map(
        map_df,
        lat='Latitude',
        lon='Longitude',
        size='Total Herd Size',
        color='Region',
        color_discrete_map=color_discrete_map,
        hover_name='Dairy Name',
        hover_data={
            'Total Herd Size': True,
            'Region': False,
            'Latitude': False,
            'Longitude': False
        },
        title=f'CAFO Facilities in California ({selected_year})',
        size_max=30,
        zoom=5.5,
        center={"lat": 37.2719, "lon": -119.2702}
    )

    return fig

def create_comparison_plots(df):
    """Create comparison plots between estimated and reported values."""
    print("\n=== Debug Information for Plots ===")
    print(f"Total rows in dataframe: {len(df)}")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Color scheme for consistency
    estimated_color = 'rgb(135, 206, 235)'  # light blue
    reported_color = 'rgb(0, 71, 171)'      # dark blue
    ucce_color = 'rgb(144, 238, 144)'       # light green
    base_color = 'rgb(255, 165, 0)'         # orange for base values
    
    # Calculate Milk Production Source
    milk_col = 'Average Milk Production (lb per cow per day)'
    if milk_col in df.columns:
        df.loc[:, 'Milk Production Source'] = df[milk_col].apply(
            lambda x: 'Reported' if pd.notna(x) and x > 0 else 'Estimated'
        )
        print(f"\nMilk Production Source distribution:")
        print(df['Milk Production Source'].value_counts())
    else:
        print(f"\nWARNING: Milk production column '{milk_col}' not found")
        df.loc[:, 'Milk Production Source'] = 'Estimated'
    
    # 1. Nitrogen Generation - Percentage Deviation Histograms
    nitrogen_fig = go.Figure()
    
    # Use the correct column names directly
    usda_col = 'USDA Nitrogen % Dev Avg'
    ucce_col = 'UCCE Nitrogen % Dev Avg'
    
    print("\n=== Nitrogen Generation Plot ===")
    print(f"Looking for columns: {usda_col}, {ucce_col}")
    
    # Filter data for USDA nitrogen deviations
    usda_nitrogen_data = df[usda_col].dropna() if usda_col in df.columns else pd.Series()
    print(f"USDA Nitrogen data points: {len(usda_nitrogen_data)}")
    if not usda_nitrogen_data.empty:
        print(f"USDA Nitrogen data range: {usda_nitrogen_data.min():.2f} to {usda_nitrogen_data.max():.2f}")
        nitrogen_fig.add_trace(
            go.Histogram(
                x=usda_nitrogen_data,
                nbinsx=50,
                name="USDA Estimate",
                marker_color=estimated_color,
                opacity=0.7
            )
        )
    else:
        print("WARNING: No USDA Nitrogen data available")
    
    # Filter data for UCCE nitrogen deviations
    ucce_nitrogen_data = df[ucce_col].dropna() if ucce_col in df.columns else pd.Series()
    print(f"UCCE Nitrogen data points: {len(ucce_nitrogen_data)}")
    if not ucce_nitrogen_data.empty:
        print(f"UCCE Nitrogen data range: {ucce_nitrogen_data.min():.2f} to {ucce_nitrogen_data.max():.2f}")
        nitrogen_fig.add_trace(
            go.Histogram(
                x=ucce_nitrogen_data,
                nbinsx=50,
                name="UCCE Estimate",
                marker_color=ucce_color,
                opacity=0.7
            )
        )
    else:
        print("WARNING: No UCCE Nitrogen data available")
    
    # Add vertical line at 0% deviation if we have any data
    if not usda_nitrogen_data.empty or not ucce_nitrogen_data.empty:
        max_count = max(
            usda_nitrogen_data.value_counts().max() if not usda_nitrogen_data.empty else 0,
            ucce_nitrogen_data.value_counts().max() if not ucce_nitrogen_data.empty else 0
        )
        nitrogen_fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, max_count],
                mode='lines',
                name='Perfect Match',
                line=dict(color='black', width=2, dash='dash')
            )
        )
    
    nitrogen_fig.update_layout(
        title="Distribution of Nitrogen Estimate Deviations",
        xaxis_title="Percentage Deviation from Reported Value",
        yaxis_title="Number of Facilities",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            xanchor="right",
            x=1.02,
            y=0.5
        ),
        barmode='overlay'  # Overlay the histograms
    )
    
    # 2. Wastewater to Milk Ratio - Histogram
    wastewater_fig = go.Figure()
    
    # Use the correct column name directly
    ratio_col = 'Wastewater Ratio Avg'
    
    print("\n=== Wastewater to Milk Ratio Plot ===")
    print(f"Looking for column: {ratio_col}")
    
    # Filter data for reported wastewater ratios
    reported_mask = df['Milk Production Source'] == 'Reported'
    reported_wastewater_data = df.loc[reported_mask, ratio_col].dropna() if ratio_col in df.columns else pd.Series()
    print(f"Reported wastewater data points: {len(reported_wastewater_data)}")
    if not reported_wastewater_data.empty:
        print(f"Reported wastewater data range: {reported_wastewater_data.min():.2f} to {reported_wastewater_data.max():.2f}")
        wastewater_fig.add_trace(
            go.Histogram(
                x=reported_wastewater_data,
                nbinsx=50,
                name="Based on Reported Milk",
                marker_color=reported_color,
                opacity=0.7
            )
        )
    else:
        print("WARNING: No reported wastewater data available")
    
    # Filter data for estimated wastewater ratios
    estimated_mask = df['Milk Production Source'] == 'Estimated'
    estimated_wastewater_data = df.loc[estimated_mask, ratio_col].dropna() if ratio_col in df.columns else pd.Series()
    print(f"Estimated wastewater data points: {len(estimated_wastewater_data)}")
    if not estimated_wastewater_data.empty:
        print(f"Estimated wastewater data range: {estimated_wastewater_data.min():.2f} to {estimated_wastewater_data.max():.2f}")
        wastewater_fig.add_trace(
            go.Histogram(
                x=estimated_wastewater_data,
                nbinsx=50,
                name="Based on Estimated Milk",
                marker_color=estimated_color,
                opacity=0.7
            )
        )
    else:
        print("WARNING: No estimated wastewater data available")
    
    # Add vertical line for estimated ratio if we have any data
    if not reported_wastewater_data.empty or not estimated_wastewater_data.empty:
        max_count = max(
            reported_wastewater_data.value_counts().max() if not reported_wastewater_data.empty else 0,
            estimated_wastewater_data.value_counts().max() if not estimated_wastewater_data.empty else 0
        )
        avg_ratio = df['Wastewater to Milk Ratio'].mean()
        if not pd.isna(avg_ratio):
            print(f"Average wastewater ratio: {avg_ratio:.2f}")
            wastewater_fig.add_trace(
                go.Scatter(
                    x=[avg_ratio] * 2,
                    y=[0, max_count],
                    mode='lines',
                    name='Average Estimated Ratio',
                    line=dict(color=base_color, width=2, dash='dash')
                )
            )
        else:
            print("WARNING: Average wastewater ratio is NaN")
    
    wastewater_fig.update_layout(
        title="Distribution of Wastewater to Milk Ratios",
        xaxis_title="Liters Wastewater per Liter Milk",
        yaxis_title="Number of Facilities",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            xanchor="right",
            x=1.02,
            y=0.5
        ),
        barmode='stack'  # Stack the histograms
    )
    
    # 3. Manure Factor - Histogram
    manure_fig = go.Figure()
    
    # Use the correct column name directly
    manure_col = 'Manure Factor Avg'
    
    print("\n=== Manure Factor Plot ===")
    print(f"Looking for column: {manure_col}")
    
    # Filter data for manure factors
    manure_data = df[manure_col].dropna() if manure_col in df.columns else pd.Series()
    print(f"Manure factor data points: {len(manure_data)}")
    if not manure_data.empty:
        print(f"Manure factor data range: {manure_data.min():.2f} to {manure_data.max():.2f}")
        manure_fig.add_trace(
            go.Histogram(
                x=manure_data,
                nbinsx=50,  # Adjust number of bins as needed
                name="Calculated Factor",
                marker_color=reported_color,
                opacity=0.7
            )
        )
        
        # Add vertical line for base factor
        manure_fig.add_trace(
            go.Scatter(
                x=[BASE_MANURE_FACTOR] * 2,
                y=[0, manure_data.value_counts().max()],
                mode='lines',
                name='Base Factor',
                line=dict(color=base_color, width=2, dash='dash')
            )
        )
    else:
        print("WARNING: No manure factor data available")
    
    manure_fig.update_layout(
        title="Distribution of Manure Factors",
        xaxis_title="Tons Manure per Cow per Year",
        yaxis_title="Number of Facilities",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            xanchor="right",
            x=1.02,
            y=0.5
        )
    )
    
    print("\n=== End of Debug Information ===\n")
    return nitrogen_fig, wastewater_fig, manure_fig

def create_facility_comparison_plots(df, facility_name):
    """Create comparison plots for a specific facility."""
    facility_df = df[df['Dairy Name'] == facility_name]
    if facility_df.empty:
        return None
    
    # Color scheme for consistency
    reported_color = 'rgb(0, 71, 171)'      # dark blue
    estimated_color = 'rgb(135, 206, 235)'  # light blue
    ucce_color = 'rgb(144, 238, 144)'       # light green
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Wastewater",
            "Manure Factor",
            "Nitrogen Generation"
        )
    )
    
    # Helper function to safely get value
    def get_value(df, col, default=0):
        if col in df.columns:
            return df[col].iloc[0] if not pd.isna(df[col].iloc[0]) else default
        return default
    
    # 1. Wastewater Comparison (Reported vs Estimated)
    reported_wastewater = get_value(facility_df, 'Total Process Wastewater Generated (L)')
    estimated_wastewater = get_value(facility_df, 'Estimated Total Manure (tons)') * 1000  # Convert tons to liters
    
    if reported_wastewater > 0:
        fig.add_trace(
            go.Bar(
                name="Reported",
                x=["Reported"],
                y=[reported_wastewater],
                marker_color=reported_color,
                legendgroup="wastewater",
                showlegend=True
            ),
            row=1, col=1
        )
    if estimated_wastewater > 0:
        fig.add_trace(
            go.Bar(
                name="Estimated",
                x=["Estimated"],
                y=[estimated_wastewater],
                marker_color=estimated_color,
                legendgroup="wastewater",
                showlegend=True if reported_wastewater <= 0 else False
            ),
            row=1, col=1
        )
    
    # 2. Manure Factor Comparison (Calculated vs Base)
    calculated_manure = get_value(facility_df, 'Calculated Manure Factor')
    if calculated_manure > 0:
        fig.add_trace(
            go.Bar(
                name="Reported",
                x=["Calculated"],
                y=[calculated_manure],
                marker_color=reported_color,
                legendgroup="manure",
                showlegend=True
            ),
            row=1, col=2
        )
    fig.add_trace(
        go.Bar(
            name="Base Factor",
            x=["Base"],
            y=[BASE_MANURE_FACTOR],
            marker_color=estimated_color,
            legendgroup="manure",
            showlegend=True if calculated_manure <= 0 else False
        ),
        row=1, col=2
    )
    
    # 3. Nitrogen Generation Comparison (Reported, USDA, UCCE)
    reported_nitrogen = get_value(facility_df, 'Total Dry Manure Generated N (lbs)') / 2.20462  # Convert lbs to kg
    usda_nitrogen = get_value(facility_df, 'USDA Nitrogen Estimate (lbs)') / 2.20462  # Convert lbs to kg
    ucce_nitrogen = get_value(facility_df, 'UCCE Nitrogen Estimate (lbs)') / 2.20462  # Convert lbs to kg
    
    if reported_nitrogen > 0:
        fig.add_trace(
            go.Bar(
                name="Reported",
                x=["Reported"],
                y=[reported_nitrogen],
                marker_color=reported_color,
                legendgroup="nitrogen",
                showlegend=True
            ),
            row=1, col=3
        )
    if usda_nitrogen > 0:
        fig.add_trace(
            go.Bar(
                name="USDA Estimate",
                x=["USDA"],
                y=[usda_nitrogen],
                marker_color=estimated_color,
                legendgroup="nitrogen",
                showlegend=True if reported_nitrogen <= 0 else False
            ),
            row=1, col=3
        )
    if ucce_nitrogen > 0:
        fig.add_trace(
            go.Bar(
                name="UCCE Estimate",
                x=["UCCE"],
                y=[ucce_nitrogen],
                marker_color=ucce_color,
                legendgroup="nitrogen",
                showlegend=True if (reported_nitrogen <= 0 and usda_nitrogen <= 0) else False
            ),
            row=1, col=3
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        title_text=f"Facility Comparison: {facility_name}",
        barmode='group',
        legend_title_text=""
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Liters per day", row=1, col=1)
    fig.update_yaxes(title_text="Tons per cow per year", row=1, col=2)
    fig.update_yaxes(title_text="kg per day", row=1, col=3)
    
    return fig

def create_consultant_comparison_plots(df):
    """Create side-by-side bar plots comparing consultant reporting patterns, with facility counts in labels."""
    # Load consultant metrics for R5 2023 only
    metrics_file = "outputs/consolidated/2023_R5_consultant_metrics.csv"
    if not os.path.exists(metrics_file):
        return None
    
    try:
        metrics_df = pd.read_csv(metrics_file)
    except Exception as e:
        st.warning(f"Error loading consultant metrics: {str(e)}")
        return None
    
    if metrics_df.empty:
        return None
    
    # Define consistent colors for consultants
    consultant_colors = {
        'Self-Reported': 'rgb(128, 128, 128)',  # grey
        'Innovative Ag': 'rgb(255, 215, 0)',    # gold
        'Livingston': 'rgb(50, 205, 50)',       # lime green
        'Provost & Pritchard': 'rgb(30, 144, 255)'  # dodger blue
    }
    
    # Prepare consultant labels with facility counts
    metrics_df['ConsultantLabel'] = metrics_df.apply(
        lambda row: f"{row['Consultant']} ({int(row['Facility Count'])})", axis=1
    )
    
    # Create subplots: 1 row, 3 columns
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Manure Factor",
            "Wastewater to\nMilk Ratio",
            "Nitrogen Deviation"
        ],
        horizontal_spacing=0.25
    )
    
    # 1. Manure Factor
    for i, row in metrics_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row['ConsultantLabel']],
                y=[row['Manure Factor Avg']],
                error_y=dict(
                    type='data',
                    array=[row['Manure Factor Std']],
                    visible=True
                ),
                marker_color=consultant_colors.get(row['Consultant'], 'rgb(200, 200, 200)'),
                showlegend=False
            ),
            row=1, col=1
        )
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=BASE_MANURE_FACTOR,
        x1=len(metrics_df) - 0.5,
        y1=BASE_MANURE_FACTOR,
        line=dict(color="red", width=2, dash="dash"),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Avg Manure Factor (tons/cow/year)", row=1, col=1)
    
    # 2. Wastewater Ratio
    for i, row in metrics_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row['ConsultantLabel']],
                y=[row['Wastewater Ratio Avg']],
                error_y=dict(
                    type='data',
                    array=[row['Wastewater Ratio Std']],
                    visible=True
                ),
                marker_color=consultant_colors.get(row['Consultant'], 'rgb(200, 200, 200)'),
                showlegend=False
            ),
            row=1, col=2
        )
    fig.update_yaxes(title_text="Avg Wastewater/Milk Ratio (L/L)", row=1, col=2)
    
    # 3. Nitrogen Deviation
    for i, row in metrics_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row['ConsultantLabel']],
                y=[row['USDA Nitrogen % Dev Avg']],
                error_y=dict(
                    type='data',
                    array=[row['USDA Nitrogen % Dev Std']],
                    visible=True
                ),
                marker_color=consultant_colors.get(row['Consultant'], 'rgb(200, 200, 200)'),
                showlegend=False
            ),
            row=1, col=3
        )
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(metrics_df) - 0.5,
        y1=0,
        line=dict(color="red", width=2, dash="dash"),
        row=1, col=3
    )
    fig.update_yaxes(title_text="Avg USDA Nitrogen % Deviation", row=1, col=3)
    
    # Update layout
    fig.update_layout(
        height=500,
        width=1400,
        barmode='group',
        showlegend=False,
        title_text="Consultant Comparison: Manure Factor, Wastewater to Milk Ratio, and Nitrogen Deviation"
    )
    
    # Improve x-axis labels for all subplots
    for i in range(1, 4):
        fig.update_xaxes(tickangle=30, row=1, col=i) 
    
    return fig

def filter_tab2(df, selected_year):
    print("\n=== Debug: Filtering Data ===")
    print(f"Initial rows: {len(df)}")
    print(f"Selected year: {selected_year}")
    
    available_regions = ['R5', 'R7']
    selected_regions = st.multiselect("Select Regions", available_regions, default=available_regions)
    print(f"Selected regions: {selected_regions}")
    
    if not selected_regions:
        return pd.DataFrame(), [], [], []
    
    available_counties = sorted(df[df['Region'].isin(selected_regions)]['County'].dropna().unique())
    selected_counties = st.multiselect("Select Counties", available_counties, default=available_counties)
    print(f"Selected counties: {selected_counties}")
    
    available_consultants = sorted(df[df['Region'].isin(selected_regions)]['Consultant'].dropna().unique())
    selected_consultants = st.multiselect("Select Consultants", available_consultants, default=available_consultants)
    print(f"Selected consultants: {selected_consultants}")
    
    # Debug each filter step
    year_filter = df['Year'] == selected_year
    print(f"\nRows after year filter: {year_filter.sum()}")
    
    region_filter = df['Region'].isin(selected_regions)
    print(f"Rows after region filter: {region_filter.sum()}")
    
    county_filter = df['County'].isin(selected_counties)
    print(f"Rows after county filter: {county_filter.sum()}")
    
    consultant_filter = df['Consultant'].isin(selected_consultants)
    print(f"Rows after consultant filter: {consultant_filter.sum()}")
    
    filtered_df = df[(df['Year'] == selected_year) &
                     (df['Region'].isin(selected_regions)) &
                     (df['County'].isin(selected_counties)) &
                     (df['Consultant'].isin(selected_consultants))]
    
    print(f"\nFinal filtered rows: {len(filtered_df)}")
    
    # Check if we have any non-null values in our key columns
    key_columns = ['USDA Nitrogen % Dev Avg', 'UCCE Nitrogen % Dev Avg', 'Wastewater Ratio Avg', 'Manure Factor Avg']
    for col in key_columns:
        if col in filtered_df.columns:
            non_null = filtered_df[col].notna().sum()
            print(f"{col}: {non_null} non-null values")
            if non_null > 0:
                print(f"Sample values: {filtered_df[col].dropna().head().tolist()}")
    
    print("=== End Debug: Filtering Data ===\n")
    return filtered_df, selected_regions, selected_counties, selected_consultants

def main():
    st.title("Heaping Piles of Fraud")
    st.markdown("""
    ### Revealing Dairy CAFO Compliance and Data Discrepancies
    This dashboard presents the first public analysis of annual reports from Dairy CAFOs (Concentrated Animal Feeding Operations).
    It reveals significant instances of noncompliance and under-reporting of Manure, Nitrogen, and Wastewater generation.
    
    This data shows what local community members have long known: that CAFO dairies are lying and not prioritizing public health.
    """)
    
    try:
        df = load_data()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["CAFO Maps and Manure Manifests", "CAFO Reporting Data", "Data Availability & Sources"])
        
        with tab1:
            st.write("""
            This section includes maps showing the geographic distribution of CAFO facilities, their size, and their manure exports and imports. 
            """)
            
            # Years filter (single select)
            years = sorted(df['Year'].unique())
            default_year_index = years.index('2023') if '2023' in years else len(years)-1
            selected_year = st.selectbox("Select Year", years, index=default_year_index, key="map_year")
            
            # Filter data for map (only by year)
            map_df = df[df['Year'] == selected_year]
            
            # Display map
            st.subheader("Facility Locations")
            if not map_df.empty and 'Latitude' in map_df.columns:
                st.metric("Total Animals", f"{map_df['Total Herd Size'].sum():,.0f}")
                map_fig = create_map(map_df, selected_year)
                if map_fig is not None:
                    st.plotly_chart(map_fig, use_container_width=True, height=1000)
            else:
                st.warning("No location data available for the selected year.")
            
            # Add ArcGIS map embed
            st.subheader("CAFO Density around Elementary Schools")
            st.components.v1.iframe(
                "https://www.arcgis.com/apps/webappviewer/index.html?id=a247a569c9854bb89689bebb01f5eee4",
                height=600,
                scrolling=True
            )
            
            # Add Manure Manifests map placeholder
            st.subheader("Manure Export Patterns")
            st.write("""
            This map visualizes the movement of manure exports throughout the Central Valley region, 
            revealing the flow of nutrients and potential environmental impacts beyond facility boundaries.
            """)
            
            # Placeholder for the map
            st.info("""
            **Manure Export Map Coming Soon**
            
            Manure manifest showing export destinations and volumes
            """)
            
            # Add spacing between maps
            st.markdown("---")
            st.markdown("<br>", unsafe_allow_html=True)
        
        with tab2:
            st.write("""
            This section focuses on the reported manure, nitrogen and wastewater production in Regions 5 and 7, where we've identified significant variations between reported and estimated values.
            """)
            
            # Years filter
            years = sorted(df['Year'].unique())
            default_year_index = years.index('2023') if '2023' in years else len(years)-1
            selected_year = st.selectbox("Select Year", years, index=default_year_index, key="plot_year")
            
            # Filter data for plots
            filtered_df, selected_regions, selected_counties, selected_consultants = filter_tab2(df, selected_year)
            
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
            nitrogen_fig, wastewater_fig, manure_fig = create_comparison_plots(filtered_df)
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
            ### Manure Generation Factor
            Under-reporting of manure generation can lead to improper nutrient management. Our analysis shows that most facilities are reporting manure generation well below established baselines.
            The base factor is 4.1 tons of manure per cow per year. This is reduced by 40% for heifers and 70% for calves.
            We calculate the dairy's reported manure generation per cow and compare it to this base factor for their herd size. 
            """)
            st.plotly_chart(manure_fig, use_container_width=True)
            
            # Facility search and comparison
            st.markdown("---")
            st.subheader("Facility Search")
            st.write("""
            Search for specific facilities to examine their reporting patterns in detail. 
            This tool helps identify individual cases of potential noncompliance.
            """)
            
            # Add county filter for facility search
            facility_counties = sorted(filtered_df['County'].dropna().unique())
            selected_facility_county = st.selectbox(
                "Filter by County",
                ["All Counties"] + list(facility_counties),
                key="facility_county"
            )
            
            # Filter facilities by selected county
            if selected_facility_county != "All Counties":
                facility_df = filtered_df[filtered_df['County'] == selected_facility_county]
            else:
                facility_df = filtered_df
            
            facility_names = sorted(facility_df['Dairy Name'].unique())
            selected_facility = st.selectbox("Select a Facility", facility_names)
            
            if selected_facility:
                # Get facility details
                facility_data = facility_df[facility_df['Dairy Name'] == selected_facility].iloc[0]
                
                # Display facility details
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Location Details:**")
                    st.write(f"County: {facility_data['County']}")
                    st.write(f"City: {facility_data['City']}")
                    st.write(f"Address: {facility_data['Street Address']}")
                    st.write(f"Zip: {facility_data['Zip']}")
                
                with col2:
                    st.write("**Facility Details:**")
                    st.write(f"Region: {facility_data['Region']}")
                    st.write(f"Consultant: {facility_data['Consultant']}")
                    st.write(f"Total Herd Size: {facility_data['Total Herd Size']:,.0f}")
                
                facility_comparison_fig = create_facility_comparison_plots(facility_df, selected_facility)
                if facility_comparison_fig is not None:
                    st.plotly_chart(facility_comparison_fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {selected_facility}")
            
            # Add consultant comparison plots
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("Consultant Comparison")
            st.write("""
            This section compares reporting patterns across different consultants and self-reported facilities.
            Each bar represents a consultant's average value, with error bars showing the standard deviation.
            """)
            
            consultant_comparison_fig = create_consultant_comparison_plots(df)
            if consultant_comparison_fig is not None:
                st.plotly_chart(consultant_comparison_fig, use_container_width=True)
            
            # Raw data
            st.subheader("Raw Data")
            st.write("""
            View and downloada the complete dataset for detailed analysis.
            Questions on the data can be directed to (insert email)
            """)
            
            # Rename columns for display
            display_df = filtered_df.copy()
            display_df = display_df.rename(columns={
                'Template': 'Template (Raw)',
                'Consultant': 'Consultant'
            })
            
            st.dataframe(display_df)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name="filtered_cafo_data.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.write("""
            This section provides information about the data availability and types for each region, based on the provided text.
            This analysis is inherently limited by the accessibility and consistency of the source data, including issues like inconsistent regional reporting formats and levels of detail, the requirement to visit in-person to get data in some regions, and different data collection periods
            """)

            st.subheader("R-1 North Coast")
            st.markdown("""
            **Data Availability:** Annual reports under Order No. R1-2019-0001 requested by emailing the R2 Water Board and transferred via email.
            - Waste discharge requirements documentation
            """)

            st.subheader("R-2 San Francisco Bay")
            st.markdown("""
            **Data Availability:** Annual reports under Order R2-2016-0031 requested by emailing the R2 Water Board and transferred via email.
            - Facility information and animal counts
            - Certification of facility monitoring programs, waste management plans, grazing management plans, and nutrient management plans
            - Pre-rainy season pollution prevention inspection documentation
            - Groundwater sampling data (when provided) or indication of group monitoring program participation
            """)

            st.subheader("R-5 Central Valley")
            st.markdown("""
            **Data Availability:** Annual reports under General Order No. R5-2007-0035 requested by emailing the Central Valley Water Board and transferred through their Transfer Portal.
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
            **Data Availability:** Annual reports under Order R7-2021-0029 requested by emailing the R7 Water Board and transferred through their Transfer Portal.
            - Animal counts
            - Composting inventory
            - Land application of manure, litter, and process wastewater report
            - Groundwater monitoring report
            - Certification
            """)
            
            st.subheader("R-8 Santa Ana")
            st.markdown("""
            **Data Availability:** Annual reports under Order No. R8-2018-0001 requested by emailing the R8 Water Board and transferred through their Transfer Portal.
            The reports are still available for download as of May 2025.
            https://ftp.waterboards.ca.gov/WebInterface/login.html?path=/CAFO%202023%20Annual%20Reports/
            Username: rb8download
            Password: Region8_public
            - Summary Report of Weekly Storm Water Management Structure Inspections (Form 2)
            - Annual Report Form (Form 3) with facility information and animal population data
            - Manure Tracking Manifests (Form 4)
            - CSV files with farm population and total manure hauled data
            """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the data format and try again.")

if __name__ == "__main__":
    main()