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
import numpy as np

def load_parameters():
    """Load parameters and create mapping dictionaries."""
    parameters = pd.read_csv('ca_cafo_compliance/parameters.csv')
    calculated_metrics = pd.read_csv('ca_cafo_compliance/calculated_metrics.csv')
    return {
        'snake_to_pretty': dict(zip(parameters['parameter_key'], parameters['parameter_name'])),
        'pretty_to_snake': dict(zip(parameters['parameter_name'], parameters['parameter_key'])),
        'data_types': dict(zip(parameters['parameter_key'], parameters['data_type'])),
        'calculated_metrics': dict(zip(calculated_metrics['metric_key'], calculated_metrics['metric_name']))
    }

with open("ca_cafo_compliance/vecteezy_steaming-pile-of-manure-on-farm-field-in-dutch-countryside_8336504.jpg", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode()

# watermark background
st.markdown(
    f'''
    <div style="position: relative; width: 100%; height: 260px; margin-bottom: -120px;">
        <img src="data:image/jpeg;base64,{img_base64}" 
             style="position: absolute; top: 0; left: 0; width: 100%; height: 260px; object-fit: cover; opacity: 0.4; z-index: 0;" />
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
    """Load data from CSV files in the outputs/consolidated directory, or from GitHub as a fallback."""    
    # Try local files first
    csv_files = glob.glob("outputs/consolidated/*.csv")
    print(f"Found {len(csv_files)} CSV files in outputs/consolidated")
    
    dfs = []
    if csv_files:
        for file in csv_files:
            print(f"\nReading file: {file}")
            df = pd.read_csv(file)
            dfs.append(df)
    else:
        print("No local CSV files found. Attempting to load from GitHub...")
        import requests
        from io import StringIO
        base_url = "https://raw.githubusercontent.com/dalywettermark/ca-cafo-compliance/main/outputs/consolidated"
        YEARS = ["2023", "2024"]
        REGIONS = ["1", "2", "5", "7", "8"]
        files_to_load = [f"{year}_{region}_master.csv" for year in YEARS for region in REGIONS]
        for file in files_to_load:
            url = f"{base_url}/{file}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text))
                    print(f"Loaded {file} from GitHub, shape: {df.shape}")
                    dfs.append(df)
                else:
                    print(f"File {file} not found on GitHub.")
            except Exception as e:
                print(f"Error loading {file} from GitHub: {e}")
    
    if not dfs:
        print("No data could be loaded from local files or GitHub.")
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True) 
    # Clean up Year column: drop NaN, convert to int then str, and filter out invalid years
    if 'Year' in combined_df.columns:
        combined_df = combined_df[combined_df['Year'].notna()]
        def year_to_str(x):
            try:
                return str(int(float(x)))
            except Exception:
                return None
        combined_df['Year'] = combined_df['Year'].apply(year_to_str)
        combined_df = combined_df[combined_df['Year'].notna()]
    
    # Load parameters for consistent column naming
    params = load_parameters()
    snake_to_pretty = params['snake_to_pretty']
    
    

    # Rename columns to ensure consistency with pretty names
    column_mapping = {
        'usda_nitrogen_pct_deviation': 'USDA Nitrogen % Deviation',
        'ucce_nitrogen_pct_deviation': 'UCCE Nitrogen % Deviation',
        'total_manure_gen_n_after_nh3_losses_lbs': 'Total Manure N (lbs)',
        'usda_nitrogen_estimate_lbs': 'USDA N Estimate (lbs)',
        'ucce_nitrogen_estimate_lbs': 'UCCE N Estimate (lbs)',
        'avg_milk_lb_per_cow_day': 'Average Milk Production (lb per cow per day)',
        'avg_milk_cows': 'Average Milk Cows',
        'avg_dry_cows': 'Average Dry Cows',
        'avg_bred_heifers': 'Average Bred Heifers',
        'avg_heifers': 'Average Heifers',
        'avg_calves_4_6_mo': 'Average Calves (4-6 mo.)',
        'avg_calves_0_3_mo': 'Average Calves (0-3 mo.)',
        'avg_other': 'Average Other',
        'total_herd_size': 'Total Herd Size',
        'reported_annual_milk_production_l': 'Reported Annual Milk Production (L)',
        'avg_milk_prod_kg_per_cow': 'Average Milk Production (kg per cow)',
        'avg_milk_prod_l_per_cow': 'Average Milk Production (L per cow)'
    }
    
    # Apply renaming
    for old_col, new_col in column_mapping.items():
        if old_col in combined_df.columns:
            print(f"Renaming {old_col} to {new_col}")
            combined_df[new_col] = combined_df[old_col]
            combined_df = combined_df.drop(columns=[old_col])
        elif new_col not in combined_df.columns:
            print(f"Warning: Neither {old_col} nor {new_col} found in columns")
    
    # Ensure all pretty names from parameters.csv exist in the dataframe
    for pretty_name in snake_to_pretty.values():
        if pretty_name not in combined_df.columns:
            combined_df[pretty_name] = np.nan
    
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
        print(df['Milk Production Source'].value_counts())
    else:
        print(f"\nWARNING: Milk production column '{milk_col}' not found")
        df.loc[:, 'Milk Production Source'] = 'Estimated'
    
    # --- Nitrogen Deviation Plot ---
    # Check for nitrogen deviation columns before renaming
    print("\nChecking nitrogen deviation columns before renaming:")
    print(f"'usda_nitrogen_pct_deviation' in columns: {'usda_nitrogen_pct_deviation' in df.columns}")
    print(f"'ucce_nitrogen_pct_deviation' in columns: {'ucce_nitrogen_pct_deviation' in df.columns}")
    print(f"'USDA Nitrogen % Deviation' in columns: {'USDA Nitrogen % Deviation' in df.columns}")
    print(f"'UCCE Nitrogen % Deviation' in columns: {'UCCE Nitrogen % Deviation' in df.columns}")

    # Use the correct column names from read_reports.py
    usda_col = 'USDA Nitrogen % Deviation'
    ucce_col = 'UCCE Nitrogen % Deviation'

    # Check for nitrogen deviation columns after renaming
    print("\nChecking nitrogen deviation columns after renaming:")
    print(f"'USDA Nitrogen % Deviation' in columns: {'USDA Nitrogen % Deviation' in df.columns}")
    print(f"'UCCE Nitrogen % Deviation' in columns: {'UCCE Nitrogen % Deviation' in df.columns}")
    if usda_col in df.columns:
        print(f"USDA Nitrogen % Deviation non-null values: {df[usda_col].notna().sum()}")
        print(f"USDA Nitrogen % Deviation highest values: {df[usda_col].dropna().nlargest(5).tolist() if df[usda_col].notna().any() else 'None'}")
    if ucce_col in df.columns:
        print(f"UCCE Nitrogen % Deviation non-null values: {df[ucce_col].notna().sum()}")
        print(f"UCCE Nitrogen % Deviation higest values: {df[ucce_col].dropna().nlargest(5).tolist() if df[ucce_col].notna().any() else 'None'}")
    
    # 1. Nitrogen Generation - Percentage Deviation Histograms
    nitrogen_fig = go.Figure()
    
    # Use the pretty column names
    usda_col = "USDA Nitrogen % Deviation"
    ucce_col = "UCCE Nitrogen % Deviation" 
    
    print("\n=== Nitrogen Generation Plot ===")
    print(f"Looking for columns: {usda_col}, {ucce_col}")
    
    # Filter data for USDA nitrogen deviations
    usda_nitrogen_data = df[usda_col].dropna() if usda_col in df.columns else pd.Series()
    print(f"USDA Nitrogen data points: {len(usda_nitrogen_data)}")
    if not usda_nitrogen_data.empty:
        # Group extreme values
        usda_nitrogen_data = usda_nitrogen_data.clip(-100, 100)
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
        # Group extreme values
        ucce_nitrogen_data = ucce_nitrogen_data.clip(-100, 100)
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
    
    # Update layout for nitrogen plot
    nitrogen_fig.update_layout(
        title="Nitrogen Generation - Percentage Deviation from Estimates",
        xaxis_title="Percentage Deviation",
        yaxis_title="Number of Facilities",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # 2. Wastewater Generation - Ratio to Milk Production
    wastewater_fig = go.Figure()
    
    # Use pretty column names
    reported_col = "Wastewater to Reported Milk Ratio"
    estimated_col = "Wastewater to Estimated Milk Ratio"
    
    print("\n=== Wastewater Generation Plot ===")
    print(f"Looking for columns: {reported_col}, {estimated_col}")
    
    # Filter data for reported wastewater ratios
    reported_mask = df['Milk Production Source'] == 'Reported'
    reported_wastewater_data = df.loc[reported_mask, reported_col].dropna() if reported_col in df.columns else pd.Series()
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
    estimated_wastewater_data = df.loc[estimated_mask, estimated_col].dropna() if estimated_col in df.columns else pd.Series()
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
    
    # Add green rectangle for expected range
    if not reported_wastewater_data.empty or not estimated_wastewater_data.empty:
        wastewater_fig.add_vrect(
            x0=0,
            x1=L_WW_PER_L_MILK_LOW,
            fillcolor="rgba(200,0,0,0.15)",
            layer="below",
            line_width=0,
            annotation_text="Likely<br>Under-Reporting",
            annotation_position="top"
        )
    
    # Update layout for wastewater plot 
    wastewater_fig.update_layout(
        title="Wastewater Generation - Ratio to Milk Production",
        xaxis_title="Liters Wastewater per Liter Milk",
        yaxis_title="Number of Facilities",
        showlegend=True,
        barmode="stack",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1
        )
    )
    
    return nitrogen_fig, wastewater_fig

# Define standardized colors for the app
NITROGEN_COLOR = 'rgb(0, 71, 171)'
NITROGEN_EST_COLOR = 'rgba(0, 71, 171, 0.5)'
WASTEWATER_COLOR = 'rgb(135, 206, 235)'
WASTEWATER_EST_COLOR = 'rgba(135, 206, 235, 0.5)'
MANURE_COLOR = 'rgb(255, 165, 0)'
MANURE_EST_COLOR = 'rgba(255, 165, 0, 0.5)'

# General dual-use bar plot function
from typing import List, Tuple

def dual_bar_plot(fig, row, col, reported_label, reported_value, estimated_label, estimated_value, color, est_color, y_label, unit):
    bars = []
    if reported_value is not None:
        bars.append(dict(
            x=[reported_label],
            y=[reported_value],
            name=reported_label,
            marker_color=color,
            marker_pattern_shape="",
            text=[f"{reported_value:,.0f} {unit}"],
            textposition='auto',
            legendgroup=reported_label,
            showlegend=True
        ))
    if estimated_value is not None:
        bars.append(dict(
            x=[estimated_label],
            y=[estimated_value],
            name=estimated_label,
            marker_color=est_color,
            marker_pattern_shape="/",
            text=[f"{estimated_value:,.0f} {unit}"],
            textposition='auto',
            legendgroup=estimated_label,
            showlegend=True
        ))
    for bar in bars:
        fig.add_trace(go.Bar(**bar), row=row, col=col)
    fig.update_yaxes(title_text=y_label, row=row, col=col)


def create_facility_comparison_plots(df, facility_name):
    """Create comparison plots for a specific facility."""
    params = load_parameters()
    calc = params['calculated_metrics']
    pretty = params['snake_to_pretty']
    herd_cols = [
        'Average Milk Cows', 'Average Dry Cows', 'Average Bred Heifers',
        'Average Heifers', 'Average Calves (4-6 mo.)', 'Average Calves (0-3 mo.)', 'Average Other'
    ]
    if facility_name not in df['Dairy Name'].values:
        return None
    facility_data = df[df['Dairy Name'] == facility_name].iloc[0]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Herd Breakdown",
            "Nitrogen Generation", 
            "Wastewater Generation",
            "Manure Generation"
        )
    )
    # 1. Herd Breakdown (solid color only)
    herd_data = [] 
    for col in herd_cols:
        if col in facility_data and pd.notna(facility_data[col]):
            herd_data.append({'name': col, 'value': facility_data[col]})
    if herd_data:
        fig.add_trace(
            go.Bar(
                x=[d['name'] for d in herd_data],
                y=[d['value'] for d in herd_data],
                name='Herd Breakdown',
                marker_color='gray',
                text=[f"{d['value']:,.0f}" for d in herd_data],
                textposition='auto',
                showlegend=False
            ),
            row=1, col=1
        )
    # 2. Nitrogen Generation (reported, USDA est, UCCE est)
    n_reported = calc.get('total_reported_n_lbs', 'Total Reported N (lbs)')
    n_usda = calc.get('usda_nitrogen_estimate_lbs', 'USDA N Estimate (lbs)')
    n_ucce = calc.get('ucce_nitrogen_estimate_lbs', 'UCCE N Estimate (lbs)')
    n_reported_val = facility_data[n_reported] if n_reported in facility_data and pd.notna(facility_data[n_reported]) else None
    n_usda_val = facility_data[n_usda] if n_usda in facility_data and pd.notna(facility_data[n_usda]) else None
    n_ucce_val = facility_data[n_ucce] if n_ucce in facility_data and pd.notna(facility_data[n_ucce]) else None
    # Plot reported (solid), USDA est (hashed), UCCE est (hashed)
    if n_reported_val is not None:
        fig.add_trace(go.Bar(
            x=['Reported N'],
            y=[n_reported_val],
            name='Reported',
            marker_color=NITROGEN_COLOR,
            marker_pattern_shape="",
            text=[f"{n_reported_val:,.0f} lbs"],
            textposition='auto',
            legendgroup='reported',
            showlegend=True
        ), row=1, col=2)
    if n_usda_val is not None:
        fig.add_trace(go.Bar(
            x=['USDA Estimate'],
            y=[n_usda_val],
            name='Estimated',
            marker_color=NITROGEN_EST_COLOR,
            marker_pattern_shape="/",
            text=[f"{n_usda_val:,.0f} lbs"],
            textposition='auto',
            legendgroup='estimated',
            showlegend=True
        ), row=1, col=2)
    if n_ucce_val is not None:
        fig.add_trace(go.Bar(
            x=['UCCE Estimate'],
            y=[n_ucce_val],
            name='UCCE Estimate',
            marker_color=NITROGEN_EST_COLOR,
            marker_pattern_shape="/",
            text=[f"{n_ucce_val:,.0f} lbs"],
            textposition='auto',
            legendgroup='UCCE Estimate',
            showlegend=True
        ), row=1, col=2)
    # 3. Wastewater Generation
    ww_reported = calc.get('total_ww_gen_liters', 'Total Wastewater Generated (L)')
    ww_estimated = None  # If you have an estimated value, set it here
    ww_reported_val = facility_data[ww_reported] if ww_reported in facility_data and pd.notna(facility_data[ww_reported]) else None
    ww_estimated_val = None  # Add logic for estimated if available
    if ww_reported_val is not None:
        fig.add_trace(go.Bar(
            x=['Reported WW'],
            y=[ww_reported_val],
            name='Reported',
            marker_color=WASTEWATER_COLOR,
            marker_pattern_shape="",
            text=[f"{ww_reported_val:,.0f} L"],
            textposition='auto',
            legendgroup='reported',
            showlegend=False
        ), row=2, col=1)
    if ww_estimated_val is not None:
        fig.add_trace(go.Bar(
            x=['Estimated WW'],
            y=[ww_estimated_val],
            name='Estimated',
            marker_color=WASTEWATER_EST_COLOR,
            marker_pattern_shape="/",
            text=[f"{ww_estimated_val:,.0f} L"],
            textposition='auto',
            legendgroup='estimated',
            showlegend=False
        ), row=2, col=1)
    # 4. Manure Generation
    manure_reported = 'total_manure_excreted_tons'
    herd_size_col = calc.get('total_herd_size', 'Total Herd Size')
    manure_reported_val = facility_data[manure_reported] if manure_reported in facility_data and pd.notna(facility_data[manure_reported]) else None
    manure_estimated_val = BASE_MANURE_FACTOR * facility_data[herd_size_col] if herd_size_col in facility_data and pd.notna(facility_data[herd_size_col]) else None
    if manure_reported_val is not None:
        fig.add_trace(go.Bar(
            x=['Reported Manure'],
            y=[manure_reported_val],
            name='Reported',
            marker_color=MANURE_COLOR,
            marker_pattern_shape="",
            text=[f"{manure_reported_val:,.0f} tons"],
            textposition='auto',
            legendgroup='reported',
            showlegend=False
        ), row=2, col=2)
    if manure_estimated_val is not None:
        fig.add_trace(go.Bar(
            x=['Estimated Manure'],
            y=[manure_estimated_val],
            name='Estimated',
            marker_color=MANURE_EST_COLOR,
            marker_pattern_shape="/",
            text=[f"{manure_estimated_val:,.0f} tons"],
            textposition='auto',
            legendgroup='estimated',
            showlegend=False
        ), row=2, col=2)
    fig.update_layout(
        showlegend=True,
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5
        )
    )
    return fig

def create_consultant_comparison_plots():
    # Load pre-calculated consultant metrics
    metrics_path = "outputs/consolidated/2023_R5_consultant_metrics.csv"
    df = pd.read_csv(metrics_path)
    consultants = df['Consultant']

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Nitrogen Deviation", "Wastewater to Milk", "Manure Factor"),
        horizontal_spacing=0.15  # Add whitespace between subplots
    )

    # 1. Nitrogen Generation
    y1 = df['USDA Nitrogen % Dev Avg']
    y1_std = df['USDA Nitrogen % Dev Std']
    y2 = df['UCCE Nitrogen % Dev Avg']
    y2_std = df['UCCE Nitrogen % Dev Std']

    # Only show error bars if std is not null
    for y, y_std, name, color, col in [
        (y1, y1_std, "Reported Nitrogen Deviation from USDA Estimate", "blue", 1),
        (y2, y2_std, "Reported Nitrogen Deviation from UCCE Estimate", "lightblue", 1)
    ]:
        error_y = dict(type='data', array=[v if not np.isnan(v) else 0 for v in y_std], visible=True) if not y_std.isnull().all() else None
        fig.add_trace(
            go.Bar(
                x=consultants,
                y=y,
                name=name,
                marker_color=color,
                error_y=error_y if error_y and not all(np.isnan(y_std)) else None,
                text=[f"{v:.0f}" if not np.isnan(v) else "" for v in y],
                textposition='auto',
                showlegend=False
            ),
            row=1, col=1
        )

    # 2. Wastewater Generation
    y3 = df['Wastewater Ratio Avg']
    y3_std = df['Wastewater Ratio Std']
    error_y = dict(type='data', array=[v if not np.isnan(v) else 0 for v in y3_std], visible=True) if not y3_std.isnull().all() else None
    fig.add_trace(
        go.Bar(
            x=consultants,
            y=y3,
            name="Based on Reported Milk",
            marker_color="red",
            error_y=error_y if error_y and not all(np.isnan(y3_std)) else None,
            text=[f"{v:.2f}" if not np.isnan(v) else "" for v in y3],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )

    # 3. Milk Production
    y4 = df['Manure Factor Avg']
    y4_std = df['Manure Factor Std']
    error_y = dict(type='data', array=[v if not np.isnan(v) else 0 for v in y4_std], visible=True) if not y4_std.isnull().all() else None
    fig.add_trace(
        go.Bar(
            x=consultants,
            y=y4,
            name="Annual Production",
            marker_color="green",
            error_y=error_y if error_y and not all(np.isnan(y4_std)) else None,
            text=[f"{v:.0f}" if not np.isnan(v) else "" for v in y4],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(
        showlegend=False,  # Remove legend
        height=500,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="Liters per Liter Milk", row=1, col=2)
    fig.update_yaxes(title_text="Manure per Cow", row=1, col=3)
    return fig

def filter_tab2(df, selected_year):
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
    key_columns = ['USDA Nitrogen % Deviation', 'UCCE Nitrogen % Deviation', 'Wastewater to Reported Milk Ratio', 'Wastewater to Estimated Milk Ratio', 'Calculated Manure Factor']
    for col in key_columns:
        if col in filtered_df.columns:
            non_null = filtered_df[col].notna().sum()
            print(f"{col}: {non_null} non-null values")
            if non_null > 0:
                print(f"Sample values: {filtered_df[col].dropna().head().tolist()}")
    
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
        tab1, tab2, tab3, tab4 = st.tabs(["CAFO Maps and Manure Manifests", "CAFO Reporting Data", "Enforcement", "Data Availability & Sources"])
        
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
                    st.write("*To do: incorporate the CADD dataset for herd size to capture more facilities. And/or Lucia's dataset*")
                    st.plotly_chart(map_fig, use_container_width=True, height=1000)
            else:
                st.warning("No location data available for the selected year.")
            
            # Add ArcGIS map embed
            st.subheader("CAFO Density around Elementary Schools")
            st.write("Just an example of how we can embed an ArcGIS map that has been published to an online url. This example is from https://www.arcgis.com/apps/webappviewer/index.html?id=a247a569c9854bb89689bebb01f5eee4")
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
            
            Manure manifest showing export destinations and volumes (from Sophia)
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

            st.write("*Note: We can incorporate any of the figures that Hailey is creating from 2023 here too*")
            
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
            ### Manure Generation Factor
            Under-reporting of manure generation can lead to improper nutrient management. Our analysis shows that most facilities are reporting manure generation well below established baselines.
            The base factor is 4.1 tons of manure per cow per year. This is reduced by 40% for heifers and 70% for calves.
            We calculate the dairy's reported manure generation per cow and compare it to this base factor for their herd size. 
            
            * note: as of May 16, the zero column is too high since there are some files being read where manure is non-zero but not being picked up. Similarly, the factor for some facilities might be over-counted because the herd size is being under-counted reading the files*
            """)
            # Generate manure_fig
            manure_col = 'Calculated Manure Factor'
            manure_fig = go.Figure()
            if manure_col in filtered_df.columns:
                manure_data = filtered_df[manure_col].dropna()
                if not manure_data.empty:
                    manure_fig.add_trace(
                        go.Histogram(
                            x=manure_data,
                            nbinsx=50,
                            name="Reported Manure Factor",
                            marker_color='rgb(255, 165, 0)',
                            opacity=0.7
                        )
                    )
                # Likely under-reporting if x < 8
                manure_fig.add_vrect(
                    x0=manure_data.min(), x1=8,
                    fillcolor="rgba(200,0,0,0.15)",
                    layer="below",
                    line_width=0,
                    annotation_text="Likely<br>Under-reporting",
                    annotation_position="top"
                    )
                manure_fig.update_layout(
                    title="Manure Generation Factor Distribution",
                    xaxis_title="Tons Manure per Cow per Year",
                    yaxis_title="Number of Facilities",
                    showlegend=True
                )
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
            selected_facility = st.selectbox("Select a Facility", facility_names, index=facility_names.index("AJ Slenders Dairy") if "AJ Slenders Dairy" in facility_names else 0)
            
            if selected_facility:
                # Get facility details
                facility_data = facility_df[facility_df['Dairy Name'] == selected_facility].iloc[0]
                
                # Display facility details
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Address: {facility_data['Street Address']}")
                    st.write(f"City, State, Zip: {facility_data['City']}, CA {facility_data['Zip']}")
                
                with col2:
                    st.write(f"Report prepared by: {facility_data['Consultant']}")
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
            
            consultant_comparison_fig = create_consultant_comparison_plots()
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
            ### Irrigated Lands Regulatory Program (ILRP) Enforcement Actions by the Central Valley Water Board (2015-2025)
                     
                     (just added this section based on the great Air team presentation, 
                     in case we want to highlight their findings!)

            | Enforcement Type                                      | Administrative Civil Liability (ACLs) | Cleanup and Abatement Orders (CAOs) |
            |------------------------------------------------------|:-------------------------------------:|:-----------------------------------:|
            | Enforcement for Failure to Obtain Regulatory Coverage| 14                                    |                                     |
            | Enforcement for Failure to Submit Evaluation Report  | 6                                     |                                     |
            | Enforcement for Sediment Discharge                   | 1                                     | 2                                   |

            *Source: Central Valley Water Board enforcement data, 2015-2025. https://www.waterboards.ca.gov/centralvalley/water_issues/irrigated_lands/formal_enforcement/
            
            *Note: Most enforcement actions are for paperwork violations, not nutrient or wastewater pollution.*
            """)
        with tab4:
            st.write("""
            This section provides information about the data availability and types for each region, based on the provided text.
            This analysis is inherently limited by the accessibility and consistency of the source data, including issues like inconsistent regional reporting formats and levels of detail, the requirement to visit in-person to get data in some regions, and different data collection periods
            """)

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

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the data format and try again.")

if __name__ == "__main__":
    main()