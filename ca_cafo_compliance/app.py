import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import geopandas as gpd
from conversion_factors import *

# Streamlit generated and modified with prompts to Claude 3.7, 
# based on the data output from

st.set_page_config(page_title="Heaping Piles of Fraud: CA CAFO Annual Report Data Exploration", layout="centered")

def load_data():
    """Load all consolidated data files and combine them."""
    all_files = glob.glob("outputs/consolidated/*.csv")
    
    # If no local files found, try loading from GitHub
    if not all_files:
        try:
            import requests
            from io import StringIO
            
            # GitHub raw content URL for the consolidated data
            base_url = "https://raw.githubusercontent.com/dalywettermark/ca-cafo-compliance/main/outputs/consolidated"
            
            # List of files to load
            files_to_load = [
                f"{year}_{region}_master.csv"
                for year in YEARS
                for region in REGIONS
            ]
            
            dfs = []
            for file in files_to_load:
                try:
                    url = f"{base_url}/{file}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        df = pd.read_csv(StringIO(response.text))
                        
                        # Extract year from filename
                        year = file.split('_')[0]
                        df['Year'] = year
                        
                        # Convert categorical columns to string type
                        categorical_cols = ['Year', 'Region', 'County', 'Template', 'Consultant', 'Dairy Name', 'Dairy Address', 'filename']
                        for col in categorical_cols:
                            if col in df.columns:
                                df[col] = df[col].astype(str)
                        
                        # Convert numeric columns to float
                        numeric_cols = [col for col in df.columns if col not in categorical_cols]
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        dfs.append(df)
                except Exception as e:
                    st.warning(f"Error loading {file} from GitHub: {str(e)}")
            
            if not dfs:
                st.error("No data could be loaded from GitHub. Please check the repository URL and file paths.")
                st.stop()
                
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
        except ImportError:
            st.error("Could not load data from GitHub.")
            st.stop()
    else:
        # Load from local files
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                
                # Extract year from filename
                year = os.path.basename(file).split('_')[0]
                df['Year'] = year
                
                # Convert categorical columns to string type
                categorical_cols = ['Year', 'Region', 'County', 'Template', 'Consultant', 'Dairy Name', 'Dairy Address', 'filename']
                for col in categorical_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str)
                
                # Convert numeric columns to float
                numeric_cols = [col for col in df.columns if col not in categorical_cols]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                dfs.append(df)
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
        
        if not dfs:
            st.error("No data could be loaded.")
            st.stop()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure Consultant column exists and is properly mapped
    if 'Consultant' not in combined_df.columns:
        combined_df['Consultant'] = combined_df['Template'].map(consultant_mapping).fillna('Unknown')
    else:
        # Update any missing consultant values
        mask = combined_df['Consultant'].isna() | (combined_df['Consultant'] == 'nan')
        combined_df.loc[mask, 'Consultant'] = combined_df.loc[mask, 'Template'].map(consultant_mapping).fillna('Unknown')
    
    # Clean up Year column
    combined_df['Year'] = combined_df['Year'].astype(str)
    

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
    
    # Create the base map with facilities
    fig = px.scatter_map(
        map_df,
        lat='Latitude',
        lon='Longitude',
        size='Total Herd Size',
        color='Region',
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
    # Color scheme for consistency
    estimated_color = 'rgb(135, 206, 235)'  # light blue
    reported_color = 'rgb(0, 71, 171)'      # dark blue
    ucce_color = 'rgb(144, 238, 144)'       # light green
    base_color = 'rgb(255, 165, 0)'         # orange for base values
    
    # 1. Nitrogen Generation - Percentage Deviation Histograms
    nitrogen_fig = go.Figure()
    
    # Create histogram of USDA estimate percentage deviations
    nitrogen_fig.add_trace(
        go.Histogram(
            x=df['USDA Nitrogen % Deviation'].dropna(),
            nbinsx=50,
            name="USDA Estimate",
            marker_color=estimated_color,
            opacity=0.7
        )
    )
    
    # Create histogram of UCCE estimate percentage deviations
    nitrogen_fig.add_trace(
        go.Histogram(
            x=df['UCCE Nitrogen % Deviation'].dropna(),
            nbinsx=50,
            name="UCCE Estimate",
            marker_color=ucce_color,
            opacity=0.7
        )
    )
    
    # Add vertical line at 0% deviation
    nitrogen_fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, df['USDA Nitrogen % Deviation'].value_counts().max()],
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
    
    # Create histogram of wastewater ratios based on reported milk production
    reported_mask = df['Milk Production Source'] == 'Reported'
    wastewater_fig.add_trace(
        go.Histogram(
            x=df[reported_mask]['Ratio of Wastewater to Milk (L/L)'].dropna(),
            nbinsx=50,
            name="Based on Reported Milk",
            marker_color=reported_color,
            opacity=0.7
        )
    )
    
    # Create histogram of wastewater ratios based on estimated milk production
    estimated_mask = df['Milk Production Source'] == 'Estimated'
    wastewater_fig.add_trace(
        go.Histogram(
            x=df[estimated_mask]['Ratio of Wastewater to Milk (L/L)'].dropna(),
            nbinsx=50,
            name="Based on Estimated Milk",
            marker_color=estimated_color,
            opacity=0.7
        )
    )
    
    # Add vertical line for estimated ratio
    wastewater_fig.add_trace(
        go.Scatter(
            x=[df['Wastewater to Milk Ratio'].mean()] * 2,
            y=[0, df['Ratio of Wastewater to Milk (L/L)'].value_counts().max()],
            mode='lines',
            name='Average Estimated Ratio',
            line=dict(color=base_color, width=2, dash='dash')
        )
    )
    
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
    
    # Create histogram of manure factors
    manure_fig.add_trace(
        go.Histogram(
            x=df['Calculated Manure Factor'].dropna(),
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
            y=[0, df['Calculated Manure Factor'].value_counts().max()],
            mode='lines',
            name='Base Factor',
            line=dict(color=base_color, width=2, dash='dash')
        )
    )
    
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
    """Create bar plots comparing consultant reporting patterns."""
    # Load consultant metrics for R5 2023 only
    metrics_file = "outputs/consolidated/2023_R5_consultant_metrics.csv"
    if not os.path.exists(metrics_file):
        return None, None, None, None
    
    try:
        metrics_df = pd.read_csv(metrics_file)
    except Exception as e:
        st.warning(f"Error loading consultant metrics: {str(e)}")
        return None, None, None, None
    
    if metrics_df.empty:
        return None, None, None, None
    
    # Define consistent colors for consultants
    consultant_colors = {
        'Self-Reported': 'rgb(128, 128, 128)',  # grey
        'Innovative Ag': 'rgb(255, 215, 0)',    # gold
        'Livingston': 'rgb(50, 205, 50)',       # lime green
        'Provost & Pritchard': 'rgb(30, 144, 255)'  # dodger blue
    }
    
    # Create four bar plots
    # 1. Manure Factor
    manure_fig = go.Figure()
    for _, row in metrics_df.iterrows():
        manure_fig.add_trace(go.Bar(
            x=[row['Consultant']],
            y=[row['Manure Factor Avg']],
            error_y=dict(
                type='data',
                array=[row['Manure Factor Std']],
                visible=True
            ),
            marker_color=consultant_colors.get(row['Consultant'], 'rgb(200, 200, 200)'),  # default grey
            showlegend=False
        ))
    
    manure_fig.add_shape(
        type="line",
        x0=-0.5,
        y0=BASE_MANURE_FACTOR,
        x1=len(metrics_df) - 0.5,
        y1=BASE_MANURE_FACTOR,
        line=dict(color="red", width=2, dash="dash")
    )
    
    manure_fig.update_layout(
        title="Average Manure Factor by Consultant (R5, 2023)",
        xaxis_title="Consultant",
        yaxis_title="Average Manure Factor (tons/cow/year)",
        showlegend=False,
        height=500
    )
    
    # 2. Wastewater Ratio
    wastewater_fig = go.Figure()
    for _, row in metrics_df.iterrows():
        wastewater_fig.add_trace(go.Bar(
            x=[row['Consultant']],
            y=[row['Wastewater Ratio Avg']],
            error_y=dict(
                type='data',
                array=[row['Wastewater Ratio Std']],
                visible=True
            ),
            marker_color=consultant_colors.get(row['Consultant'], 'rgb(200, 200, 200)'),  # default grey
            showlegend=False
        ))
    
    wastewater_fig.update_layout(
        title="Average Wastewater to Milk Ratio by Consultant (R5, 2023)",
        xaxis_title="Consultant",
        yaxis_title="Average Wastewater to Milk Ratio (L/L)",
        showlegend=False,
        height=500
    )
    
    # 3. Nitrogen Deviation
    nitrogen_fig = go.Figure()
    for _, row in metrics_df.iterrows():
        nitrogen_fig.add_trace(go.Bar(
            x=[row['Consultant']],
            y=[row['USDA Nitrogen % Dev Avg']],
            error_y=dict(
                type='data',
                array=[row['USDA Nitrogen % Dev Std']],
                visible=True
            ),
            marker_color=consultant_colors.get(row['Consultant'], 'rgb(200, 200, 200)'),  # default grey
            showlegend=False
        ))
    
    nitrogen_fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(metrics_df) - 0.5,
        y1=0,
        line=dict(color="red", width=2, dash="dash")
    )
    
    nitrogen_fig.update_layout(
        title="Average Nitrogen Deviation by Consultant (R5, 2023)",
        xaxis_title="Consultant",
        yaxis_title="Average USDA Nitrogen % Deviation",
        showlegend=False,
        height=500
    )
    
    # 4. Number of Facilities
    facilities_fig = go.Figure()
    for _, row in metrics_df.iterrows():
        facilities_fig.add_trace(go.Bar(
            x=[row['Consultant']],
            y=[row['Facility Count']],
            marker_color=consultant_colors.get(row['Consultant'], 'rgb(200, 200, 200)'),  # default grey
            showlegend=False
        ))
    
    facilities_fig.update_layout(
        title="Number of Facilities by Consultant (R5, 2023)",
        xaxis_title="Consultant",
        yaxis_title="Number of Facilities",
        showlegend=False,
        height=500
    )
    
    return manure_fig, wastewater_fig, nitrogen_fig, facilities_fig
 
def main():
    st.title("Heaping Piles of Fraud")
    st.markdown("""
    ### Revealing Dairy CAFO Compliance and Data Discrepancies
    This dashboard presents the first public analysis of annual reports from Dairy CAFOs (Concentrated Animal Feeding Operations), 
    revealing significant instances of noncompliance and data discrepancies in dairy management. Our analysis focuses on identifying
    potential under-reporting of Manure, Nitrogen, and Wastewater generation.
    
    This data shows what local community members have long known: that CAFO dairies are lying and not keeping
    public health in mind with their operation.
    """)
    
    try:
        df = load_data()
        
        # Create tabs
        tab1, tab2 = st.tabs(["CAFO Maps and Manure Manifests", "CAFO Reporting Data"])
        
        with tab1:
            st.write("""
            Explore the geographic distribution of CAFO facilities, their size, and their manure exports and imports. 
            This visualization illuminates environmental justice concerns for communities near CAFOs.
            """)
            
            # Years filter (single select)
            years = sorted(df['Year'].unique())
            selected_year = st.selectbox("Select Year", years, index=len(years)-1, key="map_year")
            
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
            Analyze detailed CAFO reporting data to identify patterns of noncompliance and data discrepancies. 
            This section focuses on Regions 5 and 7, where we've identified significant variations between reported and estimated values.
            """)
            
            # Years filter
            years = sorted(df['Year'].unique())
            selected_year = st.selectbox("Select Year", years, index=len(years)-1, key="plot_year")
            
            # Region filter (only R5 and R7)
            available_regions = ['R5', 'R7']
            selected_regions = st.multiselect(
                "Select Regions", 
                available_regions,
                default=available_regions
            )
            
            if selected_regions:
                # Filter counties based on selected regions
                available_counties = sorted(df[df['Region'].isin(selected_regions)]['County'].dropna().unique())
                selected_counties = st.multiselect(
                    "Select Counties", 
                    available_counties,
                    default=available_counties
                )
                
                # Filter consultants based on selected regions
                available_consultants = sorted(df[df['Region'].isin(selected_regions)]['Consultant'].dropna().unique())
                selected_consultants = st.multiselect(
                    "Select Consultants", 
                    available_consultants,
                    default=available_consultants
                )
                
                # Filter data for plots
                filtered_df = df[
                    (df['Year'] == selected_year) &
                    (df['Region'].isin(selected_regions)) &
                    (df['County'].isin(selected_counties)) &
                    (df['Consultant'].isin(selected_consultants))
                ]
                
                # Comparison plots with explanations
                st.subheader("Estimated vs Actual Comparisons")
                
                # Nitrogen Generation Plot
                st.markdown("""
                ### Nitrogen Generation Comparison
                This plot reveals significant discrepancies between reported and estimated nitrogen generation.
                - **Why it matters**: Under-reporting of nitrogen generation can lead to improper nutrient management and potential water quality issues. Our analysis shows that many facilities are reporting significantly less nitrogen than estimated.
                - **How to interpret**: 
                    - Values above 0% indicate facilities reporting less nitrogen than estimated
                    - The USDA estimate is based on manure generation (tons × 0.006)
                    - The UCCE estimate is based on animal units × 365 days
                    - Large positive deviations suggest potential under-reporting
                """)
                nitrogen_fig, wastewater_fig, manure_fig = create_comparison_plots(filtered_df)
                st.plotly_chart(nitrogen_fig, use_container_width=True)
                
                # Wastewater to Milk Ratio Plot
                st.markdown("""
                ### Wastewater to Milk Ratio
                This plot exposes unusual patterns in wastewater reporting relative to milk production.
                - **Why it matters**: Unusually high ratios may indicate water waste or improper reporting. Our analysis reveals that many facilities are reporting wastewater volumes that don't align with their milk production.
                - **How to interpret**:
                    - The ratio is calculated as: Total Process Wastewater (L) / Annual Milk Production (L)
                    - Milk production is either reported or estimated (using 80 lb/cow/day default)
                    - Values significantly above the average estimated ratio may indicate water waste
                    - Values significantly below may indicate under-reporting of wastewater
                """)
                st.plotly_chart(wastewater_fig, use_container_width=True)
                
                # Manure Factor Plot
                st.markdown("""
                ### Manure Generation Factor
                This plot demonstrates widespread under-reporting of manure generation.
                - **Why it matters**: Under-reporting of manure generation can lead to improper nutrient management. Our analysis shows that most facilities are reporting manure generation well below established baselines.
                - **How to interpret**:
                    - The base factor is 12.5 tons/cow/year
                    - Calculated factor = Total Manure / (Cows + Heifers×0.6 + Calves×0.3)
                    - Values significantly below the base factor suggest potential under-reporting
                    - Values above may indicate actual higher generation or reporting errors
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
                st.markdown("---")
                st.subheader("Consultant Comparison")
                st.write("""
                This section compares reporting patterns across different consultants and self-reported facilities.
                Each bar represents a consultant's average value, with error bars showing the standard deviation.
                """)
                
                manure_fig, wastewater_fig, nitrogen_fig, facilities_fig = create_consultant_comparison_plots(df)
                if manure_fig is not None:
                    st.plotly_chart(manure_fig, use_container_width=True)
                    st.plotly_chart(wastewater_fig, use_container_width=True)
                    st.plotly_chart(nitrogen_fig, use_container_width=True)
                    st.plotly_chart(facilities_fig, use_container_width=True)
                else:
                    st.warning("No consultant metrics data available.")
                
                # Raw data
                st.subheader("Raw Data")
                st.write("""
                Access the complete dataset for detailed analysis. This data reveals the full extent of reporting discrepancies 
                and can be used for further investigation of compliance issues.
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
            else:
                st.warning("Please select at least one region (R5 or R7) to view comparison plots.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the data format and try again.")

if __name__ == "__main__":
    main() 