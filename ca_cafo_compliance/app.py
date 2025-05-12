import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import geopandas as gpd
# from calculate_manure_milk import calculate_estimates
from conversion_factors import *

# Streamlit generated and modified with prompts to Claude 3.7, 
# based on the data output from 

st.set_page_config(page_title="CA CAFO Annual Report Data Exploration", layout="centered")

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
                        
                        # Convert categorical columns to string type
                        categorical_cols = ['Year', 'Region', 'County', 'Template', 'Dairy Name', 'Dairy Address', 'filename']
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
                
                # Convert categorical columns to string type
                categorical_cols = ['Year', 'Region', 'County', 'Template', 'Dairy Name', 'Dairy Address', 'filename']
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
    
    # Calculate total herd size
    combined_df['Total Herd Size'] = combined_df.apply(
        lambda row: sum([
            row.get('Average Milk Cows', 0) or 0,
            row.get('Average Dry Cows', 0) or 0,
            row.get('Average Bred Heifers', 0) or 0,
            row.get('Average Heifers', 0) or 0,
            row.get('Average Calves (4-6 mo.)', 0) or 0,
            row.get('Average Calves (0-3 mo.)', 0) or 0
        ]), axis=1
    )
    
    return combined_df

def create_map(df, selected_year):
    """Create a map visualization of facilities with regional board boundaries."""
    year_df = df[df['Year'] == str(selected_year)]
    
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
            'Dairy Address': True,
            'Total Herd Size': True,
            'County': True,
            'Latitude': False,
            'Longitude': False
        },
        title=f'CAFO Facilities in California ({selected_year})',
        size_max=30,
        zoom=5.5,
        center={"lat": 37.2719, "lon": -119.2702}
    )
    
    # # Add regional board boundaries from ArcGIS REST service
    # fig.update_layout(
    #     mapbox={
    #         'style': "carto-positron",
    #         'center': {"lat": 37.2719, "lon": -119.2702},
    #         'zoom': 5.5
    #     },
    #     margin={"r": 0, "t": 30, "l": 0, "b": 0},
    #     height=600
    # )
    
    # Fetch GeoJSON data for regional water board boundaries
    geojson_url = "https://services.arcgis.com/BLN4oKB0N1YSgvY8/arcgis/rest/services/Regional_Water_Quality_Control_Boards_Regions/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson"
    
    try:
        import requests
        response = requests.get(geojson_url)
        geojson_data = response.json()
        
        # Add choropleth layer for regional boundaries
        fig.add_choroplethmapbox(
            geojson=geojson_data,
            locations=[feature['properties']['Map_cont_1'] for feature in geojson_data['features']],
            z=[1] * len(geojson_data['features']),  # Uniform value for all regions
            featureidkey="properties.Map_cont_1",
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # Transparent fill
            showscale=False,
            marker_line_width=2,
            marker_line_color='rgba(50, 100, 150, 0.8)',
            hoverinfo='skip'
        )
    except Exception as e:
        st.warning(f"Could not load regional boundaries: {str(e)}")
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
        return None, None, None
    
    # Color scheme for consistency
    reported_color = 'rgb(0, 71, 171)'      # dark blue
    estimated_color = 'rgb(135, 206, 235)'  # light blue
    ucce_color = 'rgb(144, 238, 144)'       # light green
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Wastewater Comparison",
            "Manure Factor Comparison",
            "Nitrogen Generation Comparison"
        )
    )
    
    # 1. Wastewater Comparison
    fig.add_trace(
        go.Bar(
            name="Reported",
            x=["Reported"],
            y=[facility_df['Wastewater (L/day)'].iloc[0]],
            marker_color=reported_color
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            name="Estimated",
            x=["Estimated"],
            y=[facility_df['Estimated Wastewater (L/day)'].iloc[0]],
            marker_color=estimated_color
        ),
        row=1, col=1
    )
    
    # 2. Manure Factor Comparison
    fig.add_trace(
        go.Bar(
            name="Calculated",
            x=["Calculated"],
            y=[facility_df['Calculated Manure Factor'].iloc[0]],
            marker_color=reported_color
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            name="Base Factor",
            x=["Base"],
            y=[BASE_MANURE_FACTOR],
            marker_color=estimated_color
        ),
        row=1, col=2
    )
    
    # 3. Nitrogen Generation Comparison
    fig.add_trace(
        go.Bar(
            name="Reported",
            x=["Reported"],
            y=[facility_df['Nitrogen Generation (kg/day)'].iloc[0]],
            marker_color=reported_color
        ),
        row=1, col=3
    )
    fig.add_trace(
        go.Bar(
            name="USDA Estimate",
            x=["USDA"],
            y=[facility_df['USDA Nitrogen Estimate (kg/day)'].iloc[0]],
            marker_color=estimated_color
        ),
        row=1, col=3
    )
    fig.add_trace(
        go.Bar(
            name="UCCE Estimate",
            x=["UCCE"],
            y=[facility_df['UCCE Nitrogen Estimate (kg/day)'].iloc[0]],
            marker_color=ucce_color
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text=f"Facility Comparison: {facility_name}",
        barmode='group'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Liters per day", row=1, col=1)
    fig.update_yaxes(title_text="Tons per cow per year", row=1, col=2)
    fig.update_yaxes(title_text="kg per day", row=1, col=3)
    
    return fig

def main():
    st.title("California Dairy CAFO Annual Report Data Exploration")
    st.write("Explore compliance data across different regions, counties, and years.")
    
    try:
        # Load and process data
        df = load_data()
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Years filter (single select)
        years = sorted(df['Year'].unique())
        selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)  # Default to most recent year
        
        # Regions filter
        regions = sorted(df['Region'].unique())
        selected_regions = st.sidebar.multiselect("Select Regions", regions, default=regions)
        
        # Filter counties based on selected regions
        available_counties = sorted(df[df['Region'].isin(selected_regions)]['County'].unique())
        selected_counties = st.sidebar.multiselect(
            "Select Counties", 
            available_counties,
            default=available_counties
        )
        
        # Filter templates based on selected regions
        available_templates = sorted(df[df['Region'].isin(selected_regions)]['Template'].unique())
        selected_templates = st.sidebar.multiselect(
            "Select Templates", 
            available_templates,
            default=available_templates
        )
        
        # Filter data
        filtered_df = df[
            (df['Year'] == selected_year) &
            (df['Region'].isin(selected_regions)) &
            (df['County'].isin(selected_counties)) &
            (df['Template'].isin(selected_templates))
        ]
        
        # Display map
        st.subheader("Facility Locations")
        if not filtered_df.empty and 'Latitude' in filtered_df.columns:
            st.metric("Total Animals", f"{filtered_df['Total Herd Size'].sum():,.0f}")
            map_fig = create_map(filtered_df, selected_year)
            if map_fig is not None:
                st.plotly_chart(map_fig, use_container_width=True, height=800)  # Increased height
        else:
            st.warning("No location data available for the selected filters.")
        
        # Add spacing between map and comparison charts
        st.markdown("---")  # Add a horizontal divider
        st.markdown("<br>", unsafe_allow_html=True)  # Add some vertical space
        
        # Comparison plots
        st.subheader("Estimated vs Actual Comparisons")
        nitrogen_fig, wastewater_fig, manure_fig = create_comparison_plots(filtered_df)
        
        st.plotly_chart(nitrogen_fig, use_container_width=True)
        st.plotly_chart(wastewater_fig, use_container_width=True)
        st.plotly_chart(manure_fig, use_container_width=True)
        
        # Add spacing before facility search
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Facility search and comparison
        st.subheader("Facility Search")
        facility_names = sorted(filtered_df['Dairy Name'].unique())
        selected_facility = st.selectbox("Select a Facility", facility_names)
        
        if selected_facility:
            facility_comparison_fig = create_facility_comparison_plots(filtered_df, selected_facility)
            if facility_comparison_fig is not None:
                st.plotly_chart(facility_comparison_fig, use_container_width=True)
            else:
                st.warning(f"No data available for {selected_facility}")
        
        # Raw data
        st.subheader("Raw Data")
        st.dataframe(filtered_df)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="filtered_cafo_data.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the data format and try again.")

if __name__ == "__main__":
    main() 