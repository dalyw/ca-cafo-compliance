import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import geopandas as gpd
from calculate_manure_milk import calculate_estimates

# Streamlit generated and modified with prompts to Claude 3.7, 
# based on the data output from 

st.set_page_config(page_title="CA CAFO Annual Report Data Exploration", layout="centered")

def load_data():
    """Load all consolidated data files and combine them."""
    all_files = glob.glob("outputs/consolidated/*.csv")
    if not all_files:
        st.error("No consolidated data files found. Please run consolidate_data.py first.")
        st.stop()
    
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
    
    # Calculate estimates
    estimates = pd.DataFrame([calculate_estimates(row) for _, row in combined_df.iterrows()])
    combined_df = pd.concat([combined_df, estimates], axis=1)
    
    return combined_df
def create_map(df, selected_year):
    """Create a map visualization of facilities with regional board boundaries."""
    year_df = df[df['Year'] == str(selected_year)]
    
    # Create the base map with facilities
    fig = px.scatter_mapbox(
        year_df[year_df['Latitude'].notna()],
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
    
    # Add regional board boundaries from ArcGIS REST service
    fig.update_layout(
        mapbox={
            'style': "carto-positron",
            'center': {"lat": 37.2719, "lon": -119.2702},
            'zoom': 5.5
        },
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=600
    )
    
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
    """Create comparison plots between estimated and actual values showing largest discrepancies."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Manure Generation (tons/year)",
            "Nitrogen Generation (lbs/year)",
            "Wastewater to Milk Ratio",
            "Nitrogen Estimate"
        )
    )
    
    # Color scheme for consistency
    estimated_color = 'rgb(135, 206, 235)'  # light blue
    reported_color = 'rgb(0, 71, 171)'      # dark blue
    
    # 1. Manure Generation - Top 10 discrepancies
    df['Manure_Discrepancy'] = abs(df['Estimated Total Manure (tons)'] - df['Total Manure Excreted (tons)'])
    top_manure = df.nlargest(10, 'Manure_Discrepancy')
    
    fig.add_trace(
        go.Bar(
            name="Estimated",
            x=top_manure['Dairy Name'],
            y=top_manure['Estimated Total Manure (tons)'],
            marker_color=estimated_color
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            name="Reported",
            x=top_manure['Dairy Name'],
            y=top_manure['Total Manure Excreted (tons)'],
            marker_color=reported_color
        ),
        row=1, col=1
    )
    
    # 2. Nitrogen Generation - Top 10 discrepancies
    df['Nitrogen_Discrepancy'] = abs(df['USDA Nitrogen Estimate (lbs)'] - 
                                   df['Total Dry Manure Generated N After Ammonia Losses (lbs)'])
    top_nitrogen = df.nlargest(10, 'Nitrogen_Discrepancy')
    
    fig.add_trace(
        go.Bar(
            name="Estimated (USDA)",
            x=top_nitrogen['Dairy Name'],
            y=top_nitrogen['USDA Nitrogen Estimate (lbs)'],
            marker_color=estimated_color,
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            name="Reported",
            x=top_nitrogen['Dairy Name'],
            y=top_nitrogen['Total Dry Manure Generated N After Ammonia Losses (lbs)'],
            marker_color=reported_color,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Wastewater to Milk Ratio - Top 10 outliers
    # For ratio, we'll just show the most extreme values since it's a single metric
    top_ratio = df.nlargest(10, 'Wastewater to Milk Ratio')
    
    fig.add_trace(
        go.Bar(
            name="Ratio",
            x=top_ratio['Dairy Name'],
            y=top_ratio['Wastewater to Milk Ratio'],
            marker_color=reported_color,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. USDA vs UCCE Nitrogen - Top 10 discrepancies
    df['USDA_UCCE_Discrepancy'] = abs(df['USDA Nitrogen Estimate (lbs)'] - df['UCCE Nitrogen Estimate (lbs)'])
    top_estimates = df.nlargest(10, 'USDA_UCCE_Discrepancy')
    
    fig.add_trace(
        go.Bar(
            name="USDA Estimate",
            x=top_estimates['Dairy Name'],
            y=top_estimates['USDA Nitrogen Estimate (lbs)'],
            marker_color=estimated_color,
            showlegend=False
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(
            name="UCCE Estimate",
            x=top_estimates['Dairy Name'],
            y=top_estimates['UCCE Nitrogen Estimate (lbs)'],
            marker_color=reported_color,
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=1000,  # Made taller to accommodate long facility names
        showlegend=True,
        barmode='group',
        title_text="Largest Discrepancies Between Estimated and Reported Values",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update all x-axes for better readability
    fig.update_xaxes(tickangle=45)
    
    # Add y-axis labels
    fig.update_yaxes(title_text="Tons per Year", row=1, col=1)
    fig.update_yaxes(title_text="Pounds per Year", row=1, col=2)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Pounds per Year", row=2, col=2)
    
    return fig

def main():
    st.title("California Dairy CAFO Annual Report Data Exploration")
    st.write("Explore compliance data across different regions, counties, and years.")
    
    try:
        # Load and process data
        df = load_data()
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        years = sorted(df['Year'].unique())
        selected_years = st.sidebar.multiselect("Select Years", years, default=years)
        
        regions = sorted(df['Region'].unique())
        selected_regions = st.sidebar.multiselect("Select Regions", regions, default=regions)
        
        counties = sorted(df['County'].unique())
        selected_counties = st.sidebar.multiselect("Select Counties", counties, default=counties)
        
        templates = sorted(df['Template'].unique())
        selected_templates = st.sidebar.multiselect("Select Templates", templates, default=templates)
        
        # Filter data
        filtered_df = df[
            (df['Year'].isin(selected_years)) &
            (df['Region'].isin(selected_regions)) &
            (df['County'].isin(selected_counties)) &
            (df['Template'].isin(selected_templates))
        ]
        
        # Display map
        st.subheader("Facility Locations")
        if not filtered_df.empty and 'Latitude' in filtered_df.columns:
            map_year = st.selectbox("Select Year for Map", sorted(filtered_df['Year'].unique(), reverse=True))
            st.metric("Total Animals", f"{filtered_df['Total Herd Size'].sum():,.0f}")
            st.plotly_chart(create_map(filtered_df, map_year), use_container_width=True)
        else:
            st.warning("No location data available for the selected filters.")
        
        # Comparison plots
        st.subheader("Estimated vs Actual Comparisons")
        st.plotly_chart(create_comparison_plots(filtered_df), use_container_width=True)
        
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