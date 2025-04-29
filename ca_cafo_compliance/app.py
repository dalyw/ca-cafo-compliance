import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

st.set_page_config(page_title="CA CAFO Compliance Dashboard", layout="wide")

def load_data():
    """Load all consolidated data files and combine them."""
    all_files = glob.glob("outputs/consolidated/*.csv")
    if not all_files:
        st.error("No consolidated data files found. Please run consolidate_data.py first.")
        st.stop()
    
    dfs = []
    for file in all_files:
        try:
            # Read CSV and ensure Template column is string type
            df = pd.read_csv(file)
            # Convert Template and other categorical columns to string type
            categorical_cols = ['Year', 'Region', 'County', 'Template']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
    
    if not dfs:
        st.error("No data could be loaded.")
        st.stop()
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def main():
    st.title("California CAFO Compliance Dashboard")
    st.write("Explore compliance data across different regions, counties, and years.")
    
    try:
        # Load data
        df = load_data()
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Year filter
        years = sorted(df['Year'].unique())
        selected_years = st.sidebar.multiselect("Select Years", years, default=years)
        
        # Region filter
        regions = sorted(df['Region'].unique())
        selected_regions = st.sidebar.multiselect("Select Regions", regions, default=regions)
        
        # County filter
        counties = sorted(df['County'].unique())
        selected_counties = st.sidebar.multiselect("Select Counties", counties, default=counties)
        
        # Filter data based on selections
        filtered_df = df[
            (df['Year'].isin(selected_years)) &
            (df['Region'].isin(selected_regions)) &
            (df['County'].isin(selected_counties))
        ]
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(filtered_df))
        with col2:
            st.metric("Regions", len(filtered_df['Region'].unique()))
        
        # Create visualizations
        st.subheader("Records by Region")
        if not filtered_df.empty:
            fig1 = px.bar(
                filtered_df.groupby(['Region', 'Year']).size().reset_index(name='count'),
                x='Region',
                y='count',
                color='Year',
                barmode='group',
                title='Number of Records by Region and Year'
            )
            st.plotly_chart(fig1)
            
            st.subheader("Records by County")
            fig2 = px.bar(
                filtered_df.groupby(['County', 'Region']).size().reset_index(name='count'),
                x='County',
                y='count',
                color='Region',
                title='Number of Records by County and Region'
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2)
        else:
            st.warning("No data available for the selected filters.")
        
        # Display raw data
        st.subheader("Raw Data")
        st.dataframe(filtered_df)
        
        # Download button for filtered data
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