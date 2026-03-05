import os
import requests
from io import StringIO
import pandas as pd
import streamlit as st

# Color palette (from ColorBrewer)
manure_colors = ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3']
wastewater_colors = ['#c7eae5', '#80cdc1', '#35978f', '#01665e']

PALETTE = {
    "blue": "#1f78b4",
    "orange": "#bf812d",
    "green": "#01665e",
    "purple": "#9467bd",
    "brown": "#8c510a",
    "pink": "#fb9a99",
    "gray": "#7f7f7f",
    "yellow": "#b2df8a",
    "teal": "#a6cee3",
}

# Manifest type color map for plotly
MANIFEST_TYPE_COLORS = {
    "manure": manure_colors[0],
    "wastewater": wastewater_colors[2],
    "both": PALETTE["purple"],
}

# Per-type color sequences for charts that loop over manure/wastewater
TYPE_COLOR_SEQ = {
    "Manure": manure_colors,
    "Wastewater": wastewater_colors,
}

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
            st.warning(
                f"Could not load {os.path.basename(local_path)} from local or GitHub."
            )
            return pd.DataFrame()
