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