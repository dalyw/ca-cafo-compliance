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
