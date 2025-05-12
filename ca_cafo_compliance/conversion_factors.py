# Unit conversion factors
LBS_TO_LITERS = 0.453592 * 1  # 1 lb = 0.453592 kg, assuming 1 kg milk = 1 liter
LBS_TO_KG = 0.453592  # kg/lb
KG_TO_L = 0.971  # L/kg at 20 degrees C
KG_TO_L_MILK = 1.03  # density of milk

# Time constants
DAYS_PER_YEAR = 365

# Animal factors
HEIFER_FACTOR = (1.5/4.1)
CALF_FACTOR = (0.5/4.1)
BASE_MANURE_FACTOR = 4.1  # Base manure factor for mature dairy cows
DEFAULT_MILK_PRODUCTION = 68  # lbs per cow per day

# Nutrient content
MANURE_N_CONTENT = 12.92  # Nitrogen content factor for USDA estimate

# OCR settings
OCR_DPI = 200
OCR_FAST_MODE = True
OCR_NUM_CORES = 3

# Data years and regions
YEARS = [2023, 2024]
REGIONS = ['R2', 'R3', 'R5', 'R7', 'R8']

# File paths
GEOCODING_CACHE_FILE = "outputs/geocoding_cache.json"

# Consultant name mapping
consultant_mapping = {
    'general_order': 'Self-completed',
    'generic_r7': 'Self-completed',
    'generic_r2': 'Self-completed',
    'innovative_ag': 'Innovative Ag',
    'livingston': 'Livingston',
    'provost_pritchard': 'Provost Prichard'
}