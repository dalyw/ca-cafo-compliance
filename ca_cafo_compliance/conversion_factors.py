# Unit conversion factors
LBS_TO_LITERS = 0.453592 * 1  # 1 lb = 0.453592 kg, assuming 1 kg milk = 1 liter
LBS_TO_KG = 0.453592  # kg/lb
KG_TO_L = 0.971  # L/kg at 20 degrees C
KG_PER_L_MILK = 1.03  # density of milk
KG_PER_L_WW = 1.00 # density of WW
TONS_TO_KG = 1000 # kg/ton
M3_TO_L = 1000 # L/m3

DAYS_PER_YEAR = 365
YEARS = [2023, 2024]
REGIONS = ['R1', 'R2', 'R5', 'R7', 'R8']

HEIFER_FACTOR = (1.5/4.1)
CALF_FACTOR = (0.5/4.1)
BASE_MANURE_FACTOR = 4.1  # Base manure factor for mature dairy cows
DEFAULT_MILK_PRODUCTION = 68  # lbs per cow per day

MANURE_N_CONTENT = 12.92  # Nitrogen content factor for USDA estimate

# OCR settings
OCR_DPI = 200
OCR_FAST_MODE = True
OCR_NUM_CORES = 3


M3_WW_PER_TON_MILK_LOW = 1
M3_WW_PER_TON_MILK_HIGH = 2
# kg wastewater per ton milk to L wastewater per L milk
L_WW_PER_L_MILK_LOW = M3_WW_PER_TON_MILK_LOW / TONS_TO_KG * KG_PER_L_MILK * M3_TO_L  # 0.4 kg WW per 1000 kg milk
L_WW_PER_L_MILK_HIGH = M3_WW_PER_TON_MILK_HIGH / TONS_TO_KG * KG_PER_L_MILK * M3_TO_L  # 60 kg WW per 1000 kg milk
# For each ton of raw milk processed, the dairy industry generates anywhere from 0.4 to 60 m³ of wastewater that typically has been discarded.
# https://www.fluencecorp.com/generating-power-from-dairy-waste/
# 1-2 m3 / ton
# https://envirochemie.com/cms/upload/downloads-en/fachbeitraege/Whitepaper_Wastewater_treatment_in_the_dairy_processing_industry_.pdf

# File paths
GEOCODING_CACHE_FILE = "outputs/geocoding_cache.json"

consultant_mapping = {
    'general_order': 'Self-completed',
    'generic_r7': 'Self-completed',
    'generic_r2': 'Self-completed',
    'innovative_ag': 'Innovative Ag',
    'livingston': 'Livingston',
    'provost_pritchard': 'Provost Prichard'
}