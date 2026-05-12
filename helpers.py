import pandas as pd

PATH_TO_PDF_DATA = "/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/Manure Trucking Network Analysis/data"
PARAMETERS_DF = pd.read_csv("data/parameters.csv")


def build_parameter_dicts(manifest_only=False):
    """Build parameter mapping dicts from parameters.csv."""
    df = (
        PARAMETERS_DF[PARAMETERS_DF["manifest_type"].isin(["manure", "wastewater", "both"])]
        if manifest_only
        else PARAMETERS_DF
    )
    return {
        "key_to_name": dict(zip(df["parameter_key"], df["parameter_name"])),
        "key_to_type": dict(zip(df["parameter_key"], df["data_type"])),
        "key_to_default": dict(zip(df["parameter_key"], df["default"])),
    }


def coerce_columns(df):
    """Coerce columns to their data_type from parameters.csv (in-place)."""
    for dtype, names in PARAMETERS_DF.groupby("data_type")["parameter_name"].apply(set).items():
        cols = [c for c in df.columns if c in names]
        if not cols:
            continue
        if dtype == "numeric":
            df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        elif dtype == "date":
            for c in cols:
                df[c] = pd.to_datetime(df[c], format="mixed", dayfirst=False, errors="coerce")
        elif dtype == "boolean":
            for c in cols:
                df[c] = (
                    df[c]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map({"TRUE": True, "FALSE": False, "NAN": None})
                )
    return df
