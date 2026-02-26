import pandas as pd
import numpy as np
from helpers_pdf_metrics import cf

parameters = pd.read_csv("ca_cafo_compliance/data/parameters.csv")


def calculate_metrics(df):
    """Calculate all metrics for the dataframe."""
    # Initialize all parameters as NA if they don't exist
    parameters = pd.read_csv("ca_cafo_compliance/data/parameters.csv")
    all_params = parameters["parameter_key"].unique().tolist()
    for param in all_params:
        if param not in df.columns:
            df[param] = np.nan

    # Calculate annual milk production
    df["avg_milk_prod_kg_per_cow"] = df["avg_milk_lb_per_cow_day"] * cf["LBS_TO_KG"]
    df["avg_milk_prod_l_per_cow"] = (
        df["avg_milk_lb_per_cow_day"] * cf["LBS_TO_KG"] * cf["KG_PER_L_MILK"]
    )
    df["reported_annual_milk_production_l"] = (
        df["avg_milk_lb_per_cow_day"]
        * cf["LBS_TO_KG"]
        * cf["KG_PER_L_MILK"]
        * (df["avg_milk_cows"] + df["avg_dry_cows"])
        * cf["DAYS_PER_YEAR"]
    )

    # Calculate herd size - fill NA column values as 0
    df["total_herd_size"] = (
        df["avg_milk_cows"].fillna(0)
        + df["avg_dry_cows"].fillna(0)
        + df["avg_bred_heifers"].fillna(0)
        + df["avg_heifers"].fillna(0)
        + df["avg_calves_4_6_mo"].fillna(0)
        + df["avg_calves_0_3_mo"].fillna(0)
        + df["avg_other"].fillna(0)
    )

    # Calculate nutrient metrics
    nutrient_types = ["n", "p", "k", "salt"]
    for nutrient in nutrient_types:
        # Total Applied
        dry_key = f"applied_{nutrient}_dry_manure_lbs"
        ww_key = f"applied_ww_{nutrient}_lbs"
        df[f"total_applied_{nutrient}_lbs"] = df[dry_key].fillna(0) + df[ww_key].fillna(
            0
        )

        # Total Reported
        dry_key_reported = (
            "total_manure_gen_n_after_nh3_losses_lbs"
            if nutrient == "n"
            else f"total_manure_gen_{nutrient}_lbs"
        )
        ww_key_reported = f"total_ww_gen_{nutrient}_lbs"
        df[f"total_reported_{nutrient}_lbs"] = df[dry_key_reported].fillna(0) + df[
            ww_key_reported
        ].fillna(0)

        # Unaccounted for
        exports_key = f"total_exports_{nutrient}_lbs"
        df[f"unaccounted_for_{nutrient}_lbs"] = (
            df[dry_key_reported].fillna(0)
            + df[ww_key_reported].fillna(0)
            - df[f"total_applied_{nutrient}_lbs"]
            - df[exports_key].fillna(0)
        )

    # Calculate wastewater metrics
    df["total_ww_gen_liters"] = df["total_ww_gen_gals"] * 3.78541

    # Calculate estimated annual milk production (L) - use NA for missing data
    est_milk_col = "estimated_annual_milk_production_l"

    # Check if we have milk production data
    has_milk_data = (
        "avg_milk_lb_per_cow_day" in df.columns
        and df["avg_milk_lb_per_cow_day"].notna().any()
    )

    if has_milk_data:
        # Use actual milk production data when available
        df[est_milk_col] = (
            df["avg_milk_lb_per_cow_day"]
            * cf["LBS_TO_KG"]
            * cf["KG_PER_L_MILK"]
            * (df["avg_milk_cows"].fillna(0) + df["avg_dry_cows"].fillna(0))
            * cf["DAYS_PER_YEAR"]
        )
        # Set to NA where we don't have milk production data
        df.loc[df["avg_milk_lb_per_cow_day"].isna(), est_milk_col] = np.nan
    else:
        # Use default milk production when no data available
        df[est_milk_col] = (
            cf["DEFAULT_MILK_PRODUCTION"]
            * (df["avg_milk_cows"].fillna(0) + df["avg_dry_cows"].fillna(0))
            * cf["DAYS_PER_YEAR"]
            * cf["LBS_TO_KG"]
            * cf["KG_PER_L_MILK"]
        )

    # Calculate wastewater metrics
    df["wastewater_estimated"] = df[est_milk_col] * cf["L_WW_PER_L_MILK_LOW"]

    # Calculate ratios - use NA for division by zero/missing
    df["wastewater_to_reported"] = np.where(
        df["reported_annual_milk_production_l"].notna()
        & (df["reported_annual_milk_production_l"] > 0),
        df["total_ww_gen_liters"] / df["reported_annual_milk_production_l"],
        np.nan,
    )

    df["wastewater_to_estimated"] = np.where(
        df[est_milk_col].notna() & (df[est_milk_col] > 0),
        df["total_ww_gen_liters"] / df[est_milk_col],
        np.nan,
    )

    df["wastewater_ratio_discrepancy"] = (
        df["wastewater_to_estimated"] - df["wastewater_to_reported"]
    )

    # Calculate manure metrics - fill NA herd size values as 0
    denom = (
        df["avg_milk_cows"].fillna(0)
        + df["avg_dry_cows"].fillna(0)
        + (df["avg_bred_heifers"].fillna(0) + df["avg_heifers"].fillna(0))
        * cf["HEIFER_FACTOR"]
        + (df["avg_calves_4_6_mo"].fillna(0) + df["avg_calves_0_3_mo"].fillna(0))
        * cf["CALF_FACTOR"]
    )
    df["calculated_manure_factor"] = df["total_manure_gen_tons"] / denom
    df.loc[denom <= 0, "calculated_manure_factor"] = np.nan
    df["manure_factor_discrepancy"] = (
        df["calculated_manure_factor"] - cf["MANURE_FACTOR_AVERAGE"]
    )

    # Calculate nitrogen metrics
    df["usda_nitrogen_estimate_lbs"] = (
        df["total_manure_gen_tons"] * cf["MANURE_N_CONTENT"] * 2000
    )  # tons to lbs
    df["ucce_nitrogen_estimate_lbs"] = (
        (df["avg_milk_cows"] + df["avg_dry_cows"]) * cf["MANURE_FACTOR_AVERAGE"]
        + (df["avg_bred_heifers"] + df["avg_heifers"])
        * cf["HEIFER_FACTOR"]
        * cf["MANURE_FACTOR_AVERAGE"]
        + (df["avg_calves_4_6_mo"] + df["avg_calves_0_3_mo"])
        * cf["CALF_FACTOR"]
        * cf["MANURE_FACTOR_AVERAGE"]
    ) * cf["MANURE_N_CONTENT"]

    if "total_manure_gen_n_after_nh3_losses_lbs" in df.columns:
        reported_n = df["total_manure_gen_n_after_nh3_losses_lbs"]
        df["nitrogen_discrepancy"] = reported_n - df["usda_nitrogen_estimate_lbs"]
        df["usda_nitrogen_pct_deviation"] = (
            (reported_n - df["usda_nitrogen_estimate_lbs"])
            / df["usda_nitrogen_estimate_lbs"].replace(0, np.nan)
            * 100
        )
        df["ucce_nitrogen_pct_deviation"] = (
            (reported_n - df["ucce_nitrogen_estimate_lbs"])
            / df["ucce_nitrogen_estimate_lbs"].replace(0, np.nan)
            * 100
        )
    else:
        for col in [
            "nitrogen_discrepancy",
            "usda_nitrogen_pct_deviation",
            "ucce_nitrogen_pct_deviation",
        ]:
            df[col] = np.nan

    return df


consultant_etric_keys = [
    "manure_factor",
    "wastewater_ratio",
    "usda_nitrogen_pct_dev",
    "ucce_nitrogen_pct_dev",
]


def calculate_consultant_metrics(df):
    """Calculate average under/over-reporting metrics for each consultant."""
    metrics = []
    for template, group in df.groupby("template"):
        metrics.append(
            {
                "template": template,
                "manure_factor_avg": group["calculated_manure_factor"].mean(),
                "manure_factor_std": group["calculated_manure_factor"].std(),
                "wastewater_ratio_avg": group["wastewater_to_reported"].mean(),
                "wastewater_ratio_std": group["wastewater_to_reported"].std(),
                "usda_nitrogen_pct_dev_avg": group[
                    "usda_nitrogen_pct_deviation"
                ].mean(),
                "usda_nitrogen_pct_dev_std": group["usda_nitrogen_pct_deviation"].std(),
                "ucce_nitrogen_pct_dev_avg": group[
                    "ucce_nitrogen_pct_deviation"
                ].mean(),
                "ucce_nitrogen_pct_dev_std": group["ucce_nitrogen_pct_deviation"].std(),
                "facility_count": len(group),
            }
        )
    return pd.DataFrame(metrics)
