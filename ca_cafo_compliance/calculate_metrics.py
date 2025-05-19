import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime
from conversion_factors import *

def safe_calc(df, keys, func, default=np.nan):
    """Safely calculate a value using specified columns and function."""
    if all(k in df.columns for k in keys):
        return func(df)
    return default

def calculate_metrics(df):
    """Calculate all metrics for the dataframe."""
    # Calculate annual milk production
    df['avg_milk_prod_kg_per_cow'] = safe_calc(
        df, ['avg_milk_lb_per_cow_day'],
        lambda d: d['avg_milk_lb_per_cow_day'] * LBS_TO_KG
    )
    df['avg_milk_prod_l_per_cow'] = safe_calc(
        df, ['avg_milk_lb_per_cow_day'],
        lambda d: d['avg_milk_lb_per_cow_day'] * LBS_TO_KG * KG_PER_L_MILK
    )
    df['reported_annual_milk_production_l'] = safe_calc(
        df, ['avg_milk_lb_per_cow_day', 'avg_milk_cows', 'avg_dry_cows'],
        lambda d: d['avg_milk_lb_per_cow_day'] * LBS_TO_KG * KG_PER_L_MILK * (d['avg_milk_cows'].fillna(0) + d['avg_dry_cows'].fillna(0)) * 365
    )

    # Calculate herd size
    herd_keys = [
        "avg_milk_cows", "avg_dry_cows", "avg_bred_heifers",
        "avg_heifers", "avg_calves_4_6_mo", "avg_calves_0_3_mo", "avg_other"
    ]
    df["total_herd_size"] = safe_calc(
        df, herd_keys,
        lambda d: sum(d[k].fillna(0) for k in herd_keys if k in d.columns),
        default=0
    )

    # Calculate nutrient metrics
    nutrient_types = ["n", "p", "k", "salt"]
    for nutrient in nutrient_types:
        # Total Applied
        dry_key = f"applied_{nutrient}_dry_manure_lbs"
        ww_key = f"applied_ww_{nutrient}_lbs"
        total_applied_key = f"total_applied_{nutrient}_lbs"
        df[total_applied_key] = safe_calc(
            df, [dry_key, ww_key],
            lambda d: d[dry_key].fillna(0) + d[ww_key].fillna(0)
        )

        if nutrient == "n":
            dry_key_reported = "total_manure_gen_n_after_nh3_losses_lbs"
        else:
            dry_key_reported = f"total_manure_gen_{nutrient}_lbs"
        ww_key_reported = f"total_ww_gen_{nutrient}_lbs"
        total_reported_key = f"total_reported_{nutrient}_lbs"
        df[total_reported_key] = safe_calc(
            df, [dry_key_reported, ww_key_reported],
            lambda d: d[dry_key_reported].fillna(0) + d[ww_key_reported].fillna(0)
        )

        # Unaccounted for
        exports_key = f"total_exports_{nutrient}_lbs"
        unaccounted_key = f"unaccounted_for_{nutrient}_lbs"
        df[unaccounted_key] = safe_calc(
            df, [dry_key_reported, ww_key_reported, total_applied_key, exports_key],
            lambda d: d[dry_key_reported].fillna(0) + d[ww_key_reported].fillna(0) - d[total_applied_key].fillna(0) - d[exports_key].fillna(0)
        )

    # Calculate wastewater metrics
    df["total_ww_gen_liters"] = safe_calc(
        df, ["total_ww_gen_gals"],
        lambda d: d["total_ww_gen_gals"] * 3.78541
    )
    df["ww_to_reported_milk"] = safe_calc(
        df, ["total_ww_gen_liters", "reported_annual_milk_production_l"],
        lambda d: d["total_ww_gen_liters"] / d["reported_annual_milk_production_l"].replace(0, np.nan)
    )
    df["ww_to_estimated_milk"] = safe_calc(
        df, ["total_ww_gen_liters", "estimated_annual_milk_production_l"],
        lambda d: d["total_ww_gen_liters"] / d["estimated_annual_milk_production_l"].replace(0, np.nan)
    )
    df["wastewater_ratio_discrepancy"] = safe_calc(
        df, ["ww_to_estimated_milk", "ww_to_reported_milk"],
        lambda d: d["ww_to_estimated_milk"] - d["ww_to_reported_milk"]
    )

    # Calculate manure metrics
    manure_keys = [
        "total_manure_excreted_tons", "avg_milk_cows", "avg_dry_cows",
        "avg_bred_heifers", "avg_heifers", "avg_calves_4_6_mo", "avg_calves_0_3_mo"
    ]
    def manure_factor_func(d):
        denom = (
            d["avg_milk_cows"] + d["avg_dry_cows"] +
            (d["avg_bred_heifers"] + d["avg_heifers"]) * HEIFER_FACTOR +
            (d["avg_calves_4_6_mo"] + d["avg_calves_0_3_mo"]) * CALF_FACTOR
        )
        result = d["total_manure_excreted_tons"] / denom
        result[denom <= 0] = np.nan
        return result
    df["calculated_manure_factor"] = safe_calc(df, manure_keys, manure_factor_func)
    df["manure_factor_discrepancy"] = safe_calc(
        df, ["calculated_manure_factor"],
        lambda d: d["calculated_manure_factor"] - BASE_MANURE_FACTOR
    )

    # Calculate nitrogen metrics
    n_key = "total_manure_gen_n_after_nh3_losses_lbs"
    usda_key = "usda_nitrogen_estimate_lbs"
    ucce_key = "ucce_nitrogen_estimate_lbs"
    
    if n_key in df.columns:
        reported_n = df[n_key]
        df["nitrogen_discrepancy"] = safe_calc(df, [usda_key, n_key], lambda d: d[usda_key] - reported_n)
        df["usda_nitrogen_pct_deviation"] = safe_calc(df, [usda_key, n_key], lambda d: (d[usda_key] - reported_n) / reported_n.replace(0, np.nan) * 100)
        df["ucce_nitrogen_pct_deviation"] = safe_calc(df, [ucce_key, n_key], lambda d: (d[ucce_key] - reported_n) / reported_n.replace(0, np.nan) * 100)
        df["USDA Nitrogen % Deviation"] = df["usda_nitrogen_pct_deviation"]
        df["UCCE Nitrogen % Deviation"] = df["ucce_nitrogen_pct_deviation"]
    else:
        for col in ["nitrogen_discrepancy", "usda_nitrogen_pct_deviation", "ucce_nitrogen_pct_deviation", 
                   "USDA Nitrogen % Deviation", "UCCE Nitrogen % Deviation"]:
            df[col] = np.nan

    # Fill NA values with 0 for all calculated columns
    calculated_columns = [
        "total_herd_size", "avg_milk_prod_kg_per_cow", "avg_milk_prod_l_per_cow", "reported_annual_milk_production_l",
        "total_applied_n_lbs", "total_applied_p_lbs", "total_applied_k_lbs", "total_applied_salt_lbs",
        "total_reported_n_lbs", "total_reported_p_lbs", "total_reported_k_lbs", "total_reported_salt_lbs",
        "unaccounted_for_n_lbs", "unaccounted_for_p_lbs", "unaccounted_for_k_lbs", "unaccounted_for_salt_lbs",
        "total_ww_gen_liters", "ww_to_reported_milk", "ww_to_estimated_milk",
        "calculated_manure_factor", "nitrogen_discrepancy", "wastewater_ratio_discrepancy", "manure_factor_discrepancy",
        "usda_nitrogen_pct_deviation", "ucce_nitrogen_pct_deviation"
    ]
    for col in calculated_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df

def calculate_consultant_metrics(df):
    """Calculate average under/over-reporting metrics for each consultant."""
    # Group by consultant
    consultant_groups = df.groupby('Consultant')
    
    metrics = []
    for consultant, group in consultant_groups:
        manure_avg = group['Calculated Manure Factor'].mean() if 'Calculated Manure Factor' in group.columns else None
        manure_std = group['Calculated Manure Factor'].std() if 'Calculated Manure Factor' in group.columns else None
        
        wastewater_avg = group['Wastewater to Milk Ratio'].mean() if 'Wastewater to Milk Ratio' in group.columns else None
        wastewater_std = group['Wastewater to Milk Ratio'].std() if 'Wastewater to Milk Ratio' in group.columns else None
        
        nitrogen_usda_avg = group['USDA Nitrogen % Deviation'].mean() if 'USDA Nitrogen % Deviation' in group.columns else None
        nitrogen_usda_std = group['USDA Nitrogen % Deviation'].std() if 'USDA Nitrogen % Deviation' in group.columns else None
        
        nitrogen_ucce_avg = group['UCCE Nitrogen % Deviation'].mean() if 'UCCE Nitrogen % Deviation' in group.columns else None
        nitrogen_ucce_std = group['UCCE Nitrogen % Deviation'].std() if 'UCCE Nitrogen % Deviation' in group.columns else None
        
        metrics.append({
            'Consultant': consultant,
            'Manure Factor Avg': manure_avg,
            'Manure Factor Std': manure_std,
            'Wastewater Ratio Avg': wastewater_avg,
            'Wastewater Ratio Std': wastewater_std,
            'USDA Nitrogen % Dev Avg': nitrogen_usda_avg,
            'USDA Nitrogen % Dev Std': nitrogen_usda_std,
            'UCCE Nitrogen % Dev Avg': nitrogen_ucce_avg,
            'UCCE Nitrogen % Dev Std': nitrogen_ucce_std,
            'Facility Count': len(group)
        })
    
    return pd.DataFrame(metrics) 