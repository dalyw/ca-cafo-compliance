import csv
import pandas as pd
import os
from conversion_factors import *

# PDF plumber

input_file = "../outputs/cafo_report_data.csv"

def calculate_metrics(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Total manure excreted
    df['total_manure_excreted'] = df.apply(
        lambda row: calculate_total_manure(
            row['mature_dairy_cows'],
            row['heifers'] if pd.notna(row['heifers']) else 0,
            row['calves'] if pd.notna(row['calves']) else 0
        ), axis=1
    )
    
    # Manure produced
    df['manure_conversion_factor'] = 4.1  # Base factor for mature dairy cows
    
    # Ratio of Process Wastewater to Milk Produced
    df['wastewater_to_milk_ratio'] = df.apply(
        lambda row: calculate_wastewater_milk_ratio(
            row['average_milk_production'] if 'average_milk_production' in df.columns else 0,
            row['mature_dairy_cows'],
            row['process_wastewater_generated'] if 'process_wastewater_generated' in df.columns else 0
        ), axis=1
    )
    
    # Calculate USDA Nitrogen Estimate and Ratio
    df['usda_nitrogen_estimate'] = df['total_manure_excreted'] * MANURE_N_CONTENT
    df['ratio_usda_to_reported_n'] = df.apply(
        lambda row: row['usda_nitrogen_estimate'] / row['total_nitrogen_from_manure'] 
        if 'total_nitrogen_from_manure' in df.columns and row['total_nitrogen_from_manure'] != 0 
        else 0, axis=1
    )
    
    # Calculate UCCE Nitrogen Estimate and Ratio
    df['ucce_nitrogen_estimate'] = df.apply(
        lambda row: calculate_ucce_estimate(
            row['mature_dairy_cows'],
            row['heifers'] if pd.notna(row['heifers']) else 0,
            row['calves'] if pd.notna(row['calves']) else 0
        ), axis=1
    )
    df['ratio_ucce_to_reported_n'] = df.apply(
        lambda row: row['ucce_nitrogen_estimate'] / row['total_nitrogen_from_manure']
        if 'total_nitrogen_from_manure' in df.columns and row['total_nitrogen_from_manure'] != 0
        else 0, axis=1
    )
    
    # Save to new CSV file
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def calculate_total_manure(milk_dry_cows, heifers, calves):
    """
    Calculate total manure using the equation:
    (Milk_Cows + Dry_Cows)(x) + (Bred_Heifers + Heifers)(1.5/4.1)(x) + (Calves)(0.5/4.1)(x) = Total_Manure
    Where x is the base manure factor (4.1)
    """
    base_factor = 4.1  # Base manure factor for mature dairy cows
    
    total = (milk_dry_cows * base_factor) + \
            (heifers * HEIFER_FACTOR * base_factor) + \
            (calves * CALF_FACTOR * base_factor)
    
    return total

def calculate_manure_factor(milk_dry_cows, heifers, calves, total_manure):
    """
    Solve for x in the equation:
    (Milk_Cows + Dry_Cows)(x) + (Bred_Heifers + Heifers)(1.5/4.1)(x) + (Calves)(0.5/4.1)(x) = Total_Manure
    """
    
    denominator = milk_dry_cows + (heifers * HEIFER_FACTOR) + (calves * CALF_FACTOR)
    if denominator == 0:
        return 0
    return total_manure / denominator

def calculate_wastewater_milk_ratio(avg_milk_production, total_cows, wastewater):
    """
    Calculate the ratio of process wastewater to milk produced:
    1. Convert milk production to liters and multiply by cow count
    2. Calculate annual milk production
    3. Divide wastewater by annual milk production
    """
    daily_milk_liters = avg_milk_production * LBS_TO_LITERS * total_cows
    annual_milk_liters = daily_milk_liters * DAYS_PER_YEAR
    
    if annual_milk_liters == 0:
        return 0
    return wastewater / annual_milk_liters

def calculate_ucce_estimate(milk_dry_cows, heifers, calves):
    """
    Calculate UCCE Nitrogen Estimate:
    ((Milk_Cows + Dry_Cows) + (Bred_Heifers + Heifers)(1.5/4.1) + (Calves)(0.5/4.1)) * 365
    """
    
    daily_estimate = milk_dry_cows + (heifers * HEIFER_FACTOR) + (calves * CALF_FACTOR)
    return daily_estimate * DAYS_PER_YEAR

if __name__ == "__main__":
    # input_file = input("Enter the path to the input CSV file: ")
    output_file = os.path.splitext(input_file)[0] + "_results.csv"
    calculate_metrics(input_file, output_file)
