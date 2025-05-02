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
    df['usda_nitrogen_estimate'], df['ucce_nitrogen_estimate'] = zip(*df.apply(
        lambda row: calculate_nitrogen_estimates(
            row['mature_dairy_cows'],
            row['heifers'] if pd.notna(row['heifers']) else 0,
            row['calves'] if pd.notna(row['calves']) else 0
        ), axis=1
    ))
    df['ratio_usda_to_reported_n'] = df.apply(
        lambda row: row['usda_nitrogen_estimate'] / row['total_nitrogen_from_manure'] 
        if 'total_nitrogen_from_manure' in df.columns and row['total_nitrogen_from_manure'] != 0 
        else 0, axis=1
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

def calculate_nitrogen_estimates(milk_dry_cows, heifers, calves):
    """
    Calculate both USDA and UCCE nitrogen estimates
    Returns: (usda_estimate, ucce_estimate)
    """
    # USDA estimate based on total manure
    total_manure = calculate_total_manure(milk_dry_cows, heifers, calves)
    usda_estimate = total_manure * MANURE_N_CONTENT
    
    # UCCE estimate based on animal units
    daily_estimate = milk_dry_cows + (heifers * HEIFER_FACTOR) + (calves * CALF_FACTOR)
    ucce_estimate = daily_estimate * DAYS_PER_YEAR
    
    return usda_estimate, ucce_estimate

def calculate_estimates(row):
    """Calculate estimated values for manure, nitrogen, and wastewater metrics.
    Returns a dictionary of estimates that can be added to the dataframe."""
    
    # Get animal counts, replacing NaN with 0
    milk_dry_cows = (row.get('Average Milk Cows', 0) or 0) + (row.get('Average Dry Cows', 0) or 0)
    heifers = (row.get('Average Bred Heifers', 0) or 0) + (row.get('Average Heifers', 0) or 0)
    calves = (row.get('Average Calves (4-6 mo.)', 0) or 0) + (row.get('Average Calves (0-3 mo.)', 0) or 0)
    
    # Calculate manure generation using base factor of 4.1 tons/cow/year
    base_factor = 4.1
    estimated_manure = (milk_dry_cows * base_factor) + \
                      (heifers * HEIFER_FACTOR * base_factor) + \
                      (calves * CALF_FACTOR * base_factor)
    
    # Calculate nitrogen estimates
    # USDA estimate based on manure generation
    usda_nitrogen = estimated_manure * MANURE_N_CONTENT
    
    # UCCE estimate based on animal units
    animal_units = milk_dry_cows + (heifers * HEIFER_FACTOR) + (calves * CALF_FACTOR)
    ucce_nitrogen = animal_units * DAYS_PER_YEAR
    
    # Calculate wastewater to milk ratio if data available
    wastewater_ratio = 0
    if all(x in row for x in ['Average Milk Production (lb/cow/day)', 'Total Process Wastewater Generated (gals)']):
        milk_production = row['Average Milk Production (lb/cow/day)']
        wastewater = row['Total Process Wastewater Generated (gals)']
        
        # Convert milk to liters and calculate annual production
        daily_milk_liters = milk_production * LBS_TO_LITERS * milk_dry_cows
        annual_milk_liters = daily_milk_liters * DAYS_PER_YEAR
        
        # Calculate ratio if milk production is non-zero
        if annual_milk_liters > 0:
            wastewater_ratio = wastewater / annual_milk_liters
    
    return {
        'Estimated Total Manure (tons)': estimated_manure,
        'USDA Nitrogen Estimate (lbs)': usda_nitrogen,
        'UCCE Nitrogen Estimate (lbs)': ucce_nitrogen,
        'Wastewater to Milk Ratio': wastewater_ratio
    }

if __name__ == "__main__":
    # input_file = input("Enter the path to the input CSV file: ")
    output_file = os.path.splitext(input_file)[0] + "_results.csv"
    calculate_metrics(input_file, output_file)
