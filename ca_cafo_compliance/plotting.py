import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def create_animal_count_histogram():
    """
    Creates a histogram showing the distribution of facilities by number of animals.
    Saves the plot to the outputs folder.
    """

    df = pd.read_csv("../data/cafo_report_data_results.csv")
    
    # Calculate total animals for each facility
    df['total_animals'] = df.apply(
        lambda row: (
            (row['mature_dairy_cows'] if pd.notna(row['mature_dairy_cows']) else 0) +
            (row['other_animals_count'] if pd.notna(row['other_animals_count']) else 0) +
            (row['unspecified_animals'] if pd.notna(row['unspecified_animals']) else 0)
        ), 
        axis=1
    )
    
    df_filtered = df[df['total_animals'] > 0] # filter missing data
    
    plt.figure(figsize=(10, 6))
    plt.hist(df_filtered['total_animals'], bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel('Total Number of Animals')
    plt.ylabel('Number of Facilities')
    plt.title('Distribution of Facilities by Animal Count')
    plt.grid(axis='y', alpha=0.75)
    
    mean_animals = df_filtered['total_animals'].mean()
    median_animals = df_filtered['total_animals'].median()
    plt.axvline(mean_animals, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_animals:.1f}')
    plt.axvline(median_animals, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_animals:.1f}')
    
    plt.legend()
    
    plt.savefig('../outputs/facility_animal_count_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    create_animal_count_histogram()
