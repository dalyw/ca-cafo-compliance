import os
import re
import pandas as pd
import glob

def extract_dairy_info(file_path):
    """
    Extract dairy cow and other animal information from CAFO compliance reports.
    
    Args:
        file_path: Path to the text file to process
        
    Returns:
        Dictionary containing extracted information
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()
    
    # Extract facility name
    facility_name_match = re.search(r'Facility Name:\s*([^\n]+)', content)
    facility_name = facility_name_match.group(1).strip() if facility_name_match else "Unknown"
    
    # Extract mature dairy cows count
    mature_cows_match = re.search(r'Current # of mature dairy cows \(milking \+ dry\):\s*(\d+)', content)
    mature_cows = int(mature_cows_match.group(1)) if mature_cows_match else None
    
    # Extract other animals information
    other_animals_match = re.search(r'Current # and type of other animals:\s*([^\n]+)', content)
    
    other_animals_count = None
    other_animals_type = None
    bred_heifers = None
    heifers = None
    calves = None
    unspecified = None
    
    if other_animals_match:
        other_animals_text = other_animals_match.group(1).strip()
        # Try to extract number and type
        number_type_match = re.search(r'(\d+)\s+(.+)', other_animals_text)
        if number_type_match:
            other_animals_count = int(number_type_match.group(1))
            other_animals_type = number_type_match.group(2).strip()
            
            # Categorize animals based on type
            animal_type_lower = other_animals_type.lower()
            if 'bred heifer' in animal_type_lower:
                bred_heifers = other_animals_count
            elif 'heifer' in animal_type_lower:
                heifers = other_animals_count
            elif 'calv' in animal_type_lower or 'calf' in animal_type_lower or 'young' in animal_type_lower:
                calves = other_animals_count
            else:
                unspecified = other_animals_count
        else:
            other_animals_type = other_animals_text
            # Try to parse animal counts from the description
            if other_animals_text.isdigit():
                unspecified = int(other_animals_text)
    
    # Get filename without extension for reference
    file_name = os.path.basename(file_path)
    
    return {
        'file_name': file_name,
        'facility_name': facility_name,
        'mature_dairy_cows': mature_cows,
        'other_animals_count': other_animals_count,
        'other_animals_type': other_animals_type,
        'bred_heifers': bred_heifers,
        'heifers': heifers,
        'calves': calves,
        'unspecified_animals': unspecified
    }

def process_all_reports(directory_path):
    """
    Process all text files in the specified directory and create a DataFrame.
    
    Args:
        directory_path: Path to directory containing text files
        
    Returns:
        DataFrame with extracted information
    """
    all_data = []
    
    # Get all text files in the directory
    file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
    
    for file_path in file_paths:
        try:
            data = extract_dairy_info(file_path)
            all_data.append(data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    return df

def main():
    directory_path = "../data/R2_txt"
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    
    # Process all reports
    df = process_all_reports(directory_path)
    
    # Save to CSV
    output_file = "../outputs/cafo_report_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Data extracted and saved to {output_file}")
    
    # Display summary
    print(f"\nProcessed {len(df)} reports")
    print(f"Reports with mature dairy cows data: {df['mature_dairy_cows'].notna().sum()}")
    print(f"Reports with other animals data: {df['other_animals_count'].notna().sum()}")
    print(f"Reports with bred heifers: {df['bred_heifers'].notna().sum()}")
    print(f"Reports with heifers: {df['heifers'].notna().sum()}")
    print(f"Reports with calves: {df['calves'].notna().sum()}")
    print(f"Reports with unspecified animals: {df['unspecified_animals'].notna().sum()}")

if __name__ == "__main__":
    main()
