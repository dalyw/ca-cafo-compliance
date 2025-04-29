#!/usr/bin/env python3

import pandas as pd
from pypdf import PdfReader
import pdfplumber
import sys
import glob
import numpy as np
import os
import re
import contextlib
from conversion_factors import *

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output"""
    stderr = sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = stderr

def extract_text_from_rectangle(pdf_path, page_number, rect):
    print(f"\nDEBUG: Opening PDF for rectangle extraction: {pdf_path}")
    with suppress_stderr(), pdfplumber.open(pdf_path) as pdf:
        print(f"DEBUG: Getting page {page_number} for rectangle extraction")
        page = pdf.pages[page_number]
        page_width = float(page.width)
        page_height = float(page.height)
        x0, y0, x1, y1 = rect
        
        if x0 < 0 or x1 > page_width or y0 < 0 or y1 > page_height:
            print(f"Coordinates ({x0}, {y0}, {x1}, {y1}) outside page bounds")
            x0 = max(0, min(x0, page_width))
            x1 = max(0, min(x1, page_width))
            y0 = max(0, min(y0, page_height))
            y1 = max(0, min(y1, page_height))
        
        # Extract text from the specified rectangular region
        try:
            print("DEBUG: Extracting text from rectangle")
            text = page.within_bbox((x0, y0, x1, y1)).extract_text()
            return text
        except ValueError as e:
            print(e)
            return ""

def get_pdf_page_dimensions(pdf_path, page_number=0):
    print(f"\nDEBUG: Getting PDF dimensions for: {pdf_path}")
    with suppress_stderr():
        reader = PdfReader(pdf_path)
        page = reader.pages[page_number]
        media_box = page.get('/MediaBox')
    
    if media_box is not None:
        # Convert the MediaBox values to float
        width = float(media_box[2]) - float(media_box[0])  # x1 - x0
        height = float(media_box[3]) - float(media_box[1])  # y1 - y0
        print(f"DEBUG: PDF dimensions - width: {width}, height: {height}")
        return width, height
    else:
        print("DEBUG: No MediaBox found")
        return None, None  # Handle case where MediaBox is not found

def is_convertible_to_float(s):
    # Remove commas and whitespace
    s = s.replace(',', '').strip()
    
    # Regular expression pattern for a valid float
    pattern = r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$'
    
    # Check if the string matches the pattern
    if re.match(pattern, s):
        return True
    else:
        return False

def normalize_text(text):
    """Normalize text by removing extra spaces and standardizing characters."""
    if not text:
        return ""
    # Remove extra whitespace and normalize spaces
    text = ' '.join(text.split())
    # Replace common problematic characters
    text = text.replace('–', '-').replace(''', "'").replace('"', '"').replace('"', '"')
    return text

def text_contains(text, search_phrase):
    """Check if text contains a phrase, handling various formatting issues."""
    if not text or not search_phrase:
        return False
    
    # Normalize both texts
    text = normalize_text(text)
    search_phrase = normalize_text(search_phrase)
    
    # Try exact match first
    if search_phrase in text:
        return True
    
    # Try case-insensitive match
    if search_phrase.lower() in text.lower():
        return True
    
    # Try with flexible spacing
    search_words = search_phrase.split()
    text_words = text.split()
    
    # Check for consecutive words appearing in order
    for i in range(len(text_words) - len(search_words) + 1):
        match = True
        for j, search_word in enumerate(search_words):
            if search_word.lower() != text_words[i + j].lower():
                match = False
                break
        if match:
            return True
    
    return False

def extract_text_to_the_right_of_phrase(page, phrase, search_direction='right', item_order=0, data_type='text', separator='space'):
    """Extract text either to the right or below the search phrase."""
    text = page.extract_text()
    if text:
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if text_contains(line, phrase):
                if search_direction == 'right':
                    # Split the line at the phrase and take the right part
                    # Try exact phrase first
                    parts = line.split(phrase)
                    if len(parts) == 1:  # If exact phrase not found, try case-insensitive
                        parts = line.split(normalize_text(phrase))
                    if len(parts) == 1:  # If still not found, try with flexible spacing
                        search_words = normalize_text(phrase).split()
                        for j in range(len(search_words)):
                            if all(word.lower() in line.lower() for word in search_words[j:]):
                                # Find the position after the last matching word
                                last_word_pos = line.lower().find(search_words[-1].lower()) + len(search_words[-1])
                                parts = [line[:last_word_pos], line[last_word_pos:]]
                                break
                    
                    if len(parts) > 1:
                        right_text = parts[1].strip()
                        numbers = right_text.split()
                        return ' '.join(numbers)
                elif search_direction == 'below':
                    # Skip the first line if it's empty
                    current_idx = i + 1
                    if current_idx < len(lines) and not lines[current_idx].strip():
                        current_idx += 1
                    
                    # Collect all non-empty lines that follow until we hit an empty line
                    following_lines = []
                    while current_idx < len(lines):
                        next_line = lines[current_idx].strip()
                        # Stop if we hit an empty line (after the first line)
                        if not next_line:
                            break
                        following_lines.append(next_line)
                        current_idx += 1
                    
                    # print(f"DEBUG: Found following lines: {following_lines}")
                    
                    if not following_lines:
                        return "0" if data_type == 'numeric' else None
                    
                    if separator == 'line':
                        # For line separator, each line is treated as a separate value
                        if item_order == -1:  # Get the last non-empty line
                            return following_lines[-1]
                        elif item_order < len(following_lines):  # Get the specified line
                            return following_lines[item_order]
                        else:
                            return "0" if data_type == 'numeric' else None
                    else:  # separator == 'space'
                        # For space separator, split each line by spaces and combine all numbers
                        all_numbers = []
                        for line in following_lines:
                            numbers = line.split()
                            all_numbers.extend(numbers)
                        
                        if item_order == -1:  # Get the last number
                            return all_numbers[-1] if all_numbers else ("0" if data_type == 'numeric' else None)
                        elif item_order < len(all_numbers):  # Get the specified number
                            return all_numbers[item_order]
                        else:
                            return "0" if data_type == 'numeric' else None
    
    return "0" if data_type == 'numeric' else None

def extract_by_text(pdf, page_search_text, row_search_text, separator, item_order, search_direction='right', page_number=None, min_page=None, page_cache=None, data_type='text'):
    """Extract a value from a PDF by searching for text and getting a value from the same row or below."""
    try:
        section_page = -1
        
        # Check if we have already found this text in the cache
        if page_cache is not None and page_search_text in page_cache:
            section_page = page_cache[page_search_text]
        else:
            start_page = int(min_page) if pd.notna(min_page) else 0
            # Search for the page if not cached
            for i in range(start_page, len(pdf.pages)):
                text = pdf.pages[i].extract_text()
                if text_contains(text, page_search_text):
                    section_page = i
                    if page_cache is not None:
                        page_cache[page_search_text] = section_page
                        print(f"DEBUG: Caching page {section_page} for text: {page_search_text}")
                    break
    
        if section_page == -1:
            print(f'"{page_search_text}" not found in the PDF (searched from page {start_page}).')
            return np.nan
        
        page = pdf.pages[section_page]
        text = page.extract_text()
        
        # For exports section, check if there are no exports
        if text_contains(page_search_text, "NUTRIENT EXPORTS"):
            if "No solid nutrient exports entered" in text and "No liquid nutrient exports entered" in text:
                return 0
            
        # Look for the row text
        if text_contains(text, row_search_text):
            extracted_text = extract_text_to_the_right_of_phrase(page, row_search_text, search_direction, item_order, data_type, separator)
            if extracted_text:
                if pd.isna(item_order) or item_order == -1:
                    return extracted_text.strip()
                else:
                    values = convert_to_float_list(extracted_text)
                    if len(values) > item_order:
                        return values[item_order]
        return np.nan if data_type == 'text' else 0
    except Exception as e:
        print(f"Error extracting text value: {e}")
        return np.nan if data_type == 'text' else 0

def extract_by_number(pdf_path, x0, y0, x1, y1, page_number):
    """Extract a value from a PDF by coordinates."""
    try:
        with suppress_stderr(), pdfplumber.open(pdf_path) as pdf:
            if page_number >= len(pdf.pages):
                print(f"Page number {page_number} out of range")
                return np.nan
                
            page = pdf.pages[page_number]
            page_width = float(page.width)
            page_height = float(page.height)
            
            # If coordinates NA, use full page
            if pd.isna(x0) or pd.isna(y0) or pd.isna(x1) or pd.isna(y1):
                x0 = 0
                y0 = 0
                x1 = page_width
                y1 = page_height
            
            x0 = max(0, min(x0, page_width))
            x1 = max(0, min(x1, page_width))
            y0 = max(0, min(y0, page_height))
            y1 = max(0, min(y1, page_height))
            
            text = page.within_bbox((x0, y0, x1, y1)).extract_text()
            
            if not text or not is_convertible_to_float(text):
                return np.nan
            else:
                text = text.replace(",", "")
                return float(text)
    except Exception as e:
        print(f"Error extracting value by coordinates: {e}")
        return np.nan

def extract_parameter(pdf_path, row, page_number, page_cache=None):
    """Extract a parameter value from a PDF based on the row configuration."""
    try:
        with suppress_stderr(), pdfplumber.open(pdf_path) as pdf:
            if row['find_by'] == 'exact_location':
                print('extracting by exact location')
                return extract_by_number(pdf_path, row['x0'], row['y0'], row['x1'], row['y1'], row['page_number'])
            else:
                print('extracting by text search')
                search_direction = row['search_direction'] if pd.notna(row['search_direction']) else 'right'
                return extract_by_text(pdf, row['page_search_text'], row['row_search_text'], 
                                     row['separator'], row['item_order'], search_direction,
                                     row['page_number'], row['min_page'], page_cache, row['data_type'])
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return np.nan

def extract_nutrient_total_exports(pdf_path, dictionary):                    
    with suppress_stderr(), pdfplumber.open(pdf_path) as pdf:
        page_number = -1
        # Find relevant page
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if "Total exports for all materials" in text:
                page_number = i
                break
        if page_number != -1:
            # Extract the text to the right of where the string was found
            text = extract_text_to_the_right_of_phrase(pdf.pages[page_number], "Total exports for all materials")
            # Convert to list of floats
            l = convert_to_float_list(text)
            # Take first element
            N_exports = l[0]
            P_exports = l[1]
            K_exports = l[2]
            Salt_exports = l[3]
            # Assign the values to the dictionary
            dictionary["Total Exports N (lbs)"] = N_exports
            dictionary["Total Exports P (lbs)"] = P_exports
            dictionary["Total Exports K (lbs)"] = K_exports
            dictionary["Total Exports Salt (lbs)"] = Salt_exports
        else:
            dictionary["Total Exports N (lbs)"] = 0
            dictionary["Total Exports P (lbs)"] = 0
            dictionary["Total Exports K (lbs)"] = 0
            dictionary["Total Exports Salt (lbs)"] = 0
            print('"Total exports for all materials" not found in the entire PDF.')

def convert_to_float_list(text):
    if not text:
        return []
        
    # Split text by whitespace and remove empty strings
    components = [c for c in text.split() if c]
    
    float_numbers = []
    for component in components:
        # Remove any non-numeric characters except decimal points and negative signs
        # This will handle cases where there are units or other text mixed with numbers
        cleaned = ''.join(c for c in component if c.isdigit() or c in '.-')
        
        # Skip if we don't have any digits left
        if not any(c.isdigit() for c in cleaned):
            continue
            
        try:
            # Convert the cleaned component to a float and append to the list
            float_numbers.append(float(cleaned))
        except ValueError:
            # Skip invalid numbers but continue processing
            print(f"Could not convert '{component}' (cleaned to '{cleaned}') to float")
            continue
    
    return float_numbers

def get_pdf_page_count(pdf_path):
    """
    Get the total number of pages in a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Number of pages in the PDF
    """
    with suppress_stderr(), pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)
    
def main(test_mode=False):
    # Base paths
    years = [2023, 2024]
    regions = ['R2', 'R3', 'R5', 'R7', 'R8']
    
    # Read parameter locations from CSV
    parameter_locations = pd.read_csv('ca_cafo_compliance/parameter_locations.csv')
    parameter_locations['page_number'] = parameter_locations['page_number'].fillna(-1).astype(int)
    parameter_locations['item_order'] = parameter_locations['item_order'].fillna(-1).astype(int)
    
    # Get all available templates from parameter_locations
    available_templates = parameter_locations['template'].unique()
    
    # Process each year
    for year in years:
        base_data_path = f"data/{year}"
        base_output_path = f"outputs/{year}"
        
        # Skip if year folder doesn't exist
        if not os.path.exists(base_data_path):
            print(f"Year folder not found: {base_data_path}")
            continue
            
        # Process each region
        for region in regions:
            region_data_path = os.path.join(base_data_path, region)
            region_output_path = os.path.join(base_output_path, region)
            
            # Skip if region folder doesn't exist
            if not os.path.exists(region_data_path):
                print(f"Region folder not found: {region_data_path}")
                continue
                
            print(f"\nDEBUG: Processing year: {year}, region: {region}")
            
            # Get all counties in this region
            counties = [d for d in os.listdir(region_data_path) 
                       if os.path.isdir(os.path.join(region_data_path, d))]
            
            # Process each county
            for county in counties:
                county_data_path = os.path.join(region_data_path, county)
                county_output_path = os.path.join(region_output_path, county)
                
                print(f"\nDEBUG: Processing county: {county}")
                
                # Get all template folders in this county
                template_folders = [d for d in os.listdir(county_data_path) 
                                   if os.path.isdir(os.path.join(county_data_path, d))]
                
                # Process each template folder
                for template in template_folders:
                    # Skip if template is not in parameter_locations
                    if template not in available_templates:
                        print(f"Skipping template '{template}' - not found in parameter_locations")
                        continue
                        
                    print(f"\nDEBUG: Processing {template} template in {county}")
                    
                    folder = os.path.join(county_data_path, template)
                    output_folder = os.path.join(county_output_path, template)
                    name = f"{county.capitalize()}_{year}_{template}"
                    
                    # Filter for template
                    template_params = parameter_locations[parameter_locations['template'] == template]
                    
                    pdf_files = glob.glob(os.path.join(folder, '*.pdf'))
                    
                    if test_mode:
                        print(f"Running in test mode - processing only 2 files for {template}")
                        pdf_files = pdf_files[:2]

                    if not pdf_files:
                        print(f"No PDF files found in {folder}")
                        continue

                    # Initialize DataFrame with unique column names
                    columns = parameter_locations['parameter_name'].unique().tolist()
                    df = pd.DataFrame(columns=columns)
                    
                    for pdf_path in pdf_files:
                        dairy_name = os.path.basename(pdf_path)
                        print(f"\nDEBUG: Processing dairy: {os.path.splitext(dairy_name)[0]}")
                        
                        # Create dictionary to store all results for each dairy
                        dairy_dict = dict()
                        
                        # Create page cache dictionary for this PDF
                        page_cache = {}

                        # Open PDF once for all parameters
                        with suppress_stderr(), pdfplumber.open(pdf_path) as pdf:
                            # Process each parameter
                            for _, row in template_params.iterrows():
                                if row['find_by'] == 'exact_location':
                                    value = extract_by_number(pdf_path, row['x0'], row['y0'], row['x1'], row['y1'], row['page_number'])
                                else:
                                    search_direction = row['search_direction'] if pd.notna(row['search_direction']) else 'right'
                                    value = extract_by_text(pdf, row['page_search_text'], row['row_search_text'], 
                                                         row['separator'], row['item_order'], search_direction,
                                                         row['page_number'], row['min_page'], page_cache, row['data_type'])
                                
                                if pd.isna(value):
                                    print(f"Warning: No value found for parameter '{row['parameter_name']}'")
                                dairy_dict[row['parameter_name']] = value
                                
                                # if value is not None:
                                #     print(row['parameter_name'] + ": " + str(value))

                        # Convert the dictionary to a DataFrame
                        new_row_df = pd.DataFrame([dairy_dict])

                        # Missing column info
                        missing_cols = set(columns) - set(new_row_df.columns)
                        if missing_cols:
                            print("\nMissing columns in new row:")
                            for col in missing_cols:
                                print(f"  {col}")

                        # Add new row to df
                        df = pd.concat([df, new_row_df], ignore_index=True)

                    # Convert numeric fields to numeric based on data_type in parameter_locations
                    numeric_columns = template_params[template_params['data_type'] == 'numeric']['parameter_name'].tolist()
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            # Print any rows where column is NA
                            na_rows = df[df[col].isna()]
                            if not na_rows.empty:
                                print(f"\nRows with NA values in {col}:")
                                for idx, row in na_rows.iterrows():
                                    print(f"  Row {idx}: Original value = {row[col]}")

                    calculate_all_metrics(df)

                    os.makedirs(output_folder, exist_ok=True)
                    df.to_csv(os.path.join(output_folder, f"{name}.csv"), index=False)

def calculate_all_metrics(df):
    """Calculate all possible metrics, filling with NA where not applicable"""
    
    # General Order metrics
    try:
        df["Total Herd Size"] = df["Average Milk Cows"].fillna(0) + df["Average Dry Cows"].fillna(0) + \
                               df["Average Bred Heifers"].fillna(0) + df["Average Heifers"].fillna(0) + \
                               df["Average Calves (4-6 mo.)"].fillna(0) + df["Average Calves (0-3 mo.)"].fillna(0)
    except:
        df["Total Herd Size"] = np.nan

    # Innovative Ag metrics
    try:
        df["Total Head"] = df["Average No. of Head"]
        df["Total AUs"] = df["Average No. of AUs"]
    except:
        df["Total Head"] = np.nan
        df["Total AUs"] = np.nan

    # Common nutrient calculations for both templates
    nutrient_types = ["N", "P", "K", "Salt"]
    for nutrient in nutrient_types:
        # Total Applied
        try:
            df[f"Total Applied {nutrient} (lbs)"] = df[f"Applied {nutrient} Dry Manure (lbs)"].fillna(0) + \
                                                   df[f"Applied Process Wastewater {nutrient} (lbs)"].fillna(0)
        except:
            df[f"Total Applied {nutrient} (lbs)"] = np.nan

        # Total Reported
        try:
            if nutrient == "N":
                df[f"Total Reported {nutrient} (lbs)"] = df[f"Total Dry Manure Generated {nutrient} After Ammonia Losses (lbs)"].fillna(0) + \
                                                        df[f"Total Process Wastewater Generated {nutrient} (lbs)"].fillna(0)
            else:
                df[f"Total Reported {nutrient} (lbs)"] = df[f"Total Dry Manure Generated {nutrient} (lbs)"].fillna(0) + \
                                                        df[f"Total Process Wastewater Generated {nutrient} (lbs)"].fillna(0)
        except:
            df[f"Total Reported {nutrient} (lbs)"] = np.nan

        # Unaccounted for
        try:
            if nutrient == "N":
                df[f"Unaccounted-for {nutrient} (lbs)"] = df[f"Total Dry Manure Generated {nutrient} After Ammonia Losses (lbs)"].fillna(0) + \
                                                         df[f"Total Process Wastewater Generated {nutrient} (lbs)"].fillna(0) - \
                                                         df[f"Total Applied {nutrient} (lbs)"].fillna(0) - \
                                                         df[f"Total Exports {nutrient} (lbs)"].fillna(0)
            else:
                df[f"Unaccounted-for {nutrient} (lbs)"] = df[f"Total Dry Manure Generated {nutrient} (lbs)"].fillna(0) + \
                                                         df[f"Total Process Wastewater Generated {nutrient} (lbs)"].fillna(0) - \
                                                         df[f"Total Applied {nutrient} (lbs)"].fillna(0) - \
                                                         df[f"Total Exports {nutrient} (lbs)"].fillna(0)
        except:
            df[f"Unaccounted-for {nutrient} (lbs)"] = np.nan

    # Wastewater calculations
    try:
        df["Total Process Wastewater Generated (L)"] = df["Total Process Wastewater Generated (gals)"].fillna(0) * 3.78541
    except:
        df["Total Process Wastewater Generated (L)"] = np.nan

    try:
        df["Ratio of Wastewater to Milk (L/L)"] = df["Total Process Wastewater Generated (L)"] / df["Total Annual Milk Production (L)"]
    except:
        df["Ratio of Wastewater to Milk (L/L)"] = np.nan

if __name__ == "__main__":
    # Set test_mode=True to process only 2 files
    main(test_mode=False)

# import os
# import re
# import pandas as pd
# import glob

# def extract_dairy_info(file_path):
#     """
#     Extract dairy cow and other animal information from CAFO compliance reports.
    
#     Args:
#         file_path: Path to the text file to process
        
#     Returns:
#         Dictionary containing extracted information
#     """
#     with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
#         content = file.read()
    
#     # Extract facility name
#     facility_name_match = re.search(r'Facility Name:\s*([^\n]+)', content)
#     facility_name = facility_name_match.group(1).strip() if facility_name_match else "Unknown"
    
#     # Extract mature dairy cows count
#     mature_cows_match = re.search(r'Current # of mature dairy cows \(milking \+ dry\):\s*(\d+)', content)
#     mature_cows = int(mature_cows_match.group(1)) if mature_cows_match else None
    
#     # Extract other animals information
#     other_animals_match = re.search(r'Current # and type of other animals:\s*([^\n]+)', content)
    
#     other_animals_count = None
#     other_animals_type = None
#     bred_heifers = None
#     heifers = None
#     calves = None
#     unspecified = None
    
#     if other_animals_match:
#         other_animals_text = other_animals_match.group(1).strip()
#         # Try to extract number and type
#         number_type_match = re.search(r'(\d+)\s+(.+)', other_animals_text)
#         if number_type_match:
#             other_animals_count = int(number_type_match.group(1))
#             other_animals_type = number_type_match.group(2).strip()
            
#             # Categorize animals based on type
#             animal_type_lower = other_animals_type.lower()
#             if 'bred heifer' in animal_type_lower:
#                 bred_heifers = other_animals_count
#             elif 'heifer' in animal_type_lower:
#                 heifers = other_animals_count
#             elif 'calv' in animal_type_lower or 'calf' in animal_type_lower or 'young' in animal_type_lower:
#                 calves = other_animals_count
#             else:
#                 unspecified = other_animals_count
#         else:
#             other_animals_type = other_animals_text
#             # Try to parse animal counts from the description
#             if other_animals_text.isdigit():
#                 unspecified = int(other_animals_text)
    
#     # Get filename without extension for reference
#     file_name = os.path.basename(file_path)
    
#     return {
#         'file_name': file_name,
#         'facility_name': facility_name,
#         'mature_dairy_cows': mature_cows,
#         'other_animals_count': other_animals_count,
#         'other_animals_type': other_animals_type,
#         'bred_heifers': bred_heifers,
#         'heifers': heifers,
#         'calves': calves,
#         'unspecified_animals': unspecified
#     }

# def process_all_reports(directory_path):
#     """
#     Process all text files in the specified directory and create a DataFrame.
    
#     Args:
#         directory_path: Path to directory containing text files
        
#     Returns:
#         DataFrame with extracted information
#     """
#     all_data = []
    
#     # Get all text files in the directory
#     file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
    
#     for file_path in file_paths:
#         try:
#             data = extract_dairy_info(file_path)
#             all_data.append(data)
#         except Exception as e:
#             print(f"Error processing {file_path}: {e}")
    
#     # Create DataFrame
#     df = pd.DataFrame(all_data)
#     return df

# def main():
#     directory_path = "../data/R2_txt"
    
#     # Check if directory exists
#     if not os.path.exists(directory_path):
#         print(f"Directory {directory_path} does not exist.")
#         return
    
#     # Process all reports
#     df = process_all_reports(directory_path)
    
#     # Save to CSV
#     output_file = "../outputs/cafo_report_data.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Data extracted and saved to {output_file}")
    
#     # Display summary
#     print(f"\nProcessed {len(df)} reports")
#     print(f"Reports with mature dairy cows data: {df['mature_dairy_cows'].notna().sum()}")
#     print(f"Reports with other animals data: {df['other_animals_count'].notna().sum()}")
#     print(f"Reports with bred heifers: {df['bred_heifers'].notna().sum()}")
#     print(f"Reports with heifers: {df['heifers'].notna().sum()}")
#     print(f"Reports with calves: {df['calves'].notna().sum()}")
#     print(f"Reports with unspecified animals: {df['unspecified_animals'].notna().sum()}")

# if __name__ == "__main__":
#     main()
