#!/usr/bin/env python3

import pandas as pd
from pypdf import PdfReader
import pdfplumber
import sys
import glob
import numpy as np
import os
import re

def extract_text_from_rectangle(pdf_path, page_number, rect):
    with pdfplumber.open(pdf_path) as pdf:
        # Get the specified page
        page = pdf.pages[page_number]
        
        # Define the rectangle (x0, y0, x1, y1)
        x0, y0, x1, y1 = rect
        
        # Extract text from the specified rectangular region
        text = page.within_bbox((x0, y0, x1, y1)).extract_text()
        
        return text

def get_pdf_page_dimensions(pdf_path, page_number=0):
    # Open the PDF file
    reader = PdfReader(pdf_path)

    # Access the specified page
    page = reader.pages[page_number]

    # Get the MediaBox
    media_box = page.get('/MediaBox')
    
    if media_box is not None:
        # Convert the MediaBox values to float
        width = float(media_box[2]) - float(media_box[0])  # x1 - x0
        height = float(media_box[3]) - float(media_box[1])  # y1 - y0

        return width, height
    else:
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

def extract_dairy_info(pdf_path, page_number, rectangle, name, dictionary):
    # Extract text from rectangle
    text = extract_text_from_rectangle(pdf_path, page_number, rectangle)

    # print(name)
    # If looking at cow types, different method
    if name == "Herd Info":
        # Split the string by spaces to get a list of string numbers
        text = text.replace(",", "")
        number_list = text.split()
        # Convert the list of string numbers to a list of integers (or floats if needed)
        number_list = [int(num) for num in number_list]
        # Assign the values to the respective variables
        cow_types = ["Milk Cows", "Dry Cows", "Bred Heifers", "Heifers", "Calves (4-6 mo.)", "Calves (0-3 mo.)"]
        if len(number_list) != len(cow_types):
            print("Error: Number of cow counts does not match number of cow types.")
            raise ValueError()
            return
        else:
            for i in range(len(cow_types)):
                dictionary[cow_types[i]] = number_list[i]
            return
        return
    
    # If getting float value, different method
    if name in ["Total Manure Excreted (tons)","Total Process Wastewater Generated (gals)","Total Dry Manure Generated N (lbs)","Total Dry Manure Generated N After Ammonia Losses (lbs)","Average Milk Production (lb per cow per day)",
                "Total Dry Manure Generated P (lbs)", "Total Dry Manure Generated K (lbs)", "Total Dry Manure Generated Salt (lbs)",
                "Total Process Wastewater Generated (gals)", "Total Process Wastewater Generated N (lbs)", "Total Process Wastewater Generated P (lbs)", "Total Process Wastewater Generated K (lbs)", "Total Process Wastewater Generated Salt (lbs)"]:
        # If text is empty, assign 0
        if is_convertible_to_float(text) == False:
            dictionary[name] = 0
            return
        # remove any commas and convert to float
        else:
            text = text.replace(",", "")
            text = float(text)
            dictionary[name] = text
            return

    # For text info just add the text to the dictionary      
    else:
        # Assign the value to the dictionary
        dictionary[name] = text
        return

def extract_nutrient_application_info(pdf_path, dictionary):
    data_order =  ["Applied N Dry Manure (lbs)", "Applied P Dry Manure (lbs)", "Applied K Dry Manure (lbs)", "Applied Salt Dry Manure (lbs)",
                   "Applied Process Wastewater N (lbs)", "Applied Process Wastewater P (lbs)", "Applied Process Wastewater K (lbs)", "Applied Process Wastewater Salt (lbs)",
                   "Applied to Remove Ratio N", "Applied to Remove Ratio P", "Applied to Remove Ratio K", "Applied to Remove Ratio Salt"]
                    #"Total Dry Manure Generated N (lbs)"]
                    
    with pdfplumber.open(pdf_path) as pdf:
        page_number = -1
        # Find relevant page
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if "SUMMARY OF NUTRIENT APPLICATIONS" in text:
                page_number = i
                break
        if page_number != -1:
            #print(f'"SUMMARY OF NUTRIENT APPLICATIONS" found on page {page_number + 1}')
            # Define the coordinates of the nutrient balance and applied to removed ratio data
            # Coordinates are in the format (x0, top, x1, bottom)
            x0, top, x1, bottom = 150, 175, 600, 210
            
            # Extract the page where the string was found
            page = pdf.pages[page_number]
            
            # Extract text within the defined rectangle
            cropped_text = page.within_bbox((x0, top, x1, bottom)).extract_text()

            # Convert the extracted text to a list of floats
            l1 = convert_to_float_list(cropped_text)

            # Next, extract the applied to removed ratio data
            x0, top, x1, bottom = 150, 290, 600, 320
            cropped_text = page.within_bbox((x0, top, x1, bottom)).extract_text()
            l2 = convert_to_float_list(cropped_text)

            # Merge lists
            l = l1 + l2

            # Next, extract the Dry Manure Total N (lbs)
            #x0, top, x1, bottom = 180, 170, 300, 190
            #cropped_text = page.within_bbox((x0, top, x1, bottom)).extract_text()
            #l2 = convert_to_float_list(cropped_text)
            #l = l + l2

            # Next assign the values to the dictionary
            if len(l) != len(data_order):
                print("Error: Number of nutrient values does not match number of nutrient types.")
                raise ValueError()
                return
            for i in range(len(data_order)):
                dictionary[data_order[i]] = l[i]
        else:
            print('"SUMMARY OF NUTRIENT APPLICATIONS" not found in the entire PDF.')

def extract_text_to_the_right_of_phrase(page, phrase):
    text = page.extract_text()
    if text:
        lines = text.split('\n')
        for line in lines:
            if phrase in line:
                # This assumes that text to the right is separated by whitespace
                # Adjust the splitting logic if needed based on your document's structure
                parts = line.split(phrase)
                if len(parts) > 1:
                    return parts[1].strip()
    return None

def extract_parameter_by_text(pdf_path, search_text, separator, item_order):
    with pdfplumber.open(pdf_path) as pdf:
        page_number = -1
        # Find relevant page
        for page in pdf.pages:
            text = page.extract_text()
            if search_text in text:
                # Extract the text to the right of where the string was found
                right_text = extract_text_to_the_right_of_phrase(page, search_text)
                if right_text:
                    # Convert to list of floats
                    values = convert_to_float_list(right_text)
                    if len(values) > item_order:
                        return values[item_order]
        return 0  # Default value if not found

def extract_parameter(pdf_path, row, dairy_dict):
    print('Extracting ' + row['parameter_name'])
    if row['find_by'] == 'exact_location':
        extract_dairy_info(pdf_path, row['page_number'], 
                         (row['x0'], row['y0'], row['x1'], row['y1']), 
                         row['parameter_name'], dairy_dict)
    elif row['find_by'] == 'text':
        value = extract_parameter_by_text(pdf_path, row['search_text'], row['separator'], int(row['item_order']))
        dairy_dict[row['parameter_name']] = value

def extract_nutrient_total_exports(pdf_path, dictionary):                    
    with pdfplumber.open(pdf_path) as pdf:
        page_number = -1
        # Find relevant page
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if "Total exports for all materials" in text:
                page_number = i
                break
        if page_number != -1:
            # Extract the text to the right of where the string was found
            #print(f'"Total exports for all materials" found on page {page_number + 1}')
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
    # Remove any unwanted characters and split the text by whitespace
    components = re.split(r'\s+', text.strip())
    
    float_numbers = []
    for component in components:
        # Remove commas used as thousands separators
        cleaned_component = component.replace(',', '')

        try:
            # Convert the cleaned component to a float and append to the list
            float_numbers.append(float(cleaned_component))
        except ValueError:
            # Handle the case where the conversion fails (if any)
            print(f"Could not convert '{component}' to float")
    
    return float_numbers

def main():
    # Set folder and output name
    # folder = "/Users/ianbick/Library/CloudStorage/OneDrive-Stanford/CAFO/CAFO_Water_Reports/Region 5/Tulare West Dairy/Correct_Forms"
    # output_folder = "/Users/ianbick/Library/CloudStorage/OneDrive-Stanford/CAFO/CAFO_Water_Reports/Region 5/Tulare West Dairy/Results"
    folder = "data/2023/R5/Tulare West Dairy/General_Order_Template_Test/"
    output_folder = "outputs/2023/R5/Tulare West Dairy/General_Order_Template_Test/"
    name = "TulareWest_2023_R5-2007-0035"

    # Read parameter locations from CSV
    parameter_locations = pd.read_csv('ca_cafo_compliance/parameter_locations.csv')
    # print(parameter_locations.columns)
    # Use glob to find all PDF files in the directory
    pdf_files = glob.glob(os.path.join(folder, '*.pdf'))

    # Create a pandas dataFrame to store the results
    df = pd.DataFrame(columns=parameter_locations['parameter_name'].tolist())
    # Print out the list of PDF files
    for pdf_path in pdf_files:
        # Create dictionary to store all results for each dairy
        dairy_dict = dict()
        
        # Try page 1 first for exact_location parameters
        try:
            for _, row in parameter_locations.iterrows():
                extract_parameter(pdf_path, row, dairy_dict)
        except ValueError:
            # If page 1 fails, try page 2 for exact_location parameters
            for _, row in parameter_locations.iterrows():
                # print('valueerror')
                if row['find_by'] == 'exact_location' and row['page_number'] == 1:
                    row['page_number'] = 2  # Update page number for retry
                extract_parameter(pdf_path, row, dairy_dict)

        print('****************')

        # Convert the dictionary to a DataFrame
        new_row_df = pd.DataFrame([dairy_dict])

        # Concatenate the new DataFrame with the original DataFrame
        df = pd.concat([df, new_row_df], ignore_index=True)

        # Convert numeric fields to numeric based on data_type in parameter_locations
        numeric_columns = parameter_locations[parameter_locations['data_type'] == 'numeric']['parameter_name'].tolist()
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    ##########################################################
    # Calculate Total Size of Herd
    ##########################################################
    # print(df.columns)
    df["Total Herd Size"] = df["Milk Cows"] + df["Dry Cows"] + df["Bred Heifers"] + df["Heifers"] + df["Calves (4-6 mo.)"] + df["Calves (0-3 mo.)"]

    ##########################################################
    # Sum Nutrients from Dry Manure and Process Wastewater
    ##########################################################
    df["Total Applied N (lbs)"] = df["Applied N Dry Manure (lbs)"] + df["Applied Process Wastewater N (lbs)"]
    df["Total Applied P (lbs)"] = df["Applied P Dry Manure (lbs)"] + df["Applied Process Wastewater P (lbs)"]
    df["Total Applied K (lbs)"] = df["Applied K Dry Manure (lbs)"] + df["Applied Process Wastewater K (lbs)"]
    df["Total Applied Salt (lbs)"] = df["Applied Salt Dry Manure (lbs)"] + df["Applied Process Wastewater Salt (lbs)"]

    ##########################################################
    # Sum Nutrients from Dry Manure and Process Wastewater
    ##########################################################
    df["Unaccounted-for N (lbs)"] = df["Total Dry Manure Generated N After Ammonia Losses (lbs)"] + df["Total Process Wastewater Generated N (lbs)"]  - df["Total Applied N (lbs)"] - df["Total Exports N (lbs)"]
    df["Unaccounted-for P (lbs)"] = df["Total Dry Manure Generated P (lbs)"] + df["Total Process Wastewater Generated P (lbs)"] - df["Total Applied P (lbs)"] - df["Total Exports P (lbs)"]
    df["Unaccounted-for K (lbs)"] = df["Total Dry Manure Generated K (lbs)"] + df["Total Process Wastewater Generated K (lbs)"] - df["Total Applied K (lbs)"] - df["Total Exports K (lbs)"]
    df["Unaccounted-for Salt (lbs)"] = df["Total Dry Manure Generated Salt (lbs)"] + df["Total Process Wastewater Generated Salt (lbs)"] - df["Total Applied Salt (lbs)"] - df["Total Exports Salt (lbs)"]

    ##########################################################
    ## Calculate total Reported Nutrients
    ##########################################################
    df["Total Reported N (lbs)"] = df["Total Dry Manure Generated N After Ammonia Losses (lbs)"] + df["Total Process Wastewater Generated N (lbs)"]
    df["Total Reported P (lbs)"] = df["Total Dry Manure Generated P (lbs)"] + df["Total Process Wastewater Generated P (lbs)"]
    df["Total Reported K (lbs)"] = df["Total Dry Manure Generated K (lbs)"] + df["Total Process Wastewater Generated K (lbs)"]
    df["Total Reported Salt (lbs)"] = df["Total Dry Manure Generated Salt (lbs)"] + df["Total Process Wastewater Generated Salt (lbs)"]

    # Export csv
    df.to_csv(output_folder + "/" + name + "_N_P_K_Salt_Balance_Ratios" + ".csv", index=False)

    ##########################################################
    ### Calculate Milk Production
    ##########################################################
    # Milk pounds to liters 
    # (https://books.lib.uoguelph.ca/dairyscienceandtechnologyebook/chapter/physical-properties-of-milk/#:~:text=With%20all%20of%20this%20in,m3%20at%2020°%20C.)

    lb_to_kg = 0.453592 # kg/lb
    kg_to_L = 0.971 # L/kg at 20 degrees C

    def calculateAnnualMilkProduction(x,lb_to_kg,kg_to_L):
        try:
            # Calculate Average Milk Production (kg per cow)
            x_kg = x['Average Milk Production (lb per cow per day)'] * lb_to_kg
            # Convert to liters per cow
            x_l = x_kg * kg_to_L
            # Convert to total annual milk production in L
            x_l_annual = x_l * (x['Milk Cows']+x['Dry Cows']) * 365
            return x_kg, x_l, x_l_annual
        except Exception as e:
            print(e)
            return np.nan, np.nan, np.nan

    df[['Average Milk Production (kg per cow)', 'Average Milk Production (L per cow)', 'Total Annual Milk Production (L)']] = df.apply(lambda x: calculateAnnualMilkProduction(x,lb_to_kg, kg_to_L),axis=1, result_type='expand')

    ### Calculate liters of wastewater and ratio to milk production
    def calculateWastewater(x):
        # Check if wastewater is generated
        if x['Total Process Wastewater Generated (gals)'] == 0 or pd.isna(x['Total Process Wastewater Generated (gals)']):
            return 0, np.nan
        else:
            # Convert to liters
            x_l = x['Total Process Wastewater Generated (gals)'] * 3.78541
            try:
                # Calculate ratio of wastewater to milk production
                ratio = x_l / x['Total Annual Milk Production (L)']
                return x_l, ratio
            except Exception as e:
                print(e)
                return x_l, np.nan

    df[["Total Process Wastewater Generated (L)", "Ratio of Wastewater to Milk (L/L)"]] = df.apply(lambda x: calculateWastewater(x),axis=1, result_type='expand')

    ### Calculate Nitrogen from Manure (USDA)
    # Convert tons of manure to N
    # 12.92 Pounds of nitrogen/ton wet weight manure 
    # https://www.nrcs.usda.gov/sites/default/files/2022-10/ManRpt_KelMofGol_2007_final.pdf
    def calculateUsdaNitrogenFromManure(x):
        if x["Total Herd Size"] == 0:
            return np.nan, np.nan
        else:
            try:
                x_n_usda = x['Total Manure Excreted (tons)'] * 12.92
                x_ratio = x_n_usda / x["Total Dry Manure Generated N (lbs)"]
                return x_n_usda, x_ratio
            except Exception as e:
                print(e)
                return np.nan, np.nan

    df[['USDA Nitrogen from Manure (lbs)',"Ratio of USDA N to Reported N"]] = df.apply(lambda x: calculateUsdaNitrogenFromManure(x),axis=1, result_type='expand')

    ### Calculate Nitrogen from Manure (UCCE)
    def calculateUcceNitrogenFromManure(x):
        if x["Total Herd Size"] == 0:
            return np.nan, np.nan
        else:
            try:
                x_n_ucce = ((x['Milk Cows'] + x['Dry Cows']) + ((x['Bred Heifers'] + x['Heifers'])*(1.5/4.1)) + ((x['Calves (4-6 mo.)'] + x['Calves (0-3 mo.)'])*(0.5/4.1))) * 365
                x_ratio = x_n_ucce / x["Total Dry Manure Generated N (lbs)"]
                return x_n_ucce, x_ratio
            except Exception as e:
                print(e)
                return np.nan, np.nan

    df[['UCCE Nitrogen from Manure (lbs)',"Ratio of UCCE N to Reported N"]] = df.apply(lambda x: calculateUcceNitrogenFromManure(x),axis=1, result_type='expand')

    ### Calculate Manure Conversion Factor
    def calculateManureConversionFactor(x):
        try:
            x_mcf = x["Total Manure Excreted (tons)"] / ( (x['Milk Cows'] + x['Dry Cows']) + ((x['Bred Heifers'] + x['Heifers'])*(1.5/4.1)) + ((x['Calves (4-6 mo.)'] + x['Calves (0-3 mo.)'])*(1.5/4.1)))
            return x_mcf
        except Exception as e:
            print(e)
            return np.nan
            
    df['Manure Conversion Factor (tons per cow per year)'] = df.apply(lambda x: calculateManureConversionFactor(x),axis=1)

    # Export final csv with all calculations
    df.to_csv(output_folder + "/" + name + ".csv", index=False)

if __name__ == "__main__":
    main()

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
