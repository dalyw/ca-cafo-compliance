import os
import pandas as pd
from pathlib import Path

def find_pdf_files():
    # Base directory
    base_dir = Path('data')
    
    # List to store file information
    files_info = []
    
    # Walk through the directory structure
    for year_dir in base_dir.glob('20*'):  # Match year directories (2023, 2024)
        if not year_dir.is_dir():
            print('year dir not found')
            continue
            
        year = year_dir.name
        
        # Look for region directories
        for region_dir in year_dir.glob('R*'):
            if not region_dir.is_dir():
                continue
                
            region = region_dir.name
            print(region)
            # Look for county directories
            for county_dir in region_dir.glob('*'):
                if not county_dir.is_dir():
                    continue
                    
                county = county_dir.name
                
                # Look for template directories
                for template_dir in county_dir.glob('*'):
                    if not template_dir.is_dir():
                        continue
                        
                    template = template_dir.name
                    
                    # Find PDF files
                    for pdf_file in template_dir.glob('original/*.pdf'):
                        print(pdf_file)
                        files_info.append({
                            'region': region,
                            'template': template,
                            'year': year,
                            'county': county,
                            'filename': pdf_file.name
                        })
    
    return files_info

def main():
    # Find all PDF files
    files_info = find_pdf_files()
    
    # Create DataFrame
    df = pd.DataFrame(files_info)
    print(df)
    # Sort by year, region, county, and filename
    df = df.sort_values(['year', 'region', 'county', 'filename'])
    
    # Save to CSV
    output_file = 'outputs/file_list.csv'
    df.to_csv(output_file, index=False)
    print(f"Updated file list saved to {output_file}")
    print(f"Found {len(df)} PDF files")

if __name__ == '__main__':
    main() 