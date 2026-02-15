#!/bin/bash
# for saving JUST the manifest .txt files and .pdfs into separate folder

BASE_DIR="/Users/dalywettermark/Library/CloudStorage/GoogleDrive-dalyw@stanford.edu/My Drive/ca_cafo_manifests"
OUTPUT_BASE="$BASE_DIR/all_manifests"

# Find all template directories
find "$BASE_DIR" -type d -name "original" | while read -r original_dir; do
    # Extract path components
    template_dir=$(dirname "$original_dir")
    county_dir=$(dirname "$template_dir")
    region_dir=$(dirname "$county_dir")
    year_dir=$(dirname "$region_dir")
    
    county=$(basename "$county_dir")
    region=$(basename "$region_dir")
    year=$(basename "$year_dir")
    template=$(basename "$template_dir")
    
    # Skip if year is not a number (avoid processing all_manifests itself)
    [[ ! "$year" =~ ^[0-9]+$ ]] && continue
    
    # Create output directory (no template subfolder)
    output_dir="$OUTPUT_BASE/$year/$region/$county"
    mkdir -p "$output_dir"
    
    echo "Processing: $year/$region/$county/$template"
    
    # Process each PDF in original folder (case-insensitive)
    find "$original_dir" -type f \( -name "*.pdf" -o -name "*.PDF" \) | while read -r pdf_file; do
        # Strip extension case-insensitively
        pdf_name=$(basename "$pdf_file" | sed 's/\.[pP][dD][fF]$//')
        
        # Skip if this PDF subfolder already exists (already copied from higher priority)
        if [ -d "$output_dir/$pdf_name" ]; then
            echo "  $pdf_name -> SKIPPED (already exists)"
            continue
        fi
        
        # Check priority order for output folders
        source_folder=""
        
        if [ -d "$template_dir/llmwhisperer_output/$pdf_name" ]; then
            source_folder="$template_dir/llmwhisperer_output/$pdf_name"
            output_type="llmwhisperer"
        elif [ -d "$template_dir/tesseract_output/$pdf_name" ]; then
            source_folder="$template_dir/tesseract_output/$pdf_name"
            output_type="tesseract"
        else
            echo "  $pdf_name -> SKIPPED (no output)"
            continue
        fi
        
        # Copy only files with "manifest" in the name
        if [ -n "$source_folder" ]; then
            # Create the PDF subfolder
            mkdir -p "$output_dir/$pdf_name"
            
            # Copy only manifest files
            manifest_count=0
            find "$source_folder" -type f -iname "*manifest*" | while read -r manifest_file; do
                cp "$manifest_file" "$output_dir/$pdf_name/"
                ((manifest_count++))
            done
            
            # Check if any files were copied
            if [ "$(find "$output_dir/$pdf_name" -type f | wc -l)" -gt 0 ]; then
                echo "  $pdf_name -> $output_type (manifest files)"
            else
                echo "  $pdf_name -> SKIPPED (no manifest files)"
                rmdir "$output_dir/$pdf_name" 2>/dev/null
            fi
        else
            echo "  $pdf_name -> SKIPPED (no output)"
        fi
    done
done

echo "Done!"