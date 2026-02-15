"""
Update manifests_manual.csv with values from extracted_manifests.csv.
Rows marked "x" in DONE column are left unchanged.
"""

import pandas as pd
import os

# Paths
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MANUAL_PATH = os.path.join(OUTPUTS_DIR, "manifests_manual.csv")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "extracted_manifests.csv")


def main():
    # Load both files
    manual_df = pd.read_csv(MANUAL_PATH)
    extracted_df = pd.read_csv(EXTRACTED_PATH)

    # Store original column order
    original_columns = manual_df.columns.tolist()

    # Key columns to match rows
    key_cols = ["Source PDF", "Manifest Number"]

    # Find overlapping columns (excluding keys and DONE)
    manual_cols = set(manual_df.columns)
    extracted_cols = set(extracted_df.columns)
    overlapping_cols = list(manual_cols & extracted_cols - set(key_cols) - {"DONE"})

    # Always include Parameter Template for updates
    if (
        "Parameter Template" not in overlapping_cols
        and "Parameter Template" in extracted_cols
    ):
        overlapping_cols.append("Parameter Template")

    print(f"Key columns: {key_cols}")
    print(f"Overlapping columns to update: {overlapping_cols}")
    print(f"Total rows in manual: {len(manual_df)}")
    print(f"Rows marked DONE: {(manual_df['DONE'] == 'x').sum()}")

    # Create lookup from extracted_manifests
    extracted_df["_key"] = (
        extracted_df["Source PDF"].astype(str)
        + "_"
        + extracted_df["Manifest Number"].astype(str)
    )
    extracted_lookup = extracted_df.set_index("_key")

    # Track updates
    updated_count = 0

    # Update rows not marked as done
    for idx, row in manual_df.iterrows():
        if row.get("DONE") == "x":
            continue  # Skip completed rows

        # Build key
        key = f"{row['Source PDF']}_{row['Manifest Number']}"

        if key in extracted_lookup.index:
            extracted_row = extracted_lookup.loc[key]
            # Handle duplicate keys (take first if multiple)
            if isinstance(extracted_row, pd.DataFrame):
                extracted_row = extracted_row.iloc[0]

            # Update overlapping columns
            for col in overlapping_cols:
                if col in extracted_row.index:
                    manual_df.at[idx, col] = extracted_row[col]

            updated_count += 1

    print(f"Updated {updated_count} rows")

    # Ensure original column order
    manual_df = manual_df[original_columns]

    # Save
    manual_df.to_csv(MANUAL_PATH, index=False)
    print(f"Saved to {MANUAL_PATH}")


if __name__ == "__main__":
    main()
