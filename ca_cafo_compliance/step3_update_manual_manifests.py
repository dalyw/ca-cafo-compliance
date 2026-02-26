import pandas as pd
import os
import matplotlib.pyplot as plt
from step2b_extract_manifest_parameters import manifest_params

# Paths
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MANUAL_PATH = os.path.join(OUTPUTS_DIR, "2024_manifests_manual.csv")
EXTRACTED_PATH = os.path.join(OUTPUTS_DIR, "2024_manifests_raw.csv")


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

    # Update rows
    for idx, row in manual_df.iterrows():
        is_done = row.get("DONE") == "x"

        # Build key
        key = f"{row['Source PDF']}_{row['Manifest Number']}"

        if key in extracted_lookup.index:
            extracted_row = extracted_lookup.loc[key]
            # Handle duplicate keys (take first if multiple)
            if isinstance(extracted_row, pd.DataFrame):
                extracted_row = extracted_row.iloc[0]

            if is_done:
                continue  # Skip updates for completed rows

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

    # Bar chart comparison of # extracted from automatic vs manual, plus accuracy as third bar (dual y-axis)
    plt.figure(figsize=(12, 7))
    plt_values = []
    accuracy_values = []
    for col in manifest_params.set_index("parameter_key")["parameter_name"].tolist():
        if col in manual_df.columns and col in extracted_df.columns:
            manual_count = manual_df[col].notna().sum()
            extracted_count = extracted_df[col].notna().sum()
            # Calculate accuracy
            merged = pd.merge(
                manual_df[["Source PDF", "Manifest Number", col]],
                extracted_df[["Source PDF", "Manifest Number", col]],
                on=["Source PDF", "Manifest Number"],
                how="inner",
                suffixes=("_manual", "_extracted"),
            )
            merged["match"] = merged.apply(
                lambda x: (pd.isna(x[f"{col}_manual"]) and pd.isna(x[f"{col}_extracted"]))
                or (x[f"{col}_manual"] == x[f"{col}_extracted"]),
                axis=1,
            )
            accuracy = merged["match"].mean() * 100 if len(merged) > 0 else 0
            plt_values.append((col, manual_count, extracted_count, accuracy))
    
    # Sort by manual_count descending
    plt_values.sort(key=lambda x: x[1], reverse=True)
    bar_width = 0.25
    indices = range(len(plt_values))
    manual_counts = [v[1] for v in plt_values]
    extracted_counts = [v[2] for v in plt_values]
    accuracies = [v[3] for v in plt_values]
    labels = [v[0] for v in plt_values]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(
        [i - bar_width for i in indices],
        manual_counts,
        width=bar_width,
        label="Manual",
        alpha=0.7,
        color="green",
    )
    bars2 = ax1.bar(
        indices,
        extracted_counts,
        width=bar_width,
        label="Extracted",
        alpha=0.7,
        color="red",
    )
    bars3 = ax2.bar(
        [i + bar_width for i in indices],
        accuracies,
        width=bar_width,
        label="Accuracy (%)",
        alpha=0.7,
        color="blue",
    )

    ax1.set_xlabel("Parameters")
    ax1.set_ylabel("Count of Extracted Values")
    ax2.set_ylabel("Accuracy (%)")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend([bars1, bars2, bars3], ["Manual", "Extracted", "Accuracy (%)"], loc="upper left")
    ax2.set_ylim(0, 100)
    plt.title("Manual vs Extracted Counts and Accuracy per Parameter")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "manual_vs_extracted_comparison.png"))

    # Print stats
    for i, (col, manual_count, extracted_count, accuracy) in enumerate(plt_values):
        print(f"  {col}: {manual_count}/{len(manual_df)} manual, {extracted_count}/{len(extracted_df)} extracted, {accuracy:.2f}% accuracy")
    
if __name__ == "__main__":
    main()
