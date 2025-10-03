import pandas as pd
from sklearn.model_selection import train_test_split
import os

def stratified_split_datasets(train_size=0.8):
    """
    Loads multiple CSVs, assigns a 'domain' label to each, combines them,
    and then performs a stratified split to ensure each domain is proportionally
    represented in the train, validation, and test sets.
    """
    # --- Configuration ---
    # A dictionary where keys are the domain name and values are the file paths.
    datasets_to_load = {
        'set_theory': 'set_theory_dataset.csv',
        'logic': 'logic_dataset.csv',
        'linear_algebra': 'linear_algebra.csv'
    }
    
    all_dfs = []
    print("Starting the stratified splitting process...")

    # --- 1. Load data and add domain labels ---
    for domain, file_path in datasets_to_load.items():
        if not os.path.exists(file_path):
            print(f"  - Warning: File '{file_path}' not found. Skipping this domain.")
            continue
        
        # Read the CSV and ensure column names are consistent for concatenation
        df = pd.read_csv(file_path)
        df = df.rename(columns={df.columns[0]: 'Symbolic_Maths', df.columns[1]: 'Natural_Language'})

        # Add the new 'domain' column
        df['domain'] = domain
        all_dfs.append(df)
        print(f"  - Loaded '{file_path}' with {len(df)} rows for domain '{domain}'.")

    if not all_dfs:
        print("\nError: No data was loaded. Halting script.")
        return

    # --- 2. Combine all dataframes ---
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\nTotal combined rows: {len(combined_df)}")
    print("Original domain distribution:")
    print(combined_df['domain'].value_counts(normalize=True).map('{:.2%}'.format))


    # --- 3. Perform the Stratified Split ---
    # We use the 'domain' column for stratification.
    labels = combined_df['domain']

    # First split: 90% for train+validation, 10% for test
    train_val_df, test_df = train_test_split(
        combined_df,
        train_size=0.9,
        random_state=42,  # for reproducibility
        stratify=labels   # This ensures the split is balanced by domain
    )

    # Second split: Split the 90% into 80% train and 10% validation
    val_proportion = 0.1 / 0.9  # This calculates to ~11.1% of the 90% chunk
    train_df, val_df = train_test_split(
        train_val_df,
        train_size=(1 - val_proportion),
        random_state=42,
        stratify=train_val_df['domain'] # Stratify the second split as well
    )
    
    # --- 4. Save the final files ---
    # We drop the 'domain' column as it's not needed for the model itself
    train_df[['Symbolic_Maths', 'Natural_Language']].to_csv("train.csv", index=False)
    val_df[['Symbolic_Maths', 'Natural_Language']].to_csv("validation.csv", index=False)
    test_df[['Symbolic_Maths', 'Natural_Language']].to_csv("test.csv", index=False)
    
    print("\nStratified splitting complete!")
    print(f"\n--- Output Files ---")
    print(f"train.csv:      {len(train_df)} rows")
    print("Train set distribution:")
    print(train_df['domain'].value_counts(normalize=True).map('{:.2%}'.format))
    
    print(f"\nvalidation.csv: {len(val_df)} rows")
    print("Validation set distribution:")
    print(val_df['domain'].value_counts(normalize=True).map('{:.2%}'.format))
    
    print(f"\ntest.csv:       {len(test_df)} rows")
    print("Test set distribution:")
    print(test_df['domain'].value_counts(normalize=True).map('{:.2%}'.format))


if __name__ == '__main__':
    # We'll split the dataset into 80% for training, 10% for validation, and 10% for testing.
    stratified_split_datasets(train_size=0.8)

