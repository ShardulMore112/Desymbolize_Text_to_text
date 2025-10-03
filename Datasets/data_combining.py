import pandas as pd
import glob
import os

def combine_and_prepare_datasets(file_paths, output_filename="final_training_dataset.csv"):
    """
    Loads multiple CSV files, combines them, standardizes column names,
    shuffles the data, and saves it to a new CSV file.

    Args:
        file_paths (list): A list of paths to the CSV files to combine.
        output_filename (str): The name of the final output CSV file.
    """
    if not file_paths:
        print("Error: No CSV files found to combine. Please check the file paths.")
        return

    all_dfs = []
    print("Starting the dataset combination process...")

    for file_path in file_paths:
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)

            # --- Smart Column Renaming ---
            # We assume the first column is symbolic and the second is natural language.
            if len(df.columns) >= 2:
                # Get the names of the first two columns
                original_cols = df.columns[:2]
                # Create a mapping from old names to new, standardized names
                rename_map = {
                    original_cols[0]: 'Symbolic_Maths',
                    original_cols[1]: 'Natural_Language'
                }
                df = df.rename(columns=rename_map)
                # Keep only the two standardized columns
                df = df[['Symbolic_Maths', 'Natural_Language']]
                all_dfs.append(df)
                print(f"  - Successfully processed '{os.path.basename(file_path)}' with {len(df)} rows.")
            else:
                print(f"  - Warning: Skipping '{os.path.basename(file_path)}' as it has fewer than 2 columns.")

        except FileNotFoundError:
            print(f"  - Error: The file '{os.path.basename(file_path)}' was not found. Skipping.")
        except Exception as e:
            print(f"  - Error processing '{os.path.basename(file_path)}': {e}. Skipping.")


    if not all_dfs:
        print("\nNo data was processed. Halting script.")
        return

    # Concatenate all the dataframes into a single one
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows before shuffling: {len(combined_df)}")

    # Shuffle the combined DataFrame randomly
    # frac=1 samples 100% of the rows.
    # reset_index(drop=True) cleans up the old index from before shuffling.
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
    print("Successfully shuffled the combined dataset.")

    # Save the final, ready-to-use dataset to a new CSV file
    shuffled_df.to_csv(output_filename, index=False)
    print(f"\nProcess complete! Final dataset saved to '{output_filename}' with {len(shuffled_df)} rows.")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # List the names of all the CSV files you want to combine here.
    # The script assumes they are in the same directory.
    files_to_combine = [
        'set_theory_datasetcsv',
        'logic_dataset.csv',
        'linear_algebra.csv'
        # Add any other CSV file names here
    ]

    # You can also use glob to automatically find all .csv files
    # For example: files_to_combine = glob.glob("*.csv")

    combine_and_prepare_datasets(files_to_combine)
