# This is the code I will run once you upload the CSV file.
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'deformalization_dataset.csv'
try:
    df = pd.read_csv(file_path)
    
    df.rename(columns={
        'input': 'input_text',
        'output': 'target_text'
    }, inplace=True)
    
    df.dropna(inplace=True)

    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=(1/9), random_state=42)

    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('validation_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)

    print("CSV data preparation complete!")
    print(f"Training data saved to 'train_data.csv'. Shape: {train_df.shape}")
    print(f"Validation data saved to 'validation_data.csv'. Shape: {val_df.shape}")
    print(f"Test data saved to 'test_data.csv'. Shape: {test_df.shape}")

except Exception as e:
    print(f"An error occurred: {e}")