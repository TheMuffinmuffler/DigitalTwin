import pandas as pd
import os


def load_and_clean_data(file_path='Data/data.csv'):
    """
    Imports the data, strips column names, and removes duplicates/missing values.
    """
    # Logic to handle running the script from either project root or src/
    if not os.path.exists(file_path) and os.path.exists(os.path.join('..', file_path)):
        file_path = os.path.join('..', file_path)

    # 1. Load data
    df = pd.read_csv(file_path)

    # Remove trailing spaces from headers
    df.columns = df.columns.str.strip()

    # 2. Check for duplicates
    duplicate_rows = df[df.duplicated(keep='first')]
    if not duplicate_rows.empty:
        # Save logs to the project root for visibility
        duplicate_rows.to_csv('dropped_duplicates.csv', index=False)
        print(f"Captured {len(duplicate_rows)} duplicate rows to 'dropped_duplicates.csv'.")
    else:
        print("No duplicates found.")

    # 3. Clean the data
    df = df.drop_duplicates()
    df = df.dropna()

    print("Data cleaning complete.")
    return df