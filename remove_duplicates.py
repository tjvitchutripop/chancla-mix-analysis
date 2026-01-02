import pandas as pd
import os

def remove_duplicates(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load data
    df = pd.read_csv(file_path)
    initial_count = len(df)
    
    # Check if 'Spotify Track Id' column exists
    if 'Spotify Track Id' not in df.columns:
        print("Error: 'Spotify Track Id' column not found in CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Remove duplicates based on Spotify Track Id
    # keep='first' is default, which keeps the first occurrence
    df_cleaned = df.drop_duplicates(subset=['Spotify Track Id'], keep='first')
    
    final_count = len(df_cleaned)
    removed_count = initial_count - final_count

    # Save back to CSV
    df_cleaned.to_csv(file_path, index=False)
    
    print(f"Initial rows: {initial_count}")
    print(f"Removed: {removed_count} duplicates")
    print(f"Final rows: {final_count}")
    print(f"Successfully updated {file_path}")

if __name__ == "__main__":
    remove_duplicates('data_new.csv')
