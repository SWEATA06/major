import pandas as pd
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = os.path.join(base_dir, 'data', 'processed', 'merged_dataset.csv')
    out_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_dataset.csv')
    
    df = pd.read_csv(in_path)

    # Robust column safety check
    df.columns = df.columns.str.strip()

    # remove duplicates
    df = df.drop_duplicates()

    # remove rows with too many nulls
    df = df.dropna()

    # sort by timestamp
    if "Timestamp [ms]" in df.columns:
        df = df.sort_values(by="Timestamp [ms]")
    else:
        print("Warning: 'Timestamp [ms]' not found. Available columns:", df.columns.tolist())

    df.to_csv(out_path, index=False)

    print("Cleaned shape:", df.shape)

if __name__ == "__main__":
    main()