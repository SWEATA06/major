import pandas as pd
import glob
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'raw', '*.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    os.makedirs(processed_dir, exist_ok=True)
    
    files = glob.glob(raw_path)
    if not files:
        print(f"No files found in {raw_path}")
        return

    dfs = []

    for file in files:
        df = pd.read_csv(file, sep=';')
        
        # Robust column cleaning: strips all leading/trailing whitespace including \t
        df.columns = df.columns.str.strip()
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    print("Columns after extraction and cleaning:")
    for col in merged.columns:
        print(f" - '{col}'")

    out_path = os.path.join(processed_dir, "merged_dataset.csv")
    merged.to_csv(out_path, index=False)

    print("Merged shape:", merged.shape)

if __name__ == "__main__":
    main()