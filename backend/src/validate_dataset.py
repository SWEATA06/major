import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Loading final dataset...")
    df = pd.read_csv('data/final/final_dataset.csv')
    
    print("\n=== STAGE 1: Dataset Validation ===")
    
    print(f"\n[1] Dataset Shape: {df.shape}")
    
    print("\n[2] Datatype Check:")
    print(df.dtypes)
    
    print("\n[3] Missing Value Check:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if not missing[missing > 0].empty else "SUCCESS: No missing values found!")
    
    print("\n[4] Duplicate Rows Check:")
    dupes = df.duplicated().sum()
    print(f"Found {dupes} duplicate rows.")
    if dupes > 0:
        print("Note: In time-series telemetry, consecutive identical readings can happen naturally, but should be monitored.")
    
    print("\n[5] Class Balance for targets:")
    print("Failure Label:", df['failure_label'].value_counts(normalize=True).round(4).to_dict())
    print("Overload Label:", df['overload_label'].value_counts(normalize=True).round(4).to_dict())
    
    print("\n[6] Leakage Check (Correlations):")
    corrs = df.corr()
    print("Top features correlated with Future Failure:")
    # Checking against failure_label as a proxy
    print(corrs['failure_label'].sort_values(ascending=False)[1:6])
    
    print("\n[7] Feature Importance Baseline (Random Forest Check):")
    # Identify robust cols
    exclude = ['failure_label', 'overload_label']
    features = [c for c in df.columns if c not in exclude]
    X = df[features]
    y = df['failure_label']
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("Top 5 most important baseline features:\n", importances.head(5))
    
    # Save correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corrs, cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap (Checking Leakage/Circularity)")
    plt.savefig('data/correlation_heatmap.png')
    print("\nHeatmap saved as 'data/correlation_heatmap.png'")

if __name__ == "__main__":
    main()
