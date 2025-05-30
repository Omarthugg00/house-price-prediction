import pandas as pd

def load_dataset(path="data/raw/AmesHousing.csv"):
    """Loads the Ames Housing dataset."""
    df = pd.read_csv(path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    print("\n--- First 5 rows ---")
    print(df.head())
    
    print("\n--- Column Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False))

    return df

if __name__ == "__main__":
    load_dataset()
