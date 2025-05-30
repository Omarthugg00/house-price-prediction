import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(input_path="data/raw/AmesHousing.csv", output_path="data/processed/cleaned.csv"):
    df = pd.read_csv(input_path)

    # Drop ID-like columns
    df.drop(columns=["Order", "PID"], inplace=True)

    # Drop columns with more than 50% missing values
    threshold = len(df) * 0.5
    df.dropna(thresh=threshold, axis=1, inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Save cleaned version
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path} with shape: {df.shape}")

if __name__ == "__main__":
    clean_data()
