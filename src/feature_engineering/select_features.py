import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def select_features(input_path="data/processed/cleaned.csv", output_path="data/processed/selected.csv", n_features=20):
    df = pd.read_csv(input_path)

    # Separate features and target
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # Split temporarily for feature selection (optional)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest to rank features
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Get top N feature names
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(n_features).index.tolist()

    print("✅ Top features selected:", top_features)

    # Save reduced dataset with only top features + target
    df_selected = df[top_features + ["SalePrice"]]
    df_selected.to_csv(output_path, index=False)
    print(f"🎯 Selected data saved to {output_path} with shape: {df_selected.shape}")

if __name__ == "__main__":
    select_features()
