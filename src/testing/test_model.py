import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def test_model(model_path="models/best_model.pkl", data_path="data/processed/selected.csv"):
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # Split data again using same seed (to match training split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    model = joblib.load(model_path)

    # Make predictions
    preds = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n✅ Final Test Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Optional: save predictions to CSV
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": preds
    })
    results.to_csv("data/processed/test_predictions.csv", index=False)
    print("📁 Predictions saved to: data/processed/test_predictions.csv")

if __name__ == "__main__":
    test_model()
