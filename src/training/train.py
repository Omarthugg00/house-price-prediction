import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"📊 {name} → R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return r2, model

def train_models(input_path="data/processed/selected.csv", model_dir="models/"):
    df = pd.read_csv(input_path)
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    best_score = -np.inf
    best_model = None
    best_name = ""

    os.makedirs(model_dir, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        r2, trained_model = evaluate_model(name, model, X_test, y_test)

        # Save each model separately
        joblib.dump(trained_model, f"{model_dir}/{name.replace(' ', '_').lower()}.pkl")

        if r2 > best_score:
            best_score = r2
            best_model = trained_model
            best_name = name

    # Save best model as 'best_model.pkl'
    joblib.dump(best_model, f"{model_dir}/best_model.pkl")
    print(f"\n✅ Best model saved: {best_name} with R² = {best_score:.4f}")

if __name__ == "__main__":
    train_models()
