from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from datetime import datetime
import os
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("../models/best_model.pkl")

# Feature names mapping
features = {
    'Overall Qual': 'Overall Quality (1-10)',
    'Gr Liv Area': 'Above Ground Living Area (sqft)',
    '1st Flr SF': 'First Floor Area (sqft)',
    'Total Bsmt SF': 'Total Basement Area (sqft)',
    '2nd Flr SF': 'Second Floor Area (sqft)',
    'BsmtFin SF 1': 'Finished Basement Area 1 (sqft)',
    'Full Bath': 'Number of Full Bathrooms',
    'Lot Area': 'Lot Area (sqft)',
    'Garage Cars': 'Garage Capacity (cars)',
    'Garage Area': 'Garage Area (sqft)',
    'Year Built': 'Year Built',
    'Year Remod/Add': 'Year Remodeled',
    'Kitchen Qual': 'Kitchen Quality (1–5)',
    'Lot Frontage': 'Lot Frontage (feet)',
    'Bsmt Unf SF': 'Unfinished Basement Area (sqft)',
    'Mas Vnr Area': 'Masonry Veneer Area (sqft)',
    'Neighborhood': 'Neighborhood',
    'Garage Yr Blt': 'Garage Year Built',
    'Screen Porch': 'Screen Porch Area (sqft)',
    'Wood Deck SF': 'Wood Deck Area (sqft)'
}

# Neighborhood dropdown
neighborhood_choices = {
    'Blmngtn': 0, 'Blueste': 1, 'BrDale': 2, 'BrkSide': 3, 'ClearCr': 4,
    'CollgCr': 5, 'Crawfor': 6, 'Edwards': 7, 'Gilbert': 8, 'IDOTRR': 9,
    'MeadowV': 10, 'Mitchel': 11, 'NAmes': 12, 'NPkVill': 13, 'NWAmes': 14,
    'NoRidge': 15, 'NridgHt': 16, 'OldTown': 17, 'SWISU': 18, 'Sawyer': 19,
    'SawyerW': 20, 'Somerst': 21, 'StoneBr': 22, 'Timber': 23, 'Veenker': 24
}

LOG_FILE = "../logs/predictions.csv"

def log_prediction(input_data, predicted_price):
    os.makedirs("../logs", exist_ok=True)
    data_dict = {k: [v] for k, v in zip(features.keys(), input_data)}
    data_dict["Prediction"] = [int(predicted_price)]
    data_dict["Timestamp"] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    df = pd.DataFrame(data_dict)
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

@app.route("/")
def index():
    return render_template("index.html", features=features, neighborhoods=neighborhood_choices)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = []
    for key in features:
        value = request.form[key]

        if key == "Neighborhood":
            neighborhood_str = value
            value = neighborhood_choices.get(value, 0)

        input_data.append(float(value))

    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    # Replace numeric neighborhood back with string for logging
    input_data_for_logging = input_data.copy()
    neighborhood_index = list(features.keys()).index("Neighborhood")
    input_data_for_logging[neighborhood_index] = neighborhood_str

    log_prediction(input_data_for_logging, prediction)

    return render_template("result.html", price=int(prediction))

@app.route("/admin", methods=["GET"])
def admin_dashboard():
    try:
        df = pd.read_csv(LOG_FILE)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp")

        # Time-series chart
        df["Date"] = df["Timestamp"].dt.date
        ts_data = df.groupby("Date")["Prediction"].mean().reset_index()
        chart_labels = ts_data["Date"].astype(str).tolist()
        chart_values = ts_data["Prediction"].astype(int).tolist()

        # Pie chart: count by neighborhood
        pie_data = df["Neighborhood"].value_counts()
        pie_labels = pie_data.index.tolist()
        pie_values = pie_data.values.tolist()

    except Exception as e:
        print(e)
        df = pd.DataFrame()
        chart_labels, chart_values = [], []
        pie_labels, pie_values = [], []

    return render_template(
        "admin.html",
        records=df.to_dict(orient="records"),
        columns=df.columns if not df.empty else [],
        chart_labels=chart_labels,
        chart_values=chart_values,
        pie_labels=pie_labels,
        pie_values=pie_values
    )

if __name__ == "__main__":
    app.run(debug=True)
