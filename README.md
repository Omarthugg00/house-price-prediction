# 🏠 House Price Prediction Project

A complete end-to-end Machine Learning project for predicting house prices using the Ames Housing dataset. The project includes data preprocessing, model training, evaluation, deployment with Flask, and an admin dashboard with interactive charts.

---

## 📌 Project Overview

This project predicts house sale prices based on features like house quality, square footage, garage information, and neighborhood. It includes:

- An ML pipeline with multiple models
- Clean and interactive web UI for users
- Admin dashboard with logged predictions and charts
- Prediction logging with timestamps

---

## 📊 Dataset

- **Name**: Ames Housing Dataset  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)  
- **Size**: ~3,000 rows × 80+ features  
- **Target**: `SalePrice` (house sale price in USD)

---

## 🧠 ML Pipeline Steps

### 1. Data Preprocessing
- Loaded data from `data/raw/`
- Removed outliers
- Dropped or imputed missing values
- Cleaned dataset saved to `data/processed/cleaned.csv`

### 2. Feature Selection
- Selected top 20 features based on correlation + domain expertise
- Saved reduced dataset to `data/processed/selected.csv`

### 3. Model Training & Evaluation
- Trained and compared:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - ✅ XGBoost (**Best R²: 0.91**)

- Metrics:
  - R² Score
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

### 4. Model Saving
- Best model saved to `models/best_model.pkl`

### 5. Prediction Logging
- Logs every prediction (features + result + timestamp) to `logs/predictions.csv`

### 6. Web App Deployment
- Flask-based frontend for predictions
- Admin dashboard with a table and pie chart

---

## ⚙️ Setup Instructions

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/Omarthugg00/house-price-prediction
cd house-price-prediction
```

### 🐍 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# or
source venv/bin/activate   # On macOS/Linux
```

### 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔁 Run the Full ML Pipeline

This runs the full pipeline: preprocessing → feature selection → model training → saving the best model.

```bash
python main.py
```

> 📂 `main.py` is located in the root folder. It handles the entire ML workflow end-to-end.

---

## 🚀 Run the Flask Web App

```bash
cd app
python main.py
```

Open in browser: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 👤 User Features

- Fill in a clean form with:
  - Square footage, year built, bathrooms, neighborhood, etc.
- Predict the house price instantly
- View result on the result page

---

## 🧑‍💼 Admin Dashboard

- Access via: [http://127.0.0.1:5000/admin](http://127.0.0.1:5000/admin)
- View logged predictions in a table
- Interactive pie chart for neighborhood distribution
- Simple, futuristic design

---

## 🗂️ Project Structure

```
.
├── data/
│   ├── raw/                   # Original dataset (CSV)
│   └── processed/             # Cleaned and selected feature data
│
├── models/
│   └── best_model.pkl         # Final trained model
│
├── logs/
│   └── predictions.csv        # Logged predictions
│
├── notebooks/
│   ├── eda_cleaned.ipynb      # EDA on cleaned data
│   └── eda_selected.ipynb     # EDA on selected features
│
├── src/
│   ├── preprocessing/         # Missing value handling, type conversions
│   ├── feature_engineering/   # Feature selection logic
│   ├── training/              # Model training scripts
│   └── eval/                  # Evaluation metrics and plotting
│
├── app/
│   ├── templates/
│   │   ├── index.html         # User form
│   │   ├── result.html        # Result display
│   │   └── admin.html         # Admin dashboard
│   └── main.py                # Flask web app
│
├── main.py                    # 🔁 Main ML pipeline script
├── requirements.txt
└── README.md
```

---

## 🧪 Example Prediction Input

- **Overall Quality (1-10)**: `7`
- **Above Ground Living Area (sqft)**: `2200`
- **First Floor Area (sqft)**: `1100`
- **Garage Capacity (cars)**: `2`
- **Kitchen Quality (1–5)**: `4`
- **Neighborhood**: `Edwards`

🧾 **Predicted Sale Price**: `$207,000`

---

## ✅ Status

✔ Complete ML pipeline  
✔ Local Flask app  
✔ Admin dashboard  
✔ Logged predictions  
✔ Styled responsive UI  
✔ Deployment-ready

---

## 🧑‍🏫 Authors

- Kamal Rabie  
- Omar Aboelwafa
- Omar Fouad

---

