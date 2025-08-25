

 Housing Price Prediction

This project predicts **median house values** in California districts using the **California Housing dataset**.
It demonstrates a full **Machine Learning pipeline**: data preprocessing, feature engineering, model training, evaluation, and saving artifacts for later use.

---

## 📌 Features

* Data handling with **Pandas** & **NumPy**
* Train-test splitting using **Stratified Sampling**
* Data preprocessing with **Pipelines** (imputation, scaling, encoding)
* Model training using **Random Forest Regressor**
* Cross-validation for robust evaluation
* Saving and loading models using **Joblib**
* Final evaluation on test set
* Exporting predictions to CSV

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/housing-price-prediction.git
cd housing-price-prediction
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
├── main.py              # Main script with full pipeline
├── housing.csv          # Dataset (or provide download link)
├── model.pkl            # Trained model (ignored if too large)
├── pipeline.pkl         # Preprocessing pipeline
├── output.csv           # Predictions vs Actual values
├── requirements.txt     # Dependencies
└── README.md            # Project description
```

---

## ▶️ Usage

### Train & Evaluate the Model

```bash
python main.py
```

This will:

* Train the Random Forest model
* Perform cross-validation
* Evaluate on test set
* Save artifacts:

  * `model.pkl` → Trained model
  * `pipeline.pkl` → Preprocessing pipeline
  * `output.csv` → Predictions comparison

---

## 📊 Sample Output

```
--- Cross-validation Results ---
RMSE scores: [51039.08053738 48741.94041426 45940.42771745 50501.41453432
 47387.7896427  49595.25845731 51625.68567717 48865.70709952
 47322.87631489 53301.08748462]
Mean RMSE: 49432.12678796127
Standard deviation: 2124.8587921578355

Final RMSE on Test Set: 47197.66824186381

Artifacts saved:
- Trained model → model.pkl
- Preprocessing pipeline → pipeline.pkl
- Predictions comparison → output.csv

--- Demo: New Predictions ---

New data:
        longitude  latitude  housing_median_age  ...  households  median_income  ocean_proximity
5241     -118.39     34.12                29.0  ...       960.0         8.2816        <1H OCEAN
17352    -120.42     34.89                24.0  ...       283.0         5.0099        <1H OCEAN
3505     -118.45     34.25                36.0  ...       275.0         4.3839        <1H OCEAN
7777     -118.10     33.91                35.0  ...       301.0         3.2708        <1H OCEAN
14155    -117.07     32.77                38.0  ...       614.0         4.3529       NEAR OCEAN

[5 rows x 9 columns]

---

## 📌 Future Improvements

* Try **GridSearchCV** or **RandomizedSearchCV** for hyperparameter tuning
* Add more ML models (Linear Regression, Gradient Boosting, etc.) for comparison
* Deploy with **Flask / FastAPI / Streamlit** for live demo

---



