import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# =================================================================*
# 1. Load dataset
# =================================================================
housing = pd.read_csv("housing.csv")

# =================================================================
# 2. Stratified Train-Test Split
# =================================================================
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# remove income_cat (not used in training)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# =================================================================
# 3. Separate labels
# =================================================================
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# =================================================================
# 4. Data Preprocessing Pipelines
# =================================================================
num_attribs = list(housing.drop("ocean_proximity", axis=1))
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# transform data
housing_prepared = full_pipeline.fit_transform(housing)

# =================================================================
# 5. Train Model
# =================================================================
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# =================================================================
# 6. Cross-validation
# =================================================================
scores = cross_val_score(
    forest_reg, housing_prepared, housing_labels,
    scoring="neg_mean_squared_error", cv=10
)
rmse_scores = np.sqrt(-scores)

print("\n--- Cross-validation Results ---")
print("RMSE scores:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
print("Standard deviation:", rmse_scores.std())

# =================================================================
# 7. Final Test Set Evaluation
# =================================================================
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = forest_reg.predict(X_test_prepared)

final_mse = np.mean((final_predictions - y_test) ** 2)
final_rmse = np.sqrt(final_mse)

print("\nFinal RMSE on Test Set:", final_rmse)

# =================================================================
# 8. Save model, pipeline and predictions
# =================================================================
joblib.dump(forest_reg, "model.pkl")
joblib.dump(full_pipeline, "pipeline.pkl")

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": final_predictions
})
comparison.to_csv("output.csv", index=False)

print("\nArtifacts saved:")
print("- Trained model → model.pkl")
print("- Preprocessing pipeline → pipeline.pkl")
print("- Predictions comparison → output.csv")

# =================================================================
# 9. Load & Predict on New Data
# =================================================================
print("\n--- Demo: New Predictions ---")
loaded_model = joblib.load("model.pkl")
loaded_pipeline = joblib.load("pipeline.pkl")

new_data = strat_test_set.drop("median_house_value", axis=1).iloc[:5]
new_data_prepared = loaded_pipeline.transform(new_data)
new_predictions = loaded_model.predict(new_data_prepared)

print("\nNew data:\n", new_data)
print("\nPredictions:", new_predictions)








