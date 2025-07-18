import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np
import json


# Load dataset
df = pd.read_csv("data/housing.csv")

# Separate features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Define feature types
numeric_features = [
    "longitude", "latitude", "housing_median_age", "total_rooms", 
    "total_bedrooms", "population", "households", "median_income"
]
categorical_features = ["ocean_proximity"]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Show a few actual vs predicted values
print("\nSample Predictions:")
for actual, pred in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual:,.2f}, Predicted: {pred:,.2f}")

# Save the model
joblib.dump(pipeline, "artifacts/california_housing_model.joblib")
print("\n✅ Model saved as 'california_housing_model.joblib'")

# Extract model and features
lr_model = pipeline.named_steps["regressor"]
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

# 1. Save coefficients to CSV
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": lr_model.coef_
})
coef_df.to_csv("artifacts/model_coefficients.csv", index=False)
print("✅ Coefficients saved to 'model_coefficients.csv'")

# 2. Save intercept to TXT
with open("artifacts/model_intercept.txt", "w") as f:
    f.write(f"Intercept: {lr_model.intercept_:.6f}\n")
print("✅ Intercept saved to 'model_intercept.txt'")

# 3. Save metrics to JSON
metrics = {
    "RMSE": round(rmse, 2),
    "MAE": round(mae, 2),
    "R2": round(r2, 4)
}

with open("artifacts/evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluation metrics saved to 'evaluation_metrics.json'")