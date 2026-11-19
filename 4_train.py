import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load dataset
df = pd.read_csv("dataset/medicine_demand.csv")

X = df.drop("demand", axis=1)
y = df["demand"]

# 2. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         ["medicine", "category"])
    ],
    remainder="passthrough"
)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

# 4. Linear Regression
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

predictions = linear_model.predict(X_test)

print("\nLinear Regression training completed!")

# 5. Random Forest
random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)

print("Random Forest training completed!")

# 6. Evaluation
linear_mae = mean_absolute_error(y_test, predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, predictions))
linear_r2 = r2_score(y_test, predictions)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

# 7. Model comparison
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, rf_mae],
    "RMSE": [linear_rmse, rf_rmse],
    "R2 Score": [linear_r2, rf_r2]
})

print("\n========== MODEL COMPARISON ==========")
print(results.round(2))

# 8. Select best model
if linear_mae <= rf_mae:
    best_model = linear_model
    best_model_name = "Linear Regression"
else:
    best_model = random_forest_model
    best_model_name = "Random Forest"

# 9. Save model
joblib.dump(best_model, "medicine_model.pkl")

print("\nBest Model:", best_model_name)
print("Model saved successfully!")

# 10. Graph
plt.figure(figsize=(8, 5))

plt.scatter(y_test, rf_predictions)

plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")

plt.title("Actual vs Predicted Demand - Random Forest")

plt.grid(True)
plt.show()