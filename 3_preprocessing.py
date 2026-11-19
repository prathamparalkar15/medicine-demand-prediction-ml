import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("dataset/medicine_demand.csv")

# Input features
X = df.drop("demand", axis=1)

# Target variable
y = df["demand"]

# Identify categorical and numerical columns
categorical_features = ["medicine", "category"]

numerical_features = [
    "month",
    "previous_sales",
    "price",
    "stock"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("Preprocessing completed!")