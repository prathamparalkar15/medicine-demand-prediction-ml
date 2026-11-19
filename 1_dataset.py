import os

os.makedirs("dataset", exist_ok=True)
import pandas as pd
import numpy as np

np.random.seed(42)

medicines = [
    "Paracetamol",
    "Cetirizine",
    "Azithromycin",
    "Vitamin C",
    "Ibuprofen"
]

categories = [
    "Pain Relief",
    "Allergy",
    "Antibiotic",
    "Supplement",
    "Pain Relief"
]

data = []

for i in range(1000):

    medicine_index = np.random.randint(0, 5)

    medicine = medicines[medicine_index]
    category = categories[medicine_index]

    month = np.random.randint(1, 13)

    previous_sales = np.random.randint(50, 300)

    price = np.random.randint(20, 200)

    stock = np.random.randint(20, 400)

    # Generate sample demand with some relationship to sales and season
    seasonal_effect = 20 if month in [6, 7, 8] else 0

    demand = (
        previous_sales * 0.7
        + seasonal_effect
        + np.random.randint(-30, 30)
    )

    demand = max(0, int(demand))

    data.append([
        medicine,
        category,
        month,
        previous_sales,
        price,
        stock,
        demand
    ])

df = pd.DataFrame(data, columns=[
    "medicine",
    "category",
    "month",
    "previous_sales",
    "price",
    "stock",
    "demand"
])

df.to_csv("dataset/medicine_demand.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Total records:", len(df))