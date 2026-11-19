import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/medicine_demand.csv")

print("Dataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 1. Demand distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["demand"], kde=True)
plt.title("Medicine Demand Distribution")
plt.xlabel("Demand")
plt.show()

# 2. Sales vs demand
plt.figure(figsize=(8, 5))
sns.scatterplot(x="previous_sales", y="demand", data=df)
plt.title("Previous Sales vs Demand")
plt.show()

# 3. Monthly demand
plt.figure(figsize=(8, 5))
sns.barplot(x="month", y="demand", data=df)
plt.title("Average Demand by Month")
plt.show()

# 4. Correlation
plt.figure(figsize=(8, 5))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()
