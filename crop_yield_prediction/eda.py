import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r"C:\Users\anu54\OneDrive\Desktop\crop_yield_prediction\data\Crop_Yield_Dataset.csv")

# Check data overview
print(data.info())
print(data.describe())

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Visualize relationships
sns.pairplot(data, vars=["Temperature", "Rainfall", "Humidity", "Average_Yield"], hue="Region")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
