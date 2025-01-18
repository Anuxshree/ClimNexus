import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv(r"C:\Users\anu54\OneDrive\Desktop\crop_yield_prediction\data\Crop_Yield_Dataset.csv")

# Data Preprocessing
data['Crop_Type'] = data['Crop_Type'].astype('category').cat.codes  # Encode categorical data
data['Region'] = data['Region'].astype('category').cat.codes

# Features and target
X = data[['Temperature', 'Rainfall', 'Humidity', 'Soil_pH', 'Nitrogen_Content', 'Organic_Matter', 'Crop_Type', 'Region']]
y = data['Average_Yield']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the model (optional for deployment)
import joblib
joblib.dump(model, r"C:\Users\anu54\OneDrive\Desktop\crop_yield_prediction\crop_yield_model.pkl")
