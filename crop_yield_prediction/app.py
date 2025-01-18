import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open(r"C:\Users\anu54\OneDrive\Desktop\crop_yield_prediction\crop_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

# Check if the model loaded correctly and is a valid model
if isinstance(model, object) and hasattr(model, 'predict'):
    st.write("Model loaded successfully!")
else:
    st.write("Model loading failed or is not a valid model object.")

# Input fields for the app
st.title("Crop Yield Prediction App")

# Collecting user inputs using Streamlit widgets
temperature = st.slider("Temperature (°C)", 0, 50, 25)
rainfall = st.slider("Rainfall (mm)", 0, 500, 100)
humidity = st.slider("Humidity (%)", 0, 100, 50)

# Make prediction when the button is clicked
if st.button("Predict"):
    # Reshape the input features to match model input (2D array)
    features = np.array([[temperature, rainfall, humidity]])

    # Ensure that features are in the correct format (2D array)
    st.write(f"Input Features: {features}")

    # Make the prediction using the model
    try:
        prediction = model.predict(features)
        st.write(f"Predicted Crop Yield: {prediction[0]:.2f} tons/hectare")
    except Exception as e:
        st.write(f"Error in prediction: {e}")
