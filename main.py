import streamlit as st
import numpy as np
import pickle
import pandas as pd 
from sklearn.preprocessing import LabelEncoder  
# Load model 
with open(r"C:\Users\lenovo\Documents\project\VehiclePrice\vehiclePrice.pkl","rb") as f:
    model= pickle.load(f)
data = pd.read_csv(r"C:\Users\lenovo\Documents\project\VehiclePrice\Vehicle Price.csv")

st.title("Vehicle Price Predictor")

label_encoder= LabelEncoder()
categories = data['name'].unique()
data['name_encoded'] = label_encoder.fit_transform(data['name'])
data['model_encoded'] = label_encoder.fit_transform(data['model'])
data['engine_encoded'] = label_encoder.fit_transform(data['engine'])
data['fuel_encoded'] = label_encoder.fit_transform(data['fuel'])
data['transmission_encoded'] = label_encoder.fit_transform(data['transmission'])
data['body_encoded'] = label_encoder.fit_transform(data['body'])
data['exterior_color_encoded'] = label_encoder.fit_transform(data['exterior_color'])
data['drivetrain_encoded'] = label_encoder.fit_transform(data['drivetrain'])

feature_names = ['name']

# Create a selectbox for each feature
# and a number input for numerical features
selected_category= st.selectbox("Select name of vehicle:", categories)
name_encoded = data[data['name'] == selected_category]['name'].iloc[0]
filtered_data = data[data['name'] == selected_category]


if  st.button("Display Details"):
    selected_row = filtered_data.iloc[0]
    st.subheader("Vehicle Details")
    st.write(f"**Model:** {selected_row['model']}")
    st.write(f"**Engine:** {selected_row['engine']}")
    st.write(f"**Fuel Type:** {selected_row['fuel']}")
    st.write(f"**Transmission:** {selected_row['transmission']}")
    st.write(f"**Body Type:** {selected_row['body']}")
    st.write(f"**Exterior Color:** {selected_row['exterior_color']}")
    st.write(f"**Drivetrain:** {selected_row['drivetrain']}")
    
name_encoded = filtered_data.loc[filtered_data['name'] == selected_category]['name_encoded'].iloc[0]
selected_row = filtered_data.iloc[0]
input_values = [
    selected_row['name_encoded'],
    selected_row['model_encoded'],
    selected_row['engine_encoded'],
    selected_row['fuel_encoded'],
    selected_row['transmission_encoded'],
    selected_row['body_encoded'],
    selected_row['exterior_color_encoded'],
    selected_row['drivetrain_encoded'],
    selected_row['price'],
    selected_row['mileage']
]

if  st.button("Display Price Of The Vehicle"):
    input_array = np.array([input_values])  # shape (1, 10)
    prediction = model.predict(input_array)

    st.success(f" The Predicted Price is : {prediction[0]}")