import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_delay.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Input features
delivery_distance = st.number_input('Delivery Distance', min_value=0.0, max_value=100.0, value=20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-3)', 1, 3, 2)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, max_value=30, value=5)
num_stops = st.number_input('Number of Stops', min_value=0, max_value=15, value=3)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, max_value=20, value=4)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.number_input('Package Weight (kg)', min_value=0.0, max_value=50.0, value=10.0)
fuel_efficiency = st.number_input('Fuel Efficiency (km/L)', min_value=0.0, max_value=30.0, value=15.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, max_value=100, value=45)

# Create a DataFrame from inputs
input_data = pd.DataFrame([
    [
        delivery_distance, 
        traffic_congestion, 
        weather_condition, 
        delivery_slot, 
        driver_experience, 
        num_stops, 
        vehicle_age, 
        road_condition_score, 
        package_weight, 
        fuel_efficiency, 
        warehouse_processing_time
    ]
], columns=[
    'Delivery_Distance', 
    'Traffic_Congestion', 
    'Weather_Condition', 
    'Delivery_Slot', 
    'Driver_Experience', 
    'Num_Stops', 
    'Vehicle_Age', 
    'Road_Condition_Score', 
    'Package_Weight', 
    'Fuel_Efficiency', 
    'Warehouse_Processing_Time'
])

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'Likely to have a **Delay** (Probability: {prediction_proba[0][1]:.2f})')
    else:
        st.success(f'Likely **No Delay** (Probability: {prediction_proba[0][0]:.2f})')

    st.write('---')
    st.subheader('Feature Inputs:')
    st.write(input_data)
