import streamlit as st
st.set_page_config(page_title="Car Price Prediction",page_icon="car-icon.png",layout="centered")
import pandas as pd
import joblib
from PIL import Image

load_m=joblib.load('car_price_model.pkl')
load_s=joblib.load('scaler.pkl')
load_f=joblib.load('feature_names.pkl')

col1,col2=st.columns([10,6])
with col1:
   st.header("🚗 Car Price Prediction")
   st.write("Enter the details of the car to predict its selling price.")
img = Image.open("car-icon1.jpg")
img = img.resize((200, 120))
with col2:
   st.image(img)


col1,col2=st.columns(2)

with col1:
   year = st.number_input("Year", min_value=1900, max_value=2029, value=2020)
   present_price = st.number_input("Present_Price", min_value=0.0, value=5.0)
   kms_driven = st.number_input("Kms_Driven", min_value=0, value=10000)
   fuel_type = st.selectbox("Fuel_Type", ["Petrol", "Diesel", "CNG"])

with col2:
   seller_type = st.selectbox("Seller_Type", ["Dealer", "Individual"])
   transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
   owner = st.number_input("Owners", min_value=0, value=0)


input_data = pd.DataFrame({
    "Year": [year],
    "Present_Price": [present_price],
    "Kms_Driven": [kms_driven],
    "Fuel_Type": [fuel_type],
    "Seller_Type": [seller_type],
    "Transmission": [transmission],
    "Owner": [owner]
})  


st.subheader("Model Performance")
col1,col2,col3=st.columns(3)
with col1:st.metric("R2",0.962)
with col2:st.metric("MAE",0.610)
with col3:st.metric("RMSE",0.941)

st.divider()

input_encoded=pd.get_dummies(input_data[["Fuel_Type","Seller_Type","Transmission"]])
input_final=pd.concat([input_data[["Year","Present_Price","Kms_Driven","Owner"]],input_encoded],axis=1)
input_final=input_final.reindex(columns=load_f,fill_value=0)
input_scaler=load_s.transform(input_final)
if st.button("Predict Selling Price "):
 predict=load_m.predict(input_scaler)[0]
 st.info("This prediction is an estimate based on the trained Random Forest model")
 st.success(f"price prediction:{predict:.2f} Lakhs")
st.write("Built with Python, Scikit-learn & Streamlit")
