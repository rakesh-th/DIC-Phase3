import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
st.header("Credit Card Approval Prediction")
st.text_input("Enter your Name: ", key="name")
my_data = pd.read_csv("final_application.csv")
data = pd.read_csv("https://raw.githubusercontent.com/gurokeretcha/WishWeightPredictionApplication/master/Fish.csv")
#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data

st.subheader("Please provide details of your application!")
left_column, right_column = st.columns(2)
with left_column:
    inp_Gender = st.radio(
        'Gender of the Person:',
        np.unique(my_data['Gender']))
 
left_column, right_column = st.columns(2)
with left_column:
    inp_Car = st.radio(
        'Does he own a Car:',
        np.unique(my_data['Car']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Realty = st.radio(
        'Does he own any Realty(properties):',
        np.unique(my_data['Realty']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Income_Type = st.radio(
        'Type of Income',
        np.unique(my_data['Income_Type']))   

left_column, right_column = st.columns(2)
with left_column:
    inp_Education = st.radio(
        'Education of the Applicant',
        np.unique(my_data['Education']))  

input_Children = st.slider('Number of Children:', 0, max(my_data["Children"]), 0)
input_Family_Size = st.slider('Family Size:', 0, max(my_data["Family_Size"]), 0)
input_Salary = st.slider('Salary of the Applicant:', 0.0, max(my_data["Salary"]), 0.0)
input_AGE = st.slider('Age in Years:', 0.0, max(my_data["AGE"]), 0.0)
input_EXPERIENCE = st.slider('Experience in Years:', 0.0, max(my_data["EXPERIENCE"]), 0.0)
input_Height = st.slider('Height(cm)', 0.0, max(data["Height"]), 1.0)
input_Width = st.slider('Diagonal width(cm)', 0.0, max(data["Width"]), 1.0)


if st.button('Make Prediction'):
    input_species = encoder.transform(np.expand_dims(inp_species, -1))
    inputs = np.expand_dims(
        [int(input_species), input_Length1, input_Length2, input_Length3, input_Height, input_Width], 0)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your fish weight is: {np.squeeze(prediction, -1):.2f}g")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
    st.write(f"If you want to see more advanced applications you can follow me on [medium](https://medium.com/@gkeretchashvili)")