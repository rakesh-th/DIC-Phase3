import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import ADASYN
from sklearn.metrics import *
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

st.header("Credit Card Approval Prediction")
st.text_input("Enter your Name: ", key="name")
my_data = pd.read_csv("application.csv")

if st.checkbox('Show Training Dataframe'):
    my_data

st.subheader("Please provide details of your application!")
left_column, right_column = st.columns(2)
with left_column:
    inp_Gender = st.radio(
        'Gender of the applicant:',
        np.unique(my_data['Gender']))
 
left_column, right_column = st.columns(2)
with left_column:
    inp_Car = st.radio(
        'Does the applicant own a car:',
        np.unique(my_data['Car']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Realty = st.radio(
        'Does the applicant own any Realty(properties):',
        np.unique(my_data['Realty']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Family_Status = st.radio(
        'Is applicant Single/Married:',
        np.unique(my_data['Family_Status']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Income_Type = st.radio(
        "What is the applicant's source of Income:",
        np.unique(my_data['Income_Type']))   

left_column, right_column = st.columns(2)
with left_column:
    inp_House_Type = st.radio(
        "What is the applicant'type of House",
        np.unique(my_data['House_Type']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Education = st.radio(
        'Education of the Applicant',
        np.unique(my_data['Education']))  

input_Children = st.slider('Number of Children:', 0, max(my_data["Children"]), 3)
input_Family_Size = st.slider('Family Size:', 0, max(my_data["Family_Size"]), 4)
input_Salary = st.slider('Salary of the Applicant:', 0.0, max(my_data["Salary"]), 85000.0)
input_AGE = st.slider('Age of the applicant in Years:', 0.0, max(my_data["AGE"]), 22.8)
input_EXPERIENCE = st.slider('Experience of the applicant in Years:', 0.0, max(my_data["EXPERIENCE"]), 7.5)
input_ACCOUNT_DURATION = st.slider('Account Duration with the bank in Months:', 0, max(my_data["ACCOUNT_DURATION"]), 18)


# load model
#best_xgboost_model = XGBClassifier()
#best_xgboost_model.load_model("best_model.json")

le1=LabelEncoder()
le2=LabelEncoder()
le3=LabelEncoder()
le4=LabelEncoder()
my_data['Income_Type']=le1.fit_transform(my_data['Income_Type'].values)
le1.classes_
my_data['Family_Status']=le2.fit_transform(my_data['Family_Status'].values)
le2.classes_
my_data['House_Type']=le3.fit_transform(my_data['House_Type'].values)
le3.classes_
my_data['Education']=le4.fit_transform(my_data['Education'].values)
le4.classes_


if st.button('Make Prediction'):
    inputs = np.expand_dims([inp_Gender, inp_Car, inp_Realty, input_Children, input_Salary, inp_Income_Type, inp_Education, inp_Family_Status, inp_House_Type, input_AGE, input_EXPERIENCE, input_Family_Size, input_ACCOUNT_DURATION],0)
    prediction = best_xgboost_model.predict(inputs)
    if prediction:
        st.write("Your Credit Card is Declined")
    else:
        st.write("Your Credit Card is Approved")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
