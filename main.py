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

st.header("Credit Card Approval Prediction")
st.text_input("Enter your Name: ", key="name")
my_data = pd.read_csv("final_application.csv")

# load model
#best_xgboost_model = XGBClassifier()
#best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    my_data

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
    inp_Family_Status = st.radio(
        'Single/Married:',
        np.unique(my_data['Family_Status']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Income_Type = st.radio(
        'Type of Income',
        np.unique(my_data['Income_Type']))   

left_column, right_column = st.columns(2)
with left_column:
    inp_House_Type = st.radio(
        'Type of House',
        np.unique(my_data['House_Type']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Education = st.radio(
        'Education of the Applicant',
        np.unique(my_data['Education']))  

input_Children = st.slider('Number of Children:', 0, max(my_data["Children"]), 2)
input_Family_Size = st.slider('Family Size:', 0, max(my_data["Family_Size"]), 3)
input_Salary = st.slider('Salary of the Applicant:', 0.0, max(my_data["Salary"]), 10000.0)
input_AGE = st.slider('Age in Years:', 0.0, max(my_data["AGE"]), 22.0)
input_EXPERIENCE = st.slider('Experience in Years:', 0.0, max(my_data["EXPERIENCE"]), 3.0)
input_ACCOUNT_DURATION = st.slider('Account Duration in Months:', 0, max(my_data["ACCOUNT_DURATION"]), 5)


X = my_data.drop(['ID', 'Risk', 'Occupation_Type'], axis=1)
y = my_data['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#smote = ADASYN()
#X_train,y_train = smote.fit_resample(X_train,y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_xgboost_model = XGBClassifier(max_depth=5,n_estimators=250, min_child_weight=8)
best_xgboost_model.fit(X_train, y_train)

if st.button('Make Prediction'):
    inputs = [inp_Gender, inp_Car, inp_Realty, input_Children, input_Salary, inp_Income_Type, inp_Education, inp_Family_Status, inp_House_Type, input_AGE, input_EXPERIENCE, input_Family_Size, input_ACCOUNT_DURATION]
    prediction = best_xgboost_model.predict(inputs)
    if prediction:
        st.write("Your Credit Card is Declined")
    else:
        st.write("Your Credit Card is Approved")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")