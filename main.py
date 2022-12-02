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
my_data = pd.read_csv("final_application.csv")
copy_data = my_data.copy()

X = my_data.drop(['ID', 'Risk', 'Occupation_Type'], axis=1)
y = my_data['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#adasyn = ADASYN()
#X_train,y_train = adasyn.fit_resample(X_train,y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_xgboost_model = XGBClassifier(max_depth=5,n_estimators=250, min_child_weight=8)
best_xgboost_model.fit(X_train, y_train)
pred = best_xgboost_model.predict(X_test)

def download_model(best_xgboost_model):
    output_model = pickle.dumps(best_xgboost_model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model .pkl File</a> (right-click and save as &lt;some_name&gt;.pkl)'
    st.markdown(href, unsafe_allow_html=True)
