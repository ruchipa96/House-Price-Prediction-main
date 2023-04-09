# -*- coding: utf-8 -*-
"""app.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cq2MVS1NqjJ1EIW_wS9x3Qs-2uZ59kUH
"""

import streamlit as st
import numpy as np
import pickle
import pandas as pd
from house_price_prediction_file import predict_house_price


df = pd.read_csv('./train_new.csv')

with open('./my_model.pickle', 'rb') as f:
    model = pickle.load(f)




OverallCond = st.text_input("Enter Overall condition rating (between 1-5)")
# print the user's name
#st.write("OverallCond :", OverallCond)

TotalBsmtSF = st.text_input("Enter Total Basement Surface Area")
# print the user's name
#st.write("TotalBsmtSF :", TotalBsmtSF)

FirstFlrSF = st.text_input("Enter  First Floor Surface Area")
# print the user's name
#st.write("FirstFlrSF :", FirstFlrSF)

SecondFlrSF = st.text_input("Enter Second Floor Surface Area")
# print the user's name
#st.write("SecondFlrSF :", SecondFlrSF)

GrLivArea = st.text_input("Enter Living Room Surface Area")
# print the user's name
#st.write("GrLivArea :", GrLivArea)

FullBath = st.text_input("Enter number of Full Baths")
# print the user's name
#st.write("FullBath :", FullBath)

BedroomAbvGr = st.text_input("Enter number of Bedrooms")
# print the user's name
#st.write("BedroomAbvGr :", BedroomAbvGr)


user_inputs_app = [OverallCond, TotalBsmtSF, FirstFlrSF, SecondFlrSF, GrLivArea, FullBath, BedroomAbvGr]

if st.button('Predict House Price'):
    predict_house_price(user_inputs_app, model, df)


