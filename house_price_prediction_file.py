import streamlit as st
from sklearn.preprocessing import LabelEncoder
import numpy as np

def predict_house_price(user_inputs_app, model, df):
    features = ['OverallCond', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr']
    # features = df.columns
    # create dictionary for user input
    user_input = dict(zip(features, user_inputs_app))
    

    # convertng the features not in the list to float as we did not convert them at the input
    for feature in features:
        if feature in ['OverallCond', 'FullBath', 'BedroomAbvGr']:
            user_input[feature] = int(user_input[feature])
        else:
            user_input[feature] = float(user_input[feature])    
    
    #st.write("user_input :", user_input)pr

    # create label encoder for categorical features
    encoder = LabelEncoder()
    for feature in ['OverallCond', 'FullBath', 'BedroomAbvGr']:
        encoder.fit(np.unique(df[feature]))
        user_input[feature] = encoder.transform([user_input[feature]])[0]

    # convert user input to numpy array
    user_input_array = np.array(list(user_input.values())).reshape(1, -1)

    # make predictions on user input using the loaded model
    predicted_sale_price = model.predict(user_input_array)[0]

    # print(f"Predicted SalePrice: ${predicted_sale_price:.2f}")

    st.write("Predicted Sales Price :", predicted_sale_price)

