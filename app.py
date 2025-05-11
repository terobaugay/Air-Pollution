import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
# load model
model = joblib.load('model_air_pollution.pkl')

# Fucntion to predict clusters
def predict_clusters(data):
    data['clusters']=model.predict(data)
    return data

# Streamlit main app
st.title("Air Pollution Level Detection")
file = st.file_uploader ("upload your data in csv format",type = ['csv'])
try:
    if file is not None:
        data = pd.read_csv(file)
        st.write("Here is the prediction")
        df = predict_clusters(data)
        st.write(df)
    else:
        st.write("Please enter a valid file.")
except Exception as e:
    print(f"Error {e} occured")


