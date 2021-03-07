import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm
import datetime
import gc
from PIL import Image
import time
import joblib

image = Image.open('customer-loyalty1.png')
st.image(image)

st.write("""
# Customer Loyalty Prediction App
This app predicts the **Customer Loyalty Score**!
(Data obtained from [kaggle](https://www.kaggle.com/c/elo-merchant-category-recommendation/data)) .
""")

train = pd.read_csv('train.csv')

card_id = st.selectbox(label = 'Card id', options = train['card_id'][:50])
date = pd.to_datetime(st.selectbox(label = 'First active month', options = train['first_active_month'][:50]))
f1 = st.selectbox(label = 'Feature 1', options= [1,2,3,4,5])
f2 = st.selectbox(label = 'Feature 2', options= [1,2,3])
f3 = st.selectbox(label = 'Feature 3', options= [0,1])
data = {'card_id' : card_id,'first_active_month': date,'feature_1': f1,'feature_2' : f2,'feature_3' : f3}
data_df = pd.DataFrame(data, index = [0])

def final_predict_1(data):
    if data.ndim == 1:
        card_id = data['card_id']
    else:
        card_id = data['card_id'].values
    final_train = pd.read_pickle('final_train.pkl')
    if data.ndim==1:
        final_train = final_train[final_train['card_id'] == card_id]
    else:
        final_train = final_train[final_train['card_id'].isin(card_id)]
    cols = [col for col in final_train.columns if col not in ['card_id','outliers', 'target']]
    final_train = final_train[cols]
    model = pd.read_pickle('best_model.pkl')
    predictions = model.predict(final_train)
    print('Predicted loyalty score is:', predictions)
    return predictions 

if st.button('Predict'):
    with st.spinner('Predicting...'):
        #time.sleep(10)
        predictions = final_predict_1(data_df)
    st.write('**Predicted loyalty score is:**', predictions[0])