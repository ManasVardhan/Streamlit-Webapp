from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")


lgb = joblib.load('home_credit_model.pkl')

# def predict():
#     int_features = [int(x) for x in request.form.values()]
#     final = [np.array(int_features)]
#     print(int_features)
#     print(final)
#     prediction = model.predict_proba(final)
#     output = '{0:.{1}f}'.format(prediction[0][1], 2)

#     if output > str(0.5):
#         return render_template('forest_fire.html', pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output), bhai="kuch karna hain iska ab?")
#     else:
#         return render_template('forest_fire.html', pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output), bhai="Your Forest is Safe for now")


def pred(df):
    features = [f for f in df.columns if f not in ['Unnamed: 0',
                                                   'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    lgb = joblib.load('home_credit_model.pkl')
    y_pred = lgb.predict(df[features])
    df["TARGET"] = y_pred

    st.write(df['TARGET'])
    st.write(df.shape)
    st.write(y_pred)


st.title('Finance web app')
uploaded_file = st.file_uploader(label='Upload csv file', type=[
                                 'csv'], accept_multiple_files=False)

if uploaded_file is not None:
    try:
        dataframe = pd.read_csv(uploaded_file)
        print(dataframe)
        # st.write(dataframe)
        # dataframe.dropna(inplace=True)
        st.write(dataframe.head())
        st.write(dataframe.shape)
        # pred(dataframe)
    except Exception as e:
        st.write(e)
