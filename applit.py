from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, url_for, redirect, render_template
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
import sklearn
import flask
warnings.filterwarnings("ignore")


lgb = joblib.load('home_credit_model.pkl')


def pred(df):
    '''
    Get prediction results
    '''

    features = [f for f in df.columns if f not in ['Unnamed: 0',
                                                   'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Loading the classifier

    lgb = joblib.load('home_credit_model.pkl')

    y_pred = lgb.predict(df[features])
    df["TARGET"] = y_pred

    st.write(df['TARGET'])
    st.write(df.shape)


st.title('Finance web app')
uploaded_file = st.file_uploader(label='Upload csv file', type=[
                                 'csv'], accept_multiple_files=False)


if uploaded_file is not None:
    try:
        dataframe = pd.read_csv(uploaded_file)
        print(dataframe)
        st.write(dataframe.head())
        st.write(dataframe.shape)
        pred(dataframe)
    except Exception as e:
        st.write(e)
