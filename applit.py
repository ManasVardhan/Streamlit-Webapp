import seaborn as sns
import streamlit.components.v1 as components
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


def pred(df):
    features = [f for f in df.columns if f not in ['Unnamed: 0',
                                                   'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    lgb = joblib.load('home_credit_model.pkl')
    y_pred = lgb.predict(df[features])
    df["TARGET"] = y_pred

    st.write(df['TARGET'])
    st.write(df.shape)
    st.write(y_pred)
    # st.markdown(y_pred)


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
        pred(dataframe)
    except Exception as e:
        st.write(e)


st.title('Feature Importance')

st.write('''
Feature importances, computed as a decrease in score when feature
values are permuted (i.e. become noise). This is also known as 
permutation importance.

If feature importances are computed on the same data as used for training, 
they don't reflect importance of features for generalization. Use a held-out
dataset if you want generalization feature importances.
''')
features = pd.read_csv('feature_importance.csv', encoding='unicode_escape')
st.write(features)
