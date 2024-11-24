##Create New Folder
##Copy pickle file and data set
##Web app development using streamlit
##Load the necessary libraries
##Try to install by streamlit

import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title ("Promotion Prediction APP")

## Read the dataset to fill the values in input options of each element

df = pd.read_csv('train_hr.csv')

## Create the input elements

## Categorical Columns

department = st.selectbox("Department",pd.unique(df['department']))
region = st.selectbox("Region",pd.unique(df['region']))
education = st.selectbox("Education",pd.unique(df['education']))
gender = st.selectbox("Gender",pd.unique(df['gender']))
recruitment_channel = st.selectbox("Recruitment_Channel",pd.unique(df['recruitment_channel']))

## Non Categorial Names
no_of_trainings = st.number_input("No_of_trainings")
age = st.number_input("Age")
previous_year_rating = st.number_input("Previous_year_rating   ")
length_of_service = st.number_input("Length_of_service ")
KPIs_met_80 = st.number_input("KPIs_met > 80%")
awards_won = st.number_input("Awards_won")
avg_training_score = st.number_input("Avg_training_score")

## Map the user inputs to respective column format

inputs = {
"department": department,
"region": region,
"education": education,
"gender": gender,
"recruitment_channel": recruitment_channel,
"no_of_trainings": no_of_trainings,
"age": age,
"previous_year_rating": previous_year_rating,
"length_of_service": length_of_service,
"KPIs_met >80%": KPIs_met_80,
"awards_won?": awards_won,
"avg_training_score": avg_training_score
}

## Load the model from the Pickle files

model = joblib.load('promote_pipeline_model.pkl')

## Action for Submit Button

if st.button('Predict'):
    X_input = pd.DataFrame(inputs,index=[0])
    prediction = model.predict(X_input)
    st.write("The Predicted Value is: ")
    st.write(prediction)





