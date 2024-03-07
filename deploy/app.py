import streamlit as st
import os
from model import RacistContentClassifier

model_path = "deploy/rf_model.joblib"
classifier = RacistContentClassifier(model_path)

st.title('Racist Content Classifier')
st.write('Moderate whether text it contains racist content or not.')

text = st.text_area("Enter text below:")

if st.button('Predict'):
    if text.strip() == "":
        st.warning("Please enter text.")
    else:
        prediction = classifier.predict(text)
        if prediction == "Racist":
            st.error('Racist')
        else:
            st.success('Not Racist')
