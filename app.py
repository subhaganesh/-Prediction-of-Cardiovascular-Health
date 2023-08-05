import numpy as np
import pickle
import streamlit as st
import altair as alt
import altair.vegalite.v4.api

#loading the saved model
model = pickle.load(open("heart_disease_model.sav", 'rb'))

# creating a function for Prediction

def heart_disesse_features(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The Person does not have a Heart Disease'
    else:
      return 'The Person has Heart Disease'
    

def main():
    # Giving a title
    st.title('heart disease Prediction Web App')
    
    # Getting the input data from the user
    Age = st.number_input('age of person', min_value=0, max_value=100, step=1)
    sex = st.number_input('sex', min_value=0, max_value=1, step=1)
    cp = st.number_input('cp', min_value=0, max_value=3, step=1)
    trestbps = st.number_input('trestbps', min_value=80, max_value=200, step=1)
    chol = st.number_input('chol', min_value=100, max_value=600, step=1)
    fbs= st.number_input('fbs', min_value=0, max_value=1)
    restecg = st.number_input('restecg', min_value=0, max_value=2,step=1)
    thalach= st.number_input('thalach', min_value=70, max_value=220, step=1)
    exang= st.number_input('exang', min_value=0, max_value=1, step=1)
    oldpeak= st.number_input('oldpeak', min_value=1.0, max_value=5.5, step=1)
    slope= st.number_input('slope', min_value=0, max_value=2, step=1)
    ca= st.number_input('ca', min_value=0, max_value=4, step=1)
    thal= st.number_input('thal', min_value=0, max_value=3, step=1)

    # Code for Prediction
    diagnosis = ''
    
    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis =heart_disesse_features ([Age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()

#to run this app in browser we want to type streamlit run C:\Users\subha\heart disease prediction\app.py in command prompt
