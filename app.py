import numpy as np
import pickle
import streamlit as st
import altair as alt
from altair.vegalite.v4.api import Chart

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
    

def heart_disease_prediction():
    st.title("Heart Disease Prediction")
    
    # Input data and their corresponding min-max values
    input_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                      'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    feature_ranges = {
        'age': (20, 80),
        'sex': (0, 1),
        'cp': (0, 3),
        'trestbps': (80, 200),
        'chol': (100, 600),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'thalach': (70, 220),
        'exang': (0, 1),
        'oldpeak': (0, 6),
        'slope': (0, 2),
        'ca': (0, 4),
        'thal': (0, 3)
    }
    
    st.sidebar.header("Input Values")
    
    # Display input sliders for each feature
    input_values = {}
    for feature in input_features:
        min_val, max_val = feature_ranges[feature]
        input_values[feature] = st.sidebar.slider(f"{feature.capitalize()} ({min_val} - {max_val})", min_val, max_val)
    
    # Add a button to trigger the prediction
    if st.sidebar.button("Predict"):
        # Here you can add your heart disease prediction logic using the input values
        # For demonstration purposes, let's just display the input values
        st.subheader("Input Values")
        for feature, value in input_values.items():
            st.write(f"{feature.capitalize()}: {value}")

if __name__ == '__main__':
    heart_disease_prediction()

#to run this app in browser we want to type streamlit run C:\Users\subha\heart disease prediction\app.py in command prompt
