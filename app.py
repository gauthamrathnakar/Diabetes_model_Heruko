import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

#creating function for prediction

def diabetes_prediction(input_data):
    sc = StandardScaler()
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = sc.fit_transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)
    if prediction[0] == 0:
        return "Person is non-diabetic"
    else:
        return "Person is diabetic"
    
def main():

    #giving title
    st.title("Diabetes prediction web app")

    #getting input data from user
    Pregnencies = st.text_input("Enter number of pregnencies")
    Glucose = st.text_input("Enter the glucose level")
    BloodPressure = st.text_input("Enter BloodPressure")
    SkinThickness = st.text_input("Enter SkinThickness")
    Insulin = st.text_input("Enter Insulin")
    BMI = st.text_input("Enter BMI")
    DiabetesPedigreeFunction = st.text_input("Enter DiabetesPedigreeFunction")
    Age = st.text_input("Enter Age")

    #Code for prediction
    diagnosis = ''

    #Creating button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnencies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
