import streamlit as st
from PIL import Image
import pickle


model = pickle.load(open('../pickle-file/ML_Model_Diabetics.pkl', 'rb'))

def run():
    img1 = Image.open('bank.png')
    img1 = img1.resize((156,145))
    st.image(img1,use_column_width=False)
    st.title("Diabetics Classifcation using Machine Learning")

    ## Pregnancies
    Pregnancies = st.number_input('Number of Pregnancies', value=0)

    ## Glucose
    Glucose = st.number_input('Glucose Level', value=0)

    ## BloodPressure
    BloodPressure = st.number_input('Blood Pressure', value=0)

    ## SkinThicness
    SkinThickness = st.number_input('Skin Thickness', value=0)    

    ## Insulin
    Insulin = st.number_input('Insulin Level', value=0)

    ## BMI
    BMI = st.number_input('BMI Value', value=0.0)

    ## DiabetesPedigreeFunction
    DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', value=0.0)

    ## Age
    Age = st.number_input('Age', value=0)

    if st.button("Submit"):

        features = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]] 
 
        print(features)
        prediction = model.predict(features)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))
        if ans == 0:
            st.error(
                 'According to ML prediction, the Patient does not have Diabetics'
            )
        else:
            st.success(
                 'According to ML Prediction, the Patient has Diabetics'
            )

run()