import gradio as gr
import pandas as pd
import pickle
import numpy as np

#load the model
with open('diabetes_model.pkl','rb') as file:
    model = pickle.load(file)
    
#main logic
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
       BMI, DiabetesPedigreeFunction, Age):
    
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    input_df = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
       BMI, DiabetesPedigreeFunction, Age]],columns = columns)
    
    
    
    prediction = model.predict(input_df)[0]
    
    if prediction ==1:
        return 'Diabetes Detected'
    else:
        return 'No Diabetes Detected'
    
inputs = [
    gr.Slider(0,10,step=1,value= 2,label = 'Pregnancies',info="Number of times the woman has been pregnant (0 or more)."),
    gr.Number(value = 120,label = 'Glucose Level',info="Blood sugar level after fasting (normal: 70–140 mg/dL)."),
    gr.Number(value = 70, label = 'BloodPressure',info="Diastolic blood pressure (normal: around 70–80 mm Hg)."),
    gr.Number(value = 20, label = 'SkinThickness',info="Thickness of skin fold (Typical value is around 20)."),
    gr.Number(value = 80, label = 'Insulin',info="Average level is about 80. Many people have values between 0 and 127."),
    gr.Number(value = 32, label = 'BMI',info="Body Mass Index (normal: 18.5–24.9)."),
    gr.Number(value = 0.5, label = 'Diabetes Pedigree Function',info="Family history of diabetes (Higher values mean stronger family history(risk))."),
    gr.Number(value = 30, label = 'Age',info="Age in years.")
]


#interface
app = gr.Interface(
    fn = predict_diabetes,
    inputs = inputs,
    outputs='text',
    title='Diabetes Prediction Using ML',
    description="Enter patient medical information to predict diabetes risk."
)

#launch
app.launch()