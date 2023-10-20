import gradio as gr
import pandas as pd
import numpy as np
import joblib

#Load scaler ,encoder and model

scaler = joblib.load(r'toolkit\scaler.joblib')
model = joblib.load(r'toolkit\model_final.joblib')
encoder = joblib.load(r'toolkit\encoder.joblib')

# Create a function that applies the ML Pipeline

def predict(Gender,Urea,Cr,HbA1c,Chol,TG,HDL,LDL,VLDL,BMI,age_range):
    
    # Convertir l'âge en un format numérique
    age_range_mapping = {
        '20-30': 1,
        '30-40': 2,
        '40-50': 3,
        '50-60': 4,
        '60-70': 5,
        '70-80': 6,
        '80-90': 7,
        '90-100': 8,
    }
    age_range_numeric = age_range_mapping.get(age_range, 0)
    #Create a dataframe
    input_df = pd.DataFrame({'Gender':[Gender],
        'Urea':[Urea],
        'Cr':[Cr],
        'HbA1c':[HbA1c],
        'Chol':[Chol],
        'TG':[TG],
        'HDL':[HDL],
        'LDL':[LDL],
        'VLDL':[VLDL],
        'BMI':[BMI],
        'age_range':[age_range_numeric]
        })
    
    input_df['Urea'] = input_df['Urea'].astype(float)  # Convert  "Urea" in float
    input_df['Gender']=encoder.fit_transform(input_df['Gender'])
    # input_df['CLASS']=encoder.fit_transform(input_df['CLASS'])
    input_df['age_range']=encoder.fit_transform(input_df['age_range'])
    final_df = input_df
    # print("Gender (encoded):", input_df['Gender'])
    # print("Age Range (encoded):", input_df['age_range'])
    #Make prediction using model
    prediction = model.predict(final_df)
    
    #prediction label 
    prediction_label = {
        0: "No Diabetic",
        1: "Predicted Diabetic",
        2: "Diabetic"
    }
    # print(prediction_label)
    print("Prediction (numeric value):", prediction[0])
    print("Prediction (label):", prediction_label[int(prediction[0])])
    return prediction_label[int(prediction[0])]

input_interface=[]

with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    img = gr.Image("diabete1.jpg")
    Title =gr.Label('Predicting Diabete App')
    with gr.Row():
        img
    with gr.Row():
        Title
    
    with gr.Row():
        gr.Markdown("This app predict of Patient to mean Diabete No Diabete or Predicted Diabete")
        
    with gr.Accordion("Open for information on input"):
        gr.Markdown(""" This app receives the following as inputs and processes them return the prediction on whether a patient given the input will diabeted ,no diabete or predicted. 
                        - Gender
                        - Urea: Urea
                        - Cr : Creatinine ratio(Cr)
                        - HbA1c:Sugar Level Blood
                        - Chol : Cholesterol (Chol)
                        - TG :Triglycerides(TG)
                        - HDL :  HDL Cholesterol
                        - LDL:LDL
                        - VLDL:VLDL
                        - BMI :Body Mass Index (BMI)
                        - CLASS : Class (the patient's diabetes disease class may be Diabetic, Non-Diabetic, or Predict-Diabetic)
                        - AGE :Age
                    """)  
    with gr.Row():
        with gr.Column():
            input_interface_column_1 = [
                gr.components.Radio(['M','F'],label='Select your gender'),
                gr.components.Dropdown([1 ,2,3,4,5,6,7,8],label='Choose age tranche 1=>(20-30),2=>(30-40),3=>(40-50)...'),
                gr.components.Slider(label='Level urea mg/dl',minimum=0, maximum=100),
                gr.components.Number(label='Level Creatine mg/dl',minimum=0, maximum=100),
                gr.components.Number(label='Sugar Level Blood',minimum=0, maximum=100),
            ]
        with gr.Column():
            input_interface_column_2 = [
                gr.components.Number(label='Level Cholesterol mg/dl',minimum=0, maximum=100),
                gr.components.Number(label='Level Triglycerides mg/dl',minimum=0, maximum=100),
                gr.components.Slider(label='HDL Cholesterol',minimum=0, maximum=100),
                gr.components.Number(label='level LDL',minimum=0, maximum=100),
                gr.components.Number(label='VLDL level',minimum=0, maximum=10),
                gr.components.Slider(label='BMI Body Mass Index',minimum=0, maximum=100),
            ]
    with gr.Row():
        input_interface.extend(input_interface_column_1)
        input_interface.extend(input_interface_column_2)
        
    with gr.Row():
        predict_btn = gr.Button('Predict',variant="primary")
    
    #define the output interface
    output_interface = gr.Label(label="Diabetes Status")

    predict_btn.click(fn=predict, inputs=input_interface, outputs=output_interface)

app.launch()