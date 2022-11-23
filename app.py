import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_survival_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_survival_model.pkl")

def titanic(pclass, sex, age, fare, embarked, title, isalone):
    input_list = [] 
    input_list.append(int(pclass+1))
    input_list.append(int(sex)) 
    if age<=16:
      input_list.append(0)
    elif age>16 and age<=32:
      input_list.append(1)
    elif age>32 and age<=48:
      input_list.append(2)
    elif age>48 and age<=64:
      input_list.append(3)
    else:
      input_list.append(4)
    if fare<=7.91:
      input_list.append(0)
    elif fare>7.91 and fare<=14.454:
      input_list.append(1)
    elif fare>14.454 and fare<=31:
      input_list.append(2)
    else:
      input_list.append(3)
    if embarked=='C':
      input_list.append(1)
    elif embarked=='S':
      input_list.append(0) 
    else:
      input_list.append(2) 
    input_list.append(title)
    input_list.append(isalone)
    res = model.predict(np.asarray(input_list,dtype=object).reshape(1,-1))
    if res[0] == 0: #ded
      person="dead"
      passenger_url = "https://raw.githubusercontent.com/irena123333/titanic-prediction/main/dead.png"
      
    else:
      person="survived"
      passenger_url = "https://raw.githubusercontent.com/irena123333/titanic-prediction/main/survived.png"
    img = Image.open(requests.get(passenger_url, stream=True).raw)            
    return img
      
demo = gr.Interface(
    fn=titanic,
    title="Titanic Passenger Survival Predictive Analytics",
    description="If one person is on titanic, predict whether he or she will survive.",
    allow_flagging="never",
    inputs=[gr.inputs.Dropdown(choices=["Class 1","Class 2","Class 3"], type="index", label="pclass"),
        gr.inputs.Dropdown(choices=["Male", "Female"], type="index", label="sex"),
        gr.inputs.Slider(0,150,label='Age'),
        gr.inputs.Number(default=8.0, label="Fare"),
        gr.inputs.Radio(default='S', label="Embarkation Port", choices=['C', 'Q', 'S']),
        gr.inputs.Dropdown(choices=["Master","Miss","Mr","Mrs","Other"], type="index", label="Title"),
        gr.inputs.Dropdown(choices=["False", "True"], type="index", label="IsAlone"),
        ],
        
    outputs=gr.Image(type="pil"))

demo.launch(share=True)