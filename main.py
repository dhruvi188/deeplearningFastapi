import os
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import uvicorn
from typing import Optional

app = FastAPI()

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load your CNN model here
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

def convert_to_jsonable(value):
    if isinstance(value, np.int64):
        return int(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, '__dict__'):
        return vars(value)
    else:
        return str(value)

@app.post("/submit")
async def submit(image: UploadFile = File(...)):
    contents = await image.read()
    filename = image.filename
    file_path = os.path.join('static/uploads', filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    
    pred = prediction(file_path)
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    
    # Check if 'possible_steps' column exists before accessing it
    if 'possible_steps' in disease_info:
        prevent = disease_info['possible_steps'][pred]
    else:
        prevent = "Preventive steps information not available"
    
    image_url = disease_info['image_url'][pred]
    
    # Correct column names for supplement_info DataFrame
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]
    
    # Prepare response data
    response_data = {
        "title": title,
        "description": description,
        "prevent": prevent,
        "image_url": image_url,
        "supplement_name": supplement_name,
        "supplement_image_url": supplement_image_url,
        "supplement_buy_link": supplement_buy_link
    }
    
    # Convert response data to JSON-serializable format
    jsonable_response = {key: convert_to_jsonable(value) for key, value in response_data.items()}
    
    return jsonable_response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
