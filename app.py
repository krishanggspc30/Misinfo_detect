from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("misinformation_model")
tokenizer = DistilBertTokenizer.from_pretrained("misinformation_model")

# Define request body schema
class InputText(BaseModel):
    text: str

@app.post("/predict/")
async def predict(input_text: InputText):
    inputs = tokenizer(input_text.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return {"prediction": "Misinformation" if prediction == 1 else "Real News"}

