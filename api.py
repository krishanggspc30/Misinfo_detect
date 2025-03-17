from fastapi import FastAPI
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the trained model
model_path = "misinformation_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# FastAPI App
app = FastAPI()

# Prediction function
@app.post("/predict/")
async def predict_misinformation(post: dict):
    text = post["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    label = "Misinformation" if prediction == 1 else "Reliable"
    return {"prediction": label}
