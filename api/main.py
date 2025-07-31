from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import json

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("model")
tokenizer = DistilBertTokenizerFast.from_pretrained("model")
model.eval()  # set to eval mode for inference

# Load label map (reverse: id -> label)
with open("model/label_map.json") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}

# Define FastAPI app
app = FastAPI()

# Request schema
class TicketRequest(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict(req: TicketRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()

    return {
        "label": id2label[pred_id],
        "confidence": round(confidence, 4)
    }

# Root endpoint (for browser tests)
@app.get("/")
def root():
    return {"message": "NLP Classifier API is up and running!"}
