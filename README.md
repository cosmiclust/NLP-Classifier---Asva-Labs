# NLP-Classifier-Asva-Labs
Problem Statement
The goal is to build a text classification system that takes a support ticket and predicts whether it is related to Billing, Technical, or falls into the Other category.

Dataset
File: data/tickets.csv

Format: CSV with two columns — text (support ticket message) and label (Billing/Technical/Other)

Size: 300 examples (100 per class — balanced for fair training)

Nature: Synthetic examples generated to simulate real-world support tickets

Model and Training
Model Used: DistilBERT (distilbert-base-uncased) from Hugging Face Transformers

Framework: PyTorch (via Hugging Face Trainer)

Tokenizer: DistilBertTokenizerFast

Data Split: 80% train / 20% validation

Label Encoding: sklearn.LabelEncoder

Metrics: Accuracy

Training Epochs: 3

Batch Size: 8

Output: Model artifacts saved in model/ directory

To train the model:

bash
Copy
Edit
python model/train.py
Inference API
A REST API is built using FastAPI. It exposes a /predict endpoint that accepts plain text and returns a predicted label along with the confidence score.

Start the API:

bash
Copy
Edit
uvicorn api.main:app --reload
Then open your browser and go to:

arduino
Copy
Edit
http://127.0.0.1:8000/docs
You can test the /predict endpoint using Swagger UI.

Sample Prediction
Request:

json
Copy
Edit
{
  "text": "I'm facing issues with my invoice payment"
}
Response:

json
Copy
Edit
{
  "label": "Billing",
  "confidence": 0.91
}
Model Artifacts
Due to GitHub's file size limits, model.safetensors is hosted externally.

Download it from this Google Drive link:
Download model.safetensors

Place it in your local model/ directory.

How to Run the Project
Clone the repository

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Train the model:

bash
Copy
Edit
python model/train.py
Start the API server:

bash
Copy
Edit
uvicorn api.main:app --reload
Open your browser at http://127.0.0.1:8000/docs to test the prediction endpoint.


