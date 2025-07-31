# NLP-Classifier-Asva-Labs

---

## Problem Statement

The goal is to build a text classification system that takes a support ticket and predicts whether it is related to **Billing**, **Technical**, or falls into the **Other** category.

---

## Dataset

- File: `data/tickets.csv`  
- Format: CSV with two columns — `text` (support ticket message) and `label` (Billing/Technical/Other)  
- Size: 150 examples (50 per class — balanced for fair training)  
- Nature: Synthetic examples generated to simulate real-world support tickets  

---

## Model and Training

- **Model Used:** DistilBERT (`distilbert-base-uncased`) from Hugging Face Transformers  
- **Framework:** PyTorch (via Hugging Face Trainer)  
- **Tokenizer:** DistilBertTokenizerFast  
- **Data Split:** 80% train / 20% validation  
- **Label Encoding:** `sklearn.LabelEncoder`  
- **Metrics:** Accuracy  
- **Training Epochs:** 3  
- **Batch Size:** 8  
- **Output:** Model artifacts saved in `model/` directory  

To train the model:
```bash
python model/train.py

## **Inference API**

- **Framework:** FastAPI  
- **Endpoint:** `POST /predict`  
- **Input:** JSON with a single field `text` containing the support ticket message  
- **Output:** JSON with predicted `label` and `confidence` score  

### **Example Request
**
```json
POST /predict
{
  "text": "I'm unable to log into my dashboard after the recent update."
}

{
  "label": "Technical",
  "confidence": 0.87
}
## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/your-username/NLPClassifier.git
cd NLPClassifier
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, install manually:

```bash
pip install pandas scikit-learn torch transformers fastapi uvicorn
```

---

## Dataset

- **File:** `data/tickets.csv`
- **Format:** CSV with two columns — `text` (support ticket message) and `label` (Billing/Technical/Other)
- **Size:** 150 examples (50 per class — balanced for fair training)
- **Nature:** Synthetic examples generated to simulate real-world support tickets

---

## Model and Training

- **Model Used:** DistilBERT (`distilbert-base-uncased`)
- **Framework:** PyTorch (via Hugging Face `Trainer`)
- **Tokenizer:** DistilBertTokenizerFast
- **Data Split:** 80% train / 20% validation
- **Label Encoding:** `sklearn.LabelEncoder`
- **Metric:** Accuracy
- **Training Epochs:** 3
- **Batch Size:** 8
- **Output Artifacts:** Saved under `model/` directory

### Train the Model

```bash
python model/train.py
```

---

## Run the Inference API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

Once running, open:

```
http://127.0.0.1:8000/docs
```

to access the Swagger UI.

---

## Example Inference

### Request

```json
POST /predict
Content-Type: application/json

{
  "text": "My internet is not working after payment"
}
```

### Response

```json
{
  "label": "Technical",
  "confidence": 0.87
}
```

---

## File Structure

```
NLPClassifier/
├── data/
│   └── tickets.csv
├── model/
│   ├── train.py
│   ├── config.json
│   ├── label_map.json
│   ├── vocab.txt
│   └── model.safetensors (Google Drive, >25MB)
├── api/
│   └── main.py
├── requirements.txt
└── README.md
```

---
