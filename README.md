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
