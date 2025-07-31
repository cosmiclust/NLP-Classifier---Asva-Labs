import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

#  1. Load and validate data 
df = pd.read_csv("data/tickets.csv")

# Drop rows with missing values
df.dropna(subset=["text", "label"], inplace=True)

# Basic class distribution check (for logging/debugging)
print("Class distribution:\n", df["label"].value_counts())

# 2. Encode labels
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

# Save label mapping
os.makedirs("model", exist_ok=True)
label_map = {cls: int(idx) for cls, idx in zip(le.classes_, le.transform(le.classes_))}
with open("model/label_map.json", "w") as f:
    json.dump(label_map, f)

#  3. Train-validation split 
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label_id"], test_size=0.2, random_state=42, stratify=df["label_id"]
)

# 4. Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

#  5. Prepare PyTorch datasets 
class TicketDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = TicketDataset(train_encodings, train_labels.tolist())
val_dataset = TicketDataset(val_encodings, val_labels.tolist())

#  6. Load model 
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

# 7. Training config 
training_args = TrainingArguments(
    output_dir="model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 8. Metrics 
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 9. Train 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

#  10. Save final model + tokenizer 
model.save_pretrained("model")
tokenizer.save_pretrained("model")
