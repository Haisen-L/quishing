#########################
# Imports
#########################
import os
import sys
import time
import collections

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

from transformers import AutoImageProcessor, AutoModelForImageClassification

#########################
# Global Configurations
#########################
start_time = time.time()
torch.backends.cudnn.benchmark = True

#########################
# File Paths & Output Directories
#########################
CSV_FILE = "data/output_list/output_L.csv"
PRED_DIR = "data/predict"
MODEL_NAME = "microsoft_cvt_21"
FOLDER_NAME = "output_L"

if not os.path.exists(PRED_DIR):
    os.makedirs(PRED_DIR)
MODEL_PRED_DIR = os.path.join(PRED_DIR, MODEL_NAME)
if not os.path.exists(MODEL_PRED_DIR):
    os.makedirs(MODEL_PRED_DIR)
CM_DIR = os.path.join("cm", MODEL_NAME)
if not os.path.exists(CM_DIR):
    os.makedirs(CM_DIR)

start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print(f"Start time: {start_time_readable}")

#########################
# Data Preparation
#########################
df = pd.read_csv(CSV_FILE)
df["result"] = df["result"].astype(int)
print("Label distribution in CSV:", collections.Counter(df["result"]))

#########################
# Dataset Definition (HF)
#########################
class QRCodeHFDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["QR_Code_Path"]).convert("RGB")
        label = torch.tensor(row["result"], dtype=torch.long)
        return image, label

#########################
# Collate Function
#########################
def hf_collate(batch):
    images, labels = zip(*batch)
    labels = torch.stack(labels)
    return list(images), labels

#########################
# Train/Val/Test Split
#########################
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

#########################
# DataLoaders
#########################
batch_size = 128
train_loader = DataLoader(QRCodeHFDataset(train_df), batch_size=batch_size, shuffle=True, collate_fn=hf_collate)
val_loader   = DataLoader(QRCodeHFDataset(val_df),   batch_size=batch_size, shuffle=False, collate_fn=hf_collate)
test_loader  = DataLoader(QRCodeHFDataset(test_df),  batch_size=batch_size, shuffle=False, collate_fn=hf_collate)

#########################
# Model Loading
#########################
device = torch.device("cuda" if torch.cuda.is_available() else sys.exit("Error: No GPU found! Exiting..."))
print(f"Using device: {device}")

image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-21", use_fast=False)
model_hf = AutoModelForImageClassification.from_pretrained(
    "microsoft/cvt-21",
    num_labels=2,
    ignore_mismatched_sizes=True
).to(device)
model_hf.eval()

#########################
# Training Function
#########################
optimizer = optim.AdamW(model_hf.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

def train_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs=8):
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            enc = image_processor(images=images, return_tensors="pt").pixel_values.to(device)
            labels = labels.to(device)
            outputs = model(enc, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                enc = image_processor(images=images, return_tensors="pt").pixel_values.to(device)
                logits = model(enc).logits.cpu()
                preds = torch.argmax(logits, dim=1).tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
        p = precision_score(all_labels, all_preds)
        r = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")

# Run training
train_loop(model_hf, train_loader, val_loader, optimizer, criterion, device)

#########################
# Evaluation Function
#########################
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, labs in loader:
            enc = image_processor(images=images, return_tensors="pt").pixel_values.to(device)
            logits = model(enc).logits.cpu()
            preds.extend(torch.argmax(logits, dim=1).tolist())
            labels.extend(labs.tolist())
    return preds, labels

#########################
# Test Evaluation (Overall)
#########################
preds, labs = evaluate(model_hf, test_loader, device)
precision = precision_score(labs, preds)
recall = recall_score(labs, preds)
f1 = f1_score(labs, preds)
cm = confusion_matrix(labs, preds, labels=[0,1])
print(f"Test set (n={len(labs)}): Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign","Malicious"], yticklabels=["Benign","Malicious"])
plt.title("Confusion Matrix - Overall Test")
cm_path = os.path.join(CM_DIR, f"{FOLDER_NAME}_{MODEL_NAME}_overall_test.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved CM: {cm_path}")

#########################
# Save Overall Predictions
#########################
test_df["predict_label"] = preds
output_file = os.path.join(MODEL_PRED_DIR, f"{MODEL_NAME}_{FOLDER_NAME}_overall_v_predict.csv")
test_df.to_csv(output_file, index=False)
print(f"Overall predictions saved to: {output_file}")

#########################
# Runtime Measurement
#########################
end_time = time.time()
elapsed = end_time - start_time
hrs, rem = divmod(elapsed, 3600)
mins, secs = divmod(rem, 60)
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"Total running time: {int(hrs):02}:{int(mins):02}:{int(secs):02}")