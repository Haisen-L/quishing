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

import timm
from torchvision import transforms

#########################
# Global Configurations
#########################
start_time = time.time()
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else sys.exit("Error: No GPU found! Exiting..."))
print(f"Using device: {device}")

#########################
# File Paths & Output Directories
#########################
CSV_FILE = "data/output_list/output_L.csv"
PRED_DIR = "data/predict"
MODEL_NAME = "TNT"
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
# Image Transformations
#########################
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#########################
# Dataset Definition (Torch)
#########################
class QRCodeDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["QR_Code_Path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row["result"], dtype=torch.long)
        return image, label

#########################
# Train/Val Split
#########################
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

#########################
# DataLoaders
#########################
batch_size = 128
train_loader = DataLoader(
    QRCodeDataset(train_df, transform), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    QRCodeDataset(val_df, transform), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

#########################
# Test Data Preparation
#########################
new_csv = 'data/output_list_new/new_output_L.csv'
new_df = pd.read_csv(new_csv)
new_df['result'] = new_df['result'].astype(int)
new_df = new_df[new_df['QR_Code_Path'].notnull()]
print(f"New test set size: {len(new_df)}")

test_loader = DataLoader(
    QRCodeDataset(new_df, transform), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

#########################
# Model Definition (tnt_b_patch16_224, pretrained=False)
#########################
model = timm.create_model(
    "tnt_b_patch16_224", pretrained=False, num_classes=2
).to(device)

#########################
# Optimizer & Loss
#########################
optimizer = optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

#########################
# Training Function
#########################
def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=8):
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_val_loss = total_val_loss / len(val_loader)
        p = precision_score(all_labels, all_preds)
        r = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
            f"Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}"
        )

#########################
# Run Training
#########################
train(model, train_loader, val_loader, optimizer, criterion, device)

#########################
# Evaluation Function
#########################
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, labs in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            labels.extend(labs.tolist())
    return preds, labels

#########################
# Test Evaluation
#########################
preds, labs = evaluate(model, test_loader, device)
precision = precision_score(labs, preds)
recall = recall_score(labs, preds)
f1 = f1_score(labs, preds)
cm = confusion_matrix(labs, preds, labels=[0, 1])
print(f"\nTest set (n={len(labs)}): Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
plt.title("Confusion Matrix - Test on new dataset")
cm_path = os.path.join(CM_DIR, f"{FOLDER_NAME}_{MODEL_NAME}_new_test.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved CM: {cm_path}")

#########################
# Save Overall Predictions for New Test Set
#########################
new_df["predict_label"] = preds
output_file = os.path.join(MODEL_PRED_DIR, f"{MODEL_NAME}_{FOLDER_NAME}_new_v_predict.csv")
new_df.to_csv(output_file, index=False)
print(f"New test predictions saved to: {output_file}")

#########################
# Runtime Measurement
#########################
end_time = time.time()
elapsed = end_time - start_time
hrs, rem = divmod(elapsed, 3600)
mins, secs = divmod(rem, 60)
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"Total running time: {int(hrs):02}:{int(mins):02}:{int(secs):02}")
