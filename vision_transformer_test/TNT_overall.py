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

# Additional imports for model and image preprocessing
import timm
from torchvision import transforms


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

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


#########################
# Dataset Definition
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
# Model Loading
#########################
device = torch.device("cuda" if torch.cuda.is_available() else sys.exit("Error: No GPU found! Exiting..."))
print(f"Using device: {device}")

# Load TNT-B model
tnt_model = timm.create_model("tnt_b_patch16_224", pretrained=False)
tnt_model.to(device)
tnt_model.eval()


#########################
# Train/Val/Test Split
#########################
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


#########################
# Dataloaders
#########################
train_loader = DataLoader(QRCodeDataset(train_df, preprocess), batch_size=128, shuffle=True)
val_loader = DataLoader(QRCodeDataset(val_df, preprocess), batch_size=128, shuffle=False)


#########################
# Classifier Definition
#########################
class TNTClassifier(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        # Infer feature dimension from model
        num_features = backbone.num_features
        self.fc = nn.Linear(num_features, num_classes)
    def forward(self, x):
        # Extract features
        feats = self.backbone.forward_features(x)
        # Class token is first
        cls_token = feats[:, 0]
        return self.fc(cls_token)


#########################
# Initialize Model, Optimizer & Loss
#########################
model = TNTClassifier(tnt_model).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
params = model.module.fc.parameters() if isinstance(model, nn.DataParallel) else model.fc.parameters()
optimizer = optim.AdamW(params, lr=0.002)
criterion = nn.CrossEntropyLoss()


#########################
# Training Function
#########################
def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=8):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# Run training
train(model, train_loader, val_loader, optimizer, criterion, device)


#########################
# Evaluation Function
#########################
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, labs in loader:
            images, labs = images.to(device), labs.to(device)
            out = model(images)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels.extend(labs.cpu().numpy())
    return preds, labels


#########################
# Test Evaluation (Overall)
#########################
preds, labs = evaluate(model, DataLoader(QRCodeDataset(test_df, preprocess), batch_size=128), device)
precision = precision_score(labs, preds)
recall = recall_score(labs, preds)
f1 = f1_score(labs, preds)
cm = confusion_matrix(labs, preds, labels=[0,1])
print(f"Test set (n={len(labs)}): Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign","Malicious"], yticklabels=["Benign","Malicious"])
plt.title("Confusion Matrix - Overall Test")
cm_path = os.path.join(CM_DIR, f"{FOLDER_NAME}_tnt_b_overall_test.png")
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
