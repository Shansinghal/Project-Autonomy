import json
from pathlib import Path
import os

base_dir = Path("c:/Users/Shant/OneDrive/Desktop/Project-Autonomy/toxicity-classifier")
notebooks_dir = base_dir / "notebooks"
notebooks_dir.mkdir(parents=True, exist_ok=True)

def build_notebook(cells):
    return {
        "cells": [
            {
                "cell_type": "markdown" if c.startswith("MD|") else "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [(line + "\n") for line in (c[3:] if c.startswith("MD|") else c).split("\n")] if isinstance(c, str) else [line + "\n" for line in c]
            } for c in cells
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

n01 = [
"MD|# 01 - Data Exploration\nThis notebook loads and explores the datasets.",
"""# Datasets in Google Drive:
# 1. jigsaw-toxic-comment-classification-challenge
# 2. hasoc-2021

import sys
from google.colab import drive
drive.mount('/content/drive')

# Adjust this path if your project is in a different folder
PROJECT_PATH = '/content/drive/MyDrive/toxicity-classifier/'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

import matplotlib.pyplot as plt
import seaborn as sns
from src.data.dataset_loader import DatasetLoader
from src.utils.visualizer import generate_wordcloud

# Adjust base_path to where your datasets are stored in Drive
loader = DatasetLoader(base_path="/content/drive/MyDrive/toxicity-classifier/data/raw")
jigsaw = loader.load_jigsaw()
hasoc = loader.load_hasoc_2021()
# multi = loader.load_multimodal_hate_speech() # Skipping multimodal for now

print("Jigsaw shape:", jigsaw.shape)
print("HASOC shape:", hasoc.shape)
""",
"""if not hasoc.empty:
    hasoc['lang_heuristic'].value_counts().plot(kind='bar')
    plt.title('Language distribution in HASOC')
    plt.show()"""
]

n02 = [
"MD|# 02 - Data Preprocessing\nThis notebook cleans text, merges the datasets, and balances classes.",
"""!pip install langdetect -q""",
"""import sys
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

PROJECT_PATH = '/content/drive/MyDrive/toxicity-classifier/'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

import yaml
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import TextPreprocessor
from src.data.data_merger import DataMerger

with open(f'{PROJECT_PATH}configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

loader = DatasetLoader(base_path="/content/drive/MyDrive/toxicity-classifier/data/raw")
jigsaw = loader.load_jigsaw()
hasoc = loader.load_hasoc_2021()
multi = pd.DataFrame() # Skipping multimodal for now

merger = DataMerger(config)
train_df, val_df, test_df = merger.merge_and_split(jigsaw, hasoc, multi)

print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

import os
os.makedirs(f'{PROJECT_PATH}data/processed', exist_ok=True)

# Saving to Google Drive so they persist across Colab session disconnects
train_df.to_csv(f'{PROJECT_PATH}data/processed/train.csv', index=False)
val_df.to_csv(f'{PROJECT_PATH}data/processed/val.csv', index=False)
test_df.to_csv(f'{PROJECT_PATH}data/processed/test.csv', index=False)
"""
]

n03 = [
"MD|# 03 - Model Training\nCustom training loop for XLM-RoBERTa using PyTorch AMP.",
"""!pip install transformers datasets accelerate -q""",
"""import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yaml
import sys
import os
from tqdm.auto import tqdm

from google.colab import drive
drive.mount('/content/drive')

PROJECT_PATH = '/content/drive/MyDrive/toxicity-classifier/'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from src.models.classifier import ToxicityClassifier
from src.utils.metrics import compute_classification_metrics

with open(f'{PROJECT_PATH}configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data (saved in Notebook 02)
try:
    train_df = pd.read_csv(f'{PROJECT_PATH}data/processed/train.csv')
    val_df = pd.read_csv(f'{PROJECT_PATH}data/processed/val.csv')
except FileNotFoundError:
    print("Files not found. Run Notebook 02 first to generate train.csv and val.csv in your Drive.")

tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

train_dataset = ToxicityDataset(train_df['text'].values, train_df['label'].values, tokenizer)
val_dataset = ToxicityDataset(val_df['text'].values, val_df['label'].values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=config['training'].get('batch_size', 16), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['training'].get('batch_size', 16), shuffle=False)
""",
"""%%time
# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ToxicityClassifier(model_name=config['model']['name'], num_labels=config['model']['num_labels']).to(device)

epochs = config['training'].get('epochs', 3)
optimizer = AdamW(model.parameters(), lr=float(config['training'].get('learning_rate', 2e-5)))
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss()
scaler = GradScaler()

best_val_loss = float('inf')
os.makedirs(f"{PROJECT_PATH}models", exist_ok=True)
model_save_path = f"{PROJECT_PATH}models/best_model.pt"

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            logits, _ = model(input_ids, attention_mask)
            loss = loss_fn(logits, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation
    model.eval()
    total_val_loss = 0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            loss = loss_fn(logits, targets)
            total_val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).flatten().cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(targets.cpu().numpy())
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_metrics = compute_classification_metrics(val_targets, val_preds, num_classes=config['model']['num_labels'])
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"Saving new best model to {model_save_path}")
        torch.save(model.state_dict(), model_save_path)
"""
]

n04 = [
"MD|# 04 - Model Evaluation\nComputes Metrics like macro F1, Accuracy, MCC, ROC-AUC.",
"""!pip install transformers datasets accelerate -q""",
"""import torch
import pandas as pd
import numpy as np
import yaml
import sys
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from google.colab import drive
drive.mount('/content/drive')

PROJECT_PATH = '/content/drive/MyDrive/toxicity-classifier/'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from src.utils.metrics import compute_classification_metrics
from src.utils.visualizer import plot_confusion_matrix
from src.models.classifier import ToxicityClassifier

with open(f'{PROJECT_PATH}configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Evaluation setup
try:
    test_df = pd.read_csv(f'{PROJECT_PATH}data/processed/test.csv')
    print("Test set loaded from Drive.")
except FileNotFoundError:
    print("test.csv not found in Drive. Run Notebook 02 first.")

tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

test_dataset = ToxicityDataset(test_df['text'].values, test_df['label'].values, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
""",
"""# Inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ToxicityClassifier(model_name=config['model']['name'], num_labels=config['model']['num_labels']).to(device)

model_path = f"{PROJECT_PATH}models/best_model.pt"
print(f"Loading weights from {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

test_preds = []
test_targets = []
test_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        logits, _ = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        test_preds.extend(preds)
        test_probs.extend(probs)
        test_targets.extend(targets.cpu().numpy())

metrics = compute_classification_metrics(test_targets, test_preds, np.array(test_probs), num_classes=config['model']['num_labels'])

print("--- Test Evaluation Metrics ---")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['f1_macro']:.4f}")
print(f"MCC: {metrics['mcc']:.4f}")
if metrics['roc_auc_macro']:
    print(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")

# Plotting Confusion Matrix
# Assuming label mapping like config['labels']: {'neutral': 0, 'offensive': 1, 'hate': 2, 'threat': 3}
# You can invert config['labels'] to get class names
label_names = sorted(config['labels'], key=config['labels'].get)
plot_confusion_matrix(metrics['confusion_matrix'], classes=label_names)
"""
]

n05 = [
"MD|# 05 - Explainability\nGenerates Attention highlights and SHAP text plots.",
"""!pip install shap transformers datasets accelerate -q""",
"""import torch
import sys
import yaml

from google.colab import drive
drive.mount('/content/drive')

PROJECT_PATH = '/content/drive/MyDrive/toxicity-classifier/'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from transformers import AutoTokenizer
from src.models.classifier import ToxicityClassifier
from src.models.explainer import ExplainerUtil
from src.utils.visualizer import render_attention_html
from IPython.display import HTML, display

with open(f'{PROJECT_PATH}configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

model = ToxicityClassifier(model_name=config['model']['name'], num_labels=config['model']['num_labels']).to(device)
model_path = f"{PROJECT_PATH}models/best_model.pt"
print(f"Loading weights from {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))

explainer = ExplainerUtil(model, tokenizer, device)
""",
"""# 1. Attention Visualizer
text = "I am going to kill you, you stupid idiot"
print(f"Analyzing Text: {text}")

attention_data = explainer.extract_attention_weights(text)
print(f"Predicted Class ID: {attention_data['predicted_class']} (Confidence: {attention_data['confidence']:.2f})")

html_code = render_attention_html(attention_data['tokens'], attention_data['attention_scores'])
display(HTML(html_code))
""",
"""# 2. SHAP Visualizer
import numpy as np

def model_predict_wrapper(texts):
    # SHAP provides texts as an array of strings
    model.eval()
    probs_list = []
    
    with torch.no_grad():
        for t in texts:
            inputs = tokenizer(t, return_tensors="pt", max_length=128, truncation=True, padding='max_length')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits, _ = model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_list.append(probs)
            
    return np.array(probs_list)

shap_explainer = explainer.build_shap_explainer(model_predict_wrapper)
shap_values = shap_explainer([text])

import shap
# Visualization for the predicted class
shap.plots.text(shap_values[0, :, attention_data['predicted_class']])
"""
]

all_nbs = {
    "01_data_exploration.ipynb": n01,
    "02_data_preprocessing.ipynb": n02,
    "03_model_training.ipynb": n03,
    "04_model_evaluation.ipynb": n04,
    "05_explainability.ipynb": n05
}

for name, cells in all_nbs.items():
    with open(notebooks_dir / name, 'w', encoding='utf-8') as f:
        json.dump(build_notebook(cells), f, indent=2)

print("Notebooks scaffolded successfully.")
