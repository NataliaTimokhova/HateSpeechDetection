import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer

import wandb

from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent / "src"))

from helper_functions import AttentionPooling, HateSpeechDataset
from helper_functions import train_model, test_model, unfreeze_last_n_layers, get_class_distribution
from models import LargeCustomClassifier
from paths import DATA_CLEANED

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label = "task_1"

# load and split data only once
clean_df = pd.read_csv(DATA_CLEANED / "hasoc_2019_en_train_cleaned.tsv", sep='\t')
test_df = pd.read_csv(DATA_CLEANED / "hasoc_2019_en_test_cleaned.tsv", sep='\t')
train_df, val_df = train_test_split(clean_df, test_size=0.3, random_state=42, stratify=clean_df[label])

# encode labels
label_list = sorted(train_df[label].unique())
label_map = {label: idx for idx, label in enumerate(label_list)}
for df in [train_df, val_df, test_df]:
    df[label] = df[label].map(label_map)

class_weights = compute_class_weight("balanced", classes=np.unique(train_df[label]), y=train_df[label])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

def sweep_train():
    wandb.init(project="roberta-sweep")
    config = wandb.config

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_dataset = HateSpeechDataset(train_df, tokenizer, label=label)
    val_dataset = HateSpeechDataset(val_df, tokenizer, label=label)
    test_dataset = HateSpeechDataset(test_df, tokenizer, label=label)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = LargeCustomClassifier(
        model_name=config.model_name,
        class_weights_tensor=class_weights_tensor,
        device=device,
        pooling=config.pooling
    ).to(device)

    if config.unfrozen_last_layers > 0:
        unfreeze_last_n_layers(model, config.unfrozen_last_layers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_model_path = "best_model_sweep.pt"
    train_model(model, train_loader, val_loader, optimizer, device, epochs=config.epochs, best_model_path=best_model_path)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_model(model, test_loader, device, phase="test")