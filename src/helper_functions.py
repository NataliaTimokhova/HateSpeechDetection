import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Custom Dataset Class
class HateSpeechDataset(Dataset):
    def __init__(self, df, tokenizer, label = 'label', max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df[label].tolist()
        self.encodings = tokenizer(self.texts, padding=True, truncation=True, max_length=max_len)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)
        
        intermediate_size = hidden_size // 4
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 1),
        )

    def forward(self, hidden_states, mask):
        hidden_states = self.layernorm(hidden_states)
        # mask: (batch_size, seq_len)
        scores = self.attention(hidden_states).squeeze(-1)  # (batch_size, seq_len)
        scores = scores.masked_fill(mask == 0, -1e9)  # Mask padding
        attn_weights = torch.softmax(scores / 0.7, dim=1)  # softer softmax # (batch_size, seq_len)
        weighted_sum = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)
        
        return weighted_sum


def train_model(model, train_loader, test_loader, optimizer, device, epochs=1, best_model_path = 'best_model.pt', scheduler = False):
    """
    Train the model and validate after each epoch.
    """
    
    best_f1 = 0.0
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        
        model.train()
        train_preds, train_labels = [], []
        total_loss = 0

        # Training loop
        for batch in train_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass
            logits, loss = model(input_ids, attention_mask, model.class_weights, labels)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Compute training metrics
        avg_train_loss = total_loss / len(train_loader)
        acc = accuracy_score(train_labels, train_preds)
        prec = precision_score(train_labels, train_preds, average='macro')
        rec = recall_score(train_labels, train_preds, average='macro')
        f1 = f1_score(train_labels, train_preds, average='macro')
        f1_weighted = f1_score(train_labels, train_preds, average='weighted')

        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Train Accuracy: {acc:.4f}")
        print(f"Train F1 (macro): {f1:.4f}\n")

        # Log results
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": acc,
            "train_precision_macro": prec,
            "train_recall_macro": rec,
            "train_f1_macro": f1,
            "train_f1_weighted": f1_weighted,
            "scheduler": scheduler.__class__.__name__ if scheduler else '',
            "current_lr": optimizer.param_groups[0]['lr']
        })

        # Call the test_model function for validation after each epoch
        val_metrics = test_model(model, test_loader, device, phase="val")

        # Save best model after validation
        if (
            val_metrics["f1"] > best_f1 or 
            (val_metrics["f1"] == best_f1 and val_metrics["accuracy"] > best_acc)
        ):
            best_f1 = val_metrics["f1"]
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (F1: {best_f1:.4f}, Acc: {best_acc:.4f})")


        

def test_model(model, data_loader, device, phase="test"):
    """
    Evaluates the model on the validation or test set.
    :param phase: One of ["val", "test"] for validation or final testing. Needed for correct logging with wandb
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass (no gradient calculation for validation or testing)
        with torch.no_grad():
            # The loss is required to optimise the model (backpropagation) and is no longer important for testing. 
            # But to make coding easier we opted to not do the case destinction
            logits, loss = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                class_weights_tensor=model.class_weights, 
                labels=labels)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_loss += loss.item()

    # Compute metrics
    avg_test_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    print(f"{phase.capitalize()} Loss: {avg_test_loss:.4f}")
    print(f"{phase.capitalize()} Accuracy: {acc:.4f}")
    print(f"{phase.capitalize()} F1 (macro): {f1:.4f}\n")

    # Log the metrics in wandb
    wandb.log({
        f"{phase}_loss": avg_test_loss,
        f"{phase}_accuracy": acc,
        f"{phase}_precision_macro": prec,
        f"{phase}_recall_macro": rec,
        f"{phase}_f1_macro": f1,
        f"{phase}_f1_weighted": f1_weighted
    })

    if phase  == "test":
        cm = confusion_matrix(all_labels, all_preds)
        fig = plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d")
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        wandb.finish()

    return {"f1": f1, "accuracy": acc}

def unfreeze_last_n_layers(model, n_layers=2):
    """
    Unfreezes the last `n_layers` of the Transformer encoder.
    Assumes the transformer model has an encoder with a `layer` attribute.
    """
    # Freeze all parameters first
    for param in model.base.parameters():
        param.requires_grad = False

    # Unfreeze the last n transformer layers
    encoder_layers = model.base.encoder.layer
    for layer in encoder_layers[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Optionally unfreeze LayerNorm and pooler if available
    if hasattr(model.base, "pooler"):
        for param in model.base.pooler.parameters():
            param.requires_grad = True

    if hasattr(model.base, "embeddings") and hasattr(model.base.embeddings, "LayerNorm"):
        for param in model.base.embeddings.LayerNorm.parameters():
            param.requires_grad = True

def get_class_distribution(dataset, label):
    """
    Determines the class distribution i.e. how many elements a class has
    """
    labels = dataset[label].tolist()
    distribution = {}
    for i in range(len(labels)):
        target = labels[i]
        if target not in distribution:
            distribution[target] = 0
        distribution[target] += 1
    return distribution

def get_target_text_by_label(dataset, label, label_value):
    """
    Extracts only a specific class from a pandas dataframe
    """
    target_text = []
    texts = dataset["text"].tolist()
    labels = dataset[label].tolist()
    for i in range(len(labels)):
        if labels[i] == label_value:
            target_text.append(texts[i])
    return target_text

def oversample_dataset(dataset, label):
    """
    Randomly oversamples the dataset and returns a new panda dataframe usable for the HateSpeechDataset
    """
    dist = get_class_distribution(dataset, label)
    highest_class_val = max(dist.values())

    texts = []
    labels = []
    for element in dist:
        # Skipping the class with the most entries
        target_text = get_target_text_by_label(dataset, label, element)
        if dist[element] == highest_class_val:
            for text in target_text:
                texts.append(text)
                labels.append(element)
            continue
        diff = highest_class_val - len(target_text)
        oversampled_class = random.choices(target_text, k=diff)
        for text in target_text + oversampled_class:
            texts.append(text)
            labels.append(element)
    d = {"text": texts, f"{label}": labels}
    return pd.DataFrame(data=d)

def undersample_dataset(dataset, label):
    """
    Randomly undersamples the dataset and returns a new pandas DataFrame usable for the HateSpeechDataset.
    """
    dist = get_class_distribution(dataset, label)
    lowest_class_val = min(dist.values())

    texts = []
    labels = []
    for element in dist:
        target_text = get_target_text_by_label(dataset, label, element)
        # Random selection of lowest_class_val many texts from each class
        undersampled_class = random.sample(target_text, lowest_class_val)
        for text in undersampled_class:
            texts.append(text)
            labels.append(element)
    
    d = {"text": texts, f"{label}": labels}
    return pd.DataFrame(data=d)
