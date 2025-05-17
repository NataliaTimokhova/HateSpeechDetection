import torch
import torch.nn as nn 

from transformers import AutoModel, AutoConfig

from helper_functions import AttentionPooling

class BaseCNNClassifier(nn.Module):
    def __init__(self, model_name, class_weights_tensor, device, unfreeze_n_layers=0):
        super().__init__()
        self.device = device
        self.base = AutoModel.from_pretrained(model_name)
        self.class_weights = class_weights_tensor
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
      
        for param in self.base.parameters():
            param.requires_grad = False

        # Unfreeze the last `unfreeze_n_layers` encoder layers (if Transformer-based)
        if unfreeze_n_layers > 0:
            if hasattr(self.base, 'encoder') and hasattr(self.base.encoder, 'layer'):
                layers = self.base.encoder.layer
                for layer in layers[-unfreeze_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 2)
        )

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

    def forward(self, input_ids, attention_mask=None, class_weights_tensor = None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        x = last_hidden_state.transpose(1, 2)
        logits = self.classifier(x)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss
        return logits
        

class CustomClassifier(nn.Module):
    def __init__(self, model_name, class_weights_tensor, device, pooling='cls'):
        super().__init__()
        self.device = device
        self.class_weights = class_weights_tensor
        self.pooling = pooling.lower()
        self.base = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size  # Dynamically get the model's hidden size

        if pooling == 'attention_pooling':
            # Replace CLS pooling with attention
            self.attention_pool = AttentionPooling(hidden_size)
        elif pooling == 'max':
            self.max_pool = nn.MaxPool2d(3, stride=2)
        
        # Freeze all parameters of the base model
        for param in self.base.parameters():
            param.requires_grad = False

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, class_weights_tensor = None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)

        # Pooling strategy: either CLS token or mean pooling over token embeddings
        if self.pooling == 'mean':
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling == 'attention_pooling':
            pooled = self.attention_pool(outputs.last_hidden_state, attention_mask)
        elif self.pooling == 'max':
            pooled = self.max_pool(outputs.last_hidden_state)
        else:
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS token

        logits = self.classifier(pooled)

        # If labels are provided, calculate the loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor.to(self.device))
            loss = loss_fn(logits, labels)
            return logits, loss

        return logits


class LargeCustomClassifier(nn.Module):
    def __init__(self, model_name, class_weights_tensor, device, pooling='cls'):
        super().__init__()
        self.device = device
        self.class_weights = class_weights_tensor
        self.pooling = pooling.lower()
        self.base = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size  # Dynamically get the model's hidden size

        if pooling == 'attention_pooling':
            # Replace CLS pooling with attention
            self.attention_pool = AttentionPooling(hidden_size)
        
        # Freeze all parameters of the base model
        for param in self.base.parameters():
            param.requires_grad = False

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, class_weights_tensor = None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)

        # Pooling strategy: either CLS token or mean pooling over token embeddings
        if self.pooling == 'mean':
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling == 'attention_pooling':
            pooled = self.attention_pool(outputs.last_hidden_state, attention_mask)
        else:
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS token

        logits = self.classifier(pooled)

        # If labels are provided, calculate the loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor.to(self.device))
            loss = loss_fn(logits, labels)
            return logits, loss

        return logits