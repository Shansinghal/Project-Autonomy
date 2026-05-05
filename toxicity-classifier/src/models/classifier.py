import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ToxicityClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_labels=4, dropout=0.3):
        super().__init__()
        self.num_labels = num_labels
        
        # Load pre-trained encoder 
        # (output_attentions=True is required for notebook 5 attention visualization)
        self.encoder = AutoModel.from_pretrained(model_name, output_attentions=True)
        config = AutoConfig.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # We will use an attention pooling instead of just [CLS] token extraction
        self.attention_pool = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, attention_mask):
        # outputs[0] is the last hidden state: [batch_size, seq_len, hidden_size]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        
        # Attention Pooling
        weights = self.attention_pool(hidden_states) # [batch_size, seq_len, 1]
        
        # Mask out padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        weights = weights * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Context vector: [batch_size, hidden_size]
        context = torch.sum(hidden_states * weights, dim=1)
        
        dropped = self.dropout(context)
        logits = self.classifier(dropped)
        
        # Return logits and also transformer attention weights from encoder
        return logits, outputs.attentions
