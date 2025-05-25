import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import json
import pandas as pd
import numpy as np


# Configuration
MAX_LEN = 196   # 196 .. 128-256
BATCH_SIZE = 32 # 24-32 .. 16-64 
EPOCHS = 20  # 3-5 .. 2-10
LEARNING_RATE = 1e-3  # 2e-5 .. 1e-5-5e-5    1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# --------------------
# Dataset
# --------------------
class MyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        token_data = json.loads(row['token_data'])
        bert_tokens = [t['token'] for t in token_data]
        bert_logprobs = [t['logprob'] for t in token_data]

        # Build sequence with special tokens
        seq = [self.tokenizer.cls_token] \
            + bert_tokens[:self.max_length-2] \
            + [self.tokenizer.sep_token]
        lp  = [0.0] \
            + bert_logprobs[:self.max_length-2] \
            + [0.0]

        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(seq)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids    += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            lp            += [0.0] * pad_len
        else:
            input_ids     = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            lp             = lp[:self.max_length]

        # Optionally check
        assert len(input_ids) == self.max_length
        assert len(lp)        == self.max_length

        # Global stats
        avg_lp = float(np.mean(bert_logprobs))
        min_lp = float(np.min(bert_logprobs))
        max_lp = float(np.max(bert_logprobs))
        std_lp = float(np.std(bert_logprobs))

        return {
            'input_ids':       torch.tensor(input_ids,       dtype=torch.long),
            'attention_mask':  torch.tensor(attention_mask,  dtype=torch.long),
            'token_logprobs':  torch.tensor(lp,              dtype=torch.float32),
            'global_features': torch.tensor([avg_lp, min_lp, max_lp, std_lp], dtype=torch.float32),
            'label':           torch.tensor(row['is_hallucinated'], dtype=torch.long)
        }

# --------------------
# Embedding Wrapper
# --------------------
class MyEmbed(nn.Module):
    def __init__(self, inner_embed):
        super().__init__()
        self.inner_embed = inner_embed
        emb_dim = inner_embed.word_embeddings.embedding_dim
        self.prob_embed = nn.Sequential(
            nn.Linear(1, 128), 
            nn.LayerNorm(128),
            nn.SiLU(), 
            # nn.Dropout(0.2),
            nn.Linear(128, emb_dim)
        )
        # Initialize with small random values 
        nn.init.xavier_uniform_(self.prob_embed[0].weight)
        nn.init.xavier_uniform_(self.prob_embed[-1].weight)
        self.log_probs = None
        

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None, **kwargs):
        # Base embeddings: word + position + token_type
        base = self.inner_embed(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if self.log_probs is not None:
            logprobs_unsqueezed = self.log_probs.unsqueeze(-1)
            extra = self.prob_embed(logprobs_unsqueezed)
            return base + extra
        return base
    

# --------------------
# Model
# --------------------
class HallucinationDetectorOld(nn.Module):
    def __init__(self, use_logprobs=True, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.use_logprobs = use_logprobs
        
        # Wrap embeddings if using logprobs
        if use_logprobs:
            self.bert.embeddings = MyEmbed(self.bert.embeddings)
        
        # Freeze BERT if desired
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
            # But always unfreeze the logprob adapter if we're using it
            if use_logprobs:
                for p in self.bert.embeddings.prob_embed.parameters():
                    p.requires_grad = True
                for p in self.bert.embeddings.gate.parameters():
                    p.requires_grad = True
        
        self.dropout = nn.Dropout(0.3)
        hidden_size = self.bert.config.hidden_size
        
        # Add global features processor
        self.global_feat_proj = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, input_ids, attention_mask, token_logprobs=None, global_features=None):
        # Process logprobs if using them
        if self.use_logprobs and token_logprobs is not None:
            # Apply clipping to handle extreme values
            clipped_logprobs = torch.clamp(token_logprobs, min=-20.0, max=0.0)
            
            # Standardize per example
            mu = clipped_logprobs.mean(dim=1, keepdim=True)
            sig = clipped_logprobs.std(dim=1, keepdim=True) + 1e-6
            norm = (clipped_logprobs - mu) / sig
            
            self.bert.embeddings.log_probs = norm
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            self.bert.embeddings.log_probs = None
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        attended_output = outputs.pooler_output
        
        # Process global features if provided
        if global_features is not None:
            global_feats = self.global_feat_proj(global_features)
            # Concatenate with attended output
            combined = torch.cat([attended_output, global_feats], dim=1)
        else:
            combined = torch.cat([attended_output, torch.zeros(attended_output.size(0), 64, device=attended_output.device)], dim=1)
        
        # Final classification
        return self.classifier(attended_output).squeeze(-1)#combined)
    
class HallucinationDetectorA(nn.Module):
    def __init__(self, use_logprobs=True, hidden_size=128, prob_embed_size=32):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.bert_drop = nn.Dropout(0.1)
        
        self.prob_proj = nn.Sequential(
            nn.Linear(1, prob_embed_size),
            nn.SiLU()
        )
        
        self.cls1 = nn.Sequential(
            nn.Linear(768+prob_embed_size, hidden_size),
            nn.SiLU()
        )
        
        self.pool = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Softmax(dim=1))
        self.pool[0].weight.data *= 0
        
        self.cls2 = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, input_ids, attention_mask, token_logprobs=None, global_features=None):
        bert_output = self.bert_drop(self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        prob_proj = self.prob_proj(token_logprobs.unsqueeze(-1))
        
        combined = torch.cat([bert_output, prob_proj], dim=-1)
        
        pre_pool = self.cls1(combined)
        attn = self.pool(pre_pool)
        
        pooled = (attn * pre_pool).sum(dim=1)
        out = self.cls2(pooled).squeeze(-1)
        return out
    
        
class HallucinationDetector(nn.Module):
    # hidden_size: 512 .. 256-1024
    # prob_embed_size 16-64 .. 8-128
    # dropout 0.1-0.3 .. 0.05-0.5


    # hiden 32 , prob 64
    def __init__(self, use_logprobs=True, hidden_size=256, prob_embed_size=128):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.bert_drop = nn.Dropout(0.5)
        
        self.prob_proj = nn.Sequential(
            nn.Linear(1, prob_embed_size),
            nn.SiLU()
        )

        self.cls1 = nn.Sequential(
            nn.Linear(768+prob_embed_size, hidden_size),
            nn.SiLU()
        )
        
        self.pool = nn.Sequential(nn.Linear(768, 1), nn.Softmax(dim=1))

        
        self.cls2 = nn.Sequential(
            nn.Linear(prob_embed_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )
        
        # Cache for BERT output to avoid recomputation
        self.cached_input_ids = None
        self.cached_attention_mask = None
        self.cached_bert_output = None
        
    def forward(self, input_ids, attention_mask, token_logprobs=None, global_features=None):
        # Implement caching mechanism for BERT outputs
        # Check if we can use cached output
        use_cache = False
        if self.cached_input_ids is not None and self.cached_attention_mask is not None:
            if (input_ids.shape == self.cached_input_ids.shape and 
                attention_mask.shape == self.cached_attention_mask.shape and
                torch.all(input_ids == self.cached_input_ids) and 
                torch.all(attention_mask == self.cached_attention_mask)):
                use_cache = True
        
        # Compute or retrieve BERT output
        if not use_cache:
            with torch.set_grad_enabled(self.training):  # Only compute gradients during training
                bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                # Update cache
                if not self.training:  # Only cache during evaluation to save memory during training
                    self.cached_input_ids = input_ids.clone()
                    self.cached_attention_mask = attention_mask.clone()
                    self.cached_bert_output = bert_output.clone()
        else:
            bert_output = self.cached_bert_output
        
        # Apply dropout to BERT outputs (only during training)
        bert_output_with_drop = self.bert_drop(bert_output)
        
        # Project logprobs to the hidden dimension
        prob_proj = self.prob_proj(token_logprobs.unsqueeze(-1))
        
        # Compute attention weights
        attn = self.pool(bert_output_with_drop)
        
        # Apply attention pooling
        pooled = (attn * prob_proj).sum(dim=1)
        
        # Final classification
        out = self.cls2(pooled).squeeze(-1)
        return out
    


# --------------------
# Training & Eval
# --------------------
def train_epoch(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_logprobs=inputs.get('token_logprobs'),
            global_features=inputs.get('global_features')
        )
        
        
        loss = criterion(logits, inputs['label'].float())
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        
        # Track accuracy
        preds = logits > 0 
        correct += (preds == inputs['label']).sum().item()
        total += inputs['label'].size(0)
    
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    true_pos = false_pos = true_neg = false_neg = 0
    total_ones = 0
    
    labs = []
    probs = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            logits = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_logprobs=inputs.get('token_logprobs'),
                global_features=inputs.get('global_features')
            )
            
            loss = criterion(logits, inputs['label'].float())
            total_loss += loss.item()
            
            preds = logits > 0
            correct += (preds == inputs['label']).sum().item()
            total += inputs['label'].size(0)
            
            # Calculate metrics
            for logs, pred, label in zip(logits.cpu().numpy(), preds.cpu().numpy(), inputs['label'].cpu().numpy()):
                labs.append(label)
                probs.append(logs)
                if label == 1:  # Hallucinated
                    total_ones += 1
                    if pred == 1:
                        true_pos += 1
                    else:
                        false_neg += 1
                else:  # Not hallucinated
                    if pred == 0:
                        true_neg += 1
                    else:
                        false_pos += 1
    
    accuracy = correct / total
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    roc = roc_auc_score(labs, probs)
    
    return total_loss / len(loader), accuracy, precision, recall, f1, total_ones, total, roc

# --------------------
# Data Prep
# --------------------
def prepare_data(path):
    df = pd.read_csv(path)
    df = df[df['use_for_training']==1].reset_index(drop=True)
    
    # Check class balance
    positive = df['is_hallucinated'].sum()
    total = len(df)
    print(f"Class balance: {positive}/{total} ({positive/total:.2%}) positive examples")
    
    groups = df['story_id']
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=groups))
    
    print(len(train_idx), len(val_idx))
    
    return df.iloc[train_idx], df.iloc[val_idx]



train_df, val_df = prepare_data('bert_tokens_final.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_ds = MyDataset(train_df, tokenizer, MAX_LEN)
val_ds = MyDataset(val_df, tokenizer, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = HallucinationDetector(use_logprobs=True).to(device)
model.cuda()
print(model)

# Calculate class weights for imbalanced dataset
n_samples = len(train_df)
n_hallucinated = train_df['is_hallucinated'].sum()
hallucinated_weight = n_samples / (2 * n_hallucinated) if n_hallucinated > 0 else 1.0
non_hallucinated_weight = n_samples / (2 * (n_samples - n_hallucinated)) if n_samples > n_hallucinated else 1.0

# Ensure weights are float32
class_weights = torch.tensor([non_hallucinated_weight, hallucinated_weight], dtype=torch.float32, device=device)

# Use weighted cross entropy loss
#criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.BCEWithLogitsLoss()

# Parameter groups with different learning rates
optimizer = optim.AdamW([
    {'params': model.prob_proj.parameters(), 'lr': LEARNING_RATE},
    {'params': model.pool.parameters(), 'lr': LEARNING_RATE},
    {'params': model.cls2.parameters(), 'lr': LEARNING_RATE},
], weight_decay=1e-5)

# Learning rate scheduler with warmup
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop with early stopping
best_f1 = 0.0
patience = 300
patience_counter = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler)
    val_loss, val_acc, val_prec, val_rec, val_f1, total_ones, total, roc = eval_epoch(model, val_loader, criterion)

    print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f} | O: {total_ones}/{total} | Roc {roc:.4f}')

    # Early stopping based on F1 score
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_hallucination_detector.pt')
        print(f"New best model saved with F1: {val_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load the best model for final evaluation
model.load_state_dict(torch.load('best_hallucination_detector.pt'))
val_loss, val_acc, val_prec, val_rec, val_f1, total_ones, total, roc = eval_epoch(model, val_loader, criterion)
print(f'Final Best Model Metrics:')
print(f'Val Acc: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}')


