import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from transformers import AutoTokenizer
from utils import set_seed
from data import prepare_sft_data, collate_smoltalk
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_sft_smol_smoltalk(num_epochs=5,
                           batch_size=16,
                           lr=5e-5,
                           weight_decay=0.01,
                           seed=42,
                           save=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### MODEL SETUP ####
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    #### MODEL SETUP ####
    
    #### DATA SETUP ####
    tokenized_sft = prepare_sft_data(tokenizer)
    
    # Data loaders
    train_loader = DataLoader(tokenized_sft["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_smoltalk)
    eval_loader = DataLoader(tokenized_sft["test"], batch_size=batch_size, shuffle=True, collate_fn=collate_smoltalk)
    #### DATA SETUP ####

    #### TRAIN ####
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} train", leave=True)
        for batch in batch_iter:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
    
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            shift_attn_mask = attention_mask[:, 1:]
            
            active_logits = shift_logits[shift_attn_mask == 1]
            active_labels = shift_labels[shift_attn_mask == 1]
    
            loss = criterion(active_logits, active_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_iter.set_postfix({'loss': total_loss / (batch_iter.n + 1)})
        print(f"Epoch {epoch+1} train loss: {total_loss/len(train_loader):.4f}")
    
        # Evaluation
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
        
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                shift_attn_mask = attention_mask[:, 1:]
                
                active_logits = shift_logits[shift_attn_mask == 1]
                active_labels = shift_labels[shift_attn_mask == 1]
        
                loss = criterion(active_logits, active_labels)
                eval_loss += loss.item()
        print(f"Epoch {epoch+1} eval loss: {eval_loss/len(eval_loader):.4f}")

    if save:
        # Save the model weights
        os.makedirs("sft_model", exist_ok=True)
        torch.save(model.state_dict(), "sft_model/model.pt")
        
    return model