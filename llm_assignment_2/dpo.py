from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import set_seed
from data import benchmark_performance
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data import chat_template, collate_dpo, MAX_LEN, prepare_dpo_data

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_probs_from_model(model, tokenizer, prompts, responses):
    B = len(prompts)
    inputs = [chat_template(p, r) for p, r in zip(prompts, responses)]
    encodings = tokenizer(inputs, return_tensors="pt", max_length=MAX_LEN, 
                          truncation=True, padding="max_length", add_special_tokens=False)
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)

    # Get where prompt ends and response starts for each example
    prompt_encs = tokenizer([chat_template(p, "") for p in prompts], 
                            return_tensors="pt", max_length=MAX_LEN, 
                            truncation=True, padding="max_length", add_special_tokens=False)
    prompt_lengths = (prompt_encs.input_ids != tokenizer.pad_token_id).sum(dim=1)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)

    # Shift logits and input_ids for causal LM loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_att_mask = attention_mask[..., 1:].contiguous()

    # For each example, mask out prompt tokens and pad tokens
    batch_logprobs = []
    for i in range(B):
        resp_start = prompt_lengths[i].item()
        label = shift_labels[i]      # (L-1,)
        logit = shift_logits[i]      # (L-1, V)
        attn = shift_att_mask[i]     # (L-1,)

        # Only evaluate on the response tokens (those after prompt)
        mask = torch.zeros_like(label, dtype=torch.bool)
        mask[resp_start-1:] = 1      # response tokens start from resp_start-1 due to shifting

        valid_idx = (attn == 1) & mask
        # Get logprobs of the target tokens
        logprobs = F.log_softmax(logit, dim=-1)
        selected = logprobs[valid_idx, label[valid_idx]]
        batch_logprobs.append(selected.sum())
        
    return torch.stack(batch_logprobs)

def dpo_loss(policy_model, ref_model, tokenizer, prompts, chosens, rejecteds, beta=0.1):
    # Compute log probs under both models
    logpi_c = log_probs_from_model(policy_model, tokenizer, prompts, chosens)
    logpi_r = log_probs_from_model(policy_model, tokenizer, prompts, rejecteds)
    with torch.no_grad():
        logpref_c = log_probs_from_model(ref_model, tokenizer, prompts, chosens)
        logpref_r = log_probs_from_model(ref_model, tokenizer, prompts, rejecteds)

    # Compute preference difference
    pi_diff = (logpi_c - logpref_c) - (logpi_r - logpref_r)

    # DPO objective
    loss = -F.logsigmoid(beta * pi_diff)
    return loss

def train_dpo(model, 
              ref_model, 
              tokenizer, 
              train_loader_or_batch, 
              val_loader_or_batch=None, 
              num_epochs=2, 
              lr=5e-5, 
              weight_decay=0.1, 
              beta=0.1):

    if val_loader_or_batch:
        len_val_loader = 1 if isinstance(val_loader_or_batch, tuple) else len(val_loader_or_batch)
    # if the data is just one single batch 
    len_train_loader = 1 if isinstance(train_loader_or_batch, tuple) else len(train_loader_or_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        if isinstance(train_loader_or_batch, tuple):
            prompts, chosens, rejecteds = train_loader_or_batch
            optimizer.zero_grad()
            loss = dpo_loss(model, ref_model, tokenizer, prompts, chosens, rejecteds, beta)
            batch_loss = loss.mean()
            batch_loss.backward()
            optimizer.step()
            total_loss = batch_loss.item()
        else:
            pbar = tqdm(train_loader_or_batch, desc=f"Epoch {epoch+1}")
            for prompts, chosens, rejecteds in pbar:
                optimizer.zero_grad()
                loss = dpo_loss(model, ref_model, tokenizer, prompts, chosens, rejecteds, beta)
                batch_loss = loss.mean()
                pbar.set_description(f"Epoch {epoch+1} loss={batch_loss:.4f}")
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()

        if val_loader_or_batch:
            model.eval()
            with torch.no_grad():
                total_loss_val = 0.0
                if isinstance(train_loader_or_batch, tuple):
                    prompts, chosens, rejecteds = val_loader_or_batch
                    loss = dpo_loss(model, ref_model, tokenizer, prompts, chosens, rejecteds, beta)
                    batch_loss = loss.mean()
                    total_loss_val = batch_loss.item()
                else:
                    for prompts, chosens, rejecteds in tqdm(val_loader_or_batch, desc=f"Epoch {epoch+1} val"):
                        loss = dpo_loss(model, ref_model, tokenizer, prompts, chosens, rejecteds, beta)
                        batch_loss = loss.mean()
                        total_loss_val += batch_loss.item()
            print(f"Epoch {epoch+1}, mean DPO loss: {total_loss/len_train_loader:.4f}, val mean DPO loss: {total_loss_val/len_val_loader:.4f}")
        else:
            print(f"Epoch {epoch+1}, mean DPO loss: {total_loss/len_train_loader:.4f}")
    return model


def train_dpo_hh_rlhf(split_sizes=[5000, 500], 
                      batch_size=8,
                      num_epochs=2, 
                      lr=5e-5, 
                      weight_decay=0.01, 
                      beta=0.5,
                      seed=42,
                      perf_verbose=True):

    #### MODEL SETUP ####
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    # Ensure pad token is set for GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    sfted_model_name = "qwenzoo/utn-llm-assign2-gpt2-SFT"
    model = AutoModelForCausalLM.from_pretrained(sfted_model_name)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    ref_model = AutoModelForCausalLM.from_pretrained(sfted_model_name)
    ref_model.generation_config.pad_token_id = tokenizer.pad_token_id
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    model = model.to(device)
    ref_model = ref_model.to(device)
    
    set_seed(seed)
    #### MODEL SETUP ####

    #### DATA SETUP ####
    train_subset, val_subset = prepare_dpo_data(tokenizer=tokenizer, split_sizes=split_sizes)
    #### DATA SETUP ####

    #### TRAIN ####
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_dpo)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_dpo)

    model = train_dpo(model, 
                      ref_model, 
                      tokenizer, 
                      train_loader, 
                      val_loader,
                      num_epochs=num_epochs, 
                      lr=lr, 
                      weight_decay=weight_decay, 
                      beta=beta)

    perf = benchmark_performance(model, tokenizer, verbose=perf_verbose)

    print(f"Performance: {perf}")

    return model
    