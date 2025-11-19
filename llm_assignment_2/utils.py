from transformers import AutoModelForCausalLM
import torch
import os 
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_pt_model(model_hf_name, model_pt_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_hf_name)
    state_dict = torch.load(model_pt_path, map_location=device)  # or "cuda" if you want on GPU
    model.load_state_dict(state_dict)
    return model