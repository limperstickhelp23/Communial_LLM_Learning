import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from hydra.utils import instantiate
import torch
from torch.nn import functional as F

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def load_student(cfg)->tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(cfg.hf_model.pretrained_model_name_or_path, use_fast=True)
    model = instantiate(cfg.hf_model)
    
    # model = model.to(device) # Accelerate will handle this
    # Enable gradient checkpointing if specified
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    lora = instantiate(cfg.peft)
    model = get_peft_model(model, lora)
    return model, tok

def load_teacher(cfg)->AutoModelForCausalLM:
    model = instantiate(cfg.hf_model).to(device) # load base model
    # turn off gradient descent
    for param in model.parameters():
        param.requires_grad = False
    model.eval() # set to eval mode (Disables Dropout)
    return model 


def kl_div_loss(student_logits, teacher_logits, temperature=0.07):
    """Token-level KL divergence with temperature scaling."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/teacher logits must match. Got {student_logits.shape} vs {teacher_logits.shape}."
        )
    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_scaled, dim=-1).to(device)
    teacher_probs = F.softmax(teacher_scaled, dim=-1).to(device)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")




"""Legacy function (Using DataLoader and collator now)"""

def getQueries(split_path, folds=10)->list[str]:
    queries = [ ]
    for i in range(folds):
        path = os.path.join(split_path, f"pqal_fold{i}/dev_set.json")
        with open(path, 'r') as f:
            dataset_cl = json.load(f)
            queries = queries + [ ex["QUESTION"] for pubid, ex in dataset_cl.items() ]
    return queries

def iter_batches(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]

        



