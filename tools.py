import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from hydra.utils import instantiate
import torch
from torch.nn import functional as F

def load_student(cfg)->tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(cfg.hf_model.pretrained_model_name_or_path, use_fast=True)
    model = instantiate(cfg.hf_model)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    lora = instantiate(cfg.peft)
    model = get_peft_model(model, lora)
    return model, tok


def contrastive_loss(student_emb, teacher_emb, temperature=0.07):
    logits = (student_emb @ teacher_emb.T) / temperature
    labels = torch.arange(len(teacher_emb), device=student_emb.device)
    loss = F.cross_entropy(logits, labels)
    return loss

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

        




