import evaluate
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

rouge = evaluate.load("rouge")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

"""
Compute evaluation metrics between student and teacher texts, optionally using gold texts.

Computes the following metrics:
- ROUGE-1 and ROUGE-L (if gold_texts provided, compares student_texts to gold_texts; otherwise compares student_texts to teacher_texts)
- CLIPScore (compares student_texts to teacher_texts)
- Cosine Similarity of SBERT embeddings (compares student_texts to teacher_texts)
"""
def compute_metrics(student_texts, teacher_texts, gold_texts=None)->dict:
    metrics = {}

    if gold_texts:
        r = rouge.compute(predictions=student_texts, references=gold_texts)
        metrics.update({k: r[k] for k in ["rouge1","rougeL"]})
    else:
        r = rouge.compute(predictions=student_texts, references=teacher_texts)
        metrics.update({k+"_t": r[k] for k in ["rouge1","rougeL"]})

    # c = compute_clipscore(predictions=student_texts, references=teacher_texts)
    # metrics["clipscore"] = c["clipscore"]

    eS = sbert.encode(student_texts, convert_to_tensor=True, normalize_embeddings=True)
    eT = sbert.encode(teacher_texts, convert_to_tensor=True, normalize_embeddings=True)
    metrics["cosine_sim"] = float(F.cosine_similarity(eS, eT).mean())

    return metrics
