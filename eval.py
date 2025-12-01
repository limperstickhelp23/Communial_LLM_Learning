import evaluate
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

rouge = evaluate.load("rouge")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

"""
Compute evaluation metrics between student and teacher texts, optionally using gold texts.

Computes the following metrics:
- ROUGE-1 and ROUGE-L (if gold_texts provided, compares student_texts to gold_texts; otherwise compares student_texts to teacher_texts)
- CLIPScore (compares student_texts to teacher_texts) (ON going issue with CLIPScore installation)
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



def evaluate_model(learner, dataset, sample_size=5, max_new_tokens=128):
    """
    Evaluate the model on a sample of the dataset.
    Computes metrics comparing student and teacher generated answers.
    metrics computed: 
        ROUGE-1, ROUGE-L, Cosine Similarity
    Args:
        learner: SageLearner instance
        dataset: PubMedQADataset instance
        sample_size: Number of samples to evaluate
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Dictionary of evaluation metrics
    """
    sample_size = min(sample_size, len(dataset))
    if sample_size == 0:
        return {}
    
    # Get sample queries and gold answers
    sample_queries = dataset.get_queries()[:sample_size]
    # gold_lookup = dataset.get_gold_lookup()
    # gold_texts = [gold_lookup.get(q) for q in sample_queries]
    
    # Generate answers in batches for efficiency
    batch_size = min(8, sample_size)  # Use smaller batch for generation
    
    student_texts = []
    teacher_texts = []
    
    for i in range(0, sample_size, batch_size):
        batch_queries = sample_queries[i:i+batch_size]
        
        # Generate student answers
        student_batch = learner.generate_answer_batch(
            batch_queries, 
            use_teacher=False, 
            max_new_tokens=max_new_tokens
        )
        student_texts.extend(student_batch)
        
        # Generate teacher answers
        teacher_batch = learner.generate_answer_batch(
            batch_queries,
            use_teacher=True,
            max_new_tokens=max_new_tokens
        )
        teacher_texts.extend(teacher_batch)
    
    # Compute metrics
    if any(g is None for g in gold_texts):
        gold_texts = None
    gold_texts = None #for now 
    return compute_metrics(student_texts, teacher_texts, gold_texts=gold_texts)
