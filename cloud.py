import logging
import hydra

from eval import compute_metrics
from sagelearner import SageLearner
from tools import iter_batches, getQueries
from data.prepare_dataset import setIndex, loadPubmedQA


@hydra.main(config_path=".conf/", config_name="config", version_base="1.3")
def train_model(cfg):
    indicies = setIndex(cfg)

    vector_index = indicies[0]  # Use the first index for retrieval

    learner = SageLearner(cfg, vector_index)
    queries = getQueries(cfg.data.split_path)
    if not queries:
        raise ValueError("No queries loaded from PubMedQA splits.")
    gold_lookup = _build_gold_lookup(cfg.data.split_path)
    max_new_tokens = getattr(cfg.train, "max_gen_tokens", 128)
    eval_sample_size = getattr(cfg.train.logging, "eval_sample_size", 5)
    for epoch in range(cfg.train.epochs):
        logging.info(f"Epoch {epoch+1}/{cfg.train.epochs}")
        total_loss = 0.0
        for batch in iter_batches(queries, cfg.train.batch_size):
            for query in batch:
                loss = learner.train_step(query)
                total_loss += loss.item()
        avg_loss = total_loss / len(queries)
        logging.info(f"Average Loss: {avg_loss:.4f}.")
        print(f"Average Loss: {avg_loss:.4f}. Epoch: {epoch+1}")

        # Evaluate model at the end of each epoch
        if (epoch + 1) % cfg.train.logging.eval_every_steps == 0:
            metrics = evaluate_model(
                learner,
                queries,
                gold_lookup,
                sample_size=eval_sample_size,
                max_new_tokens=max_new_tokens,
            )
            logging.info(f"Evaluation Metrics at epoch {epoch+1}: {metrics}")
            print(f"Evaluation Metrics at epoch {epoch+1}: {metrics}")
        if (epoch + 1) % cfg.train.logging.save_every_steps == 0:
            learner.save_student()
            logging.info(f"Student model saved at epoch {epoch+1}.")
            print(f"Student model saved at epoch {epoch+1}.")
    # Save the trained student model
    learner.save_student()
    logging.info(f"Student model saved. Path: {cfg.train.logging.save_dir}")
    print("Student model saved. Path:", cfg.train.logging.save_dir)

def evaluate_model(learner, queries, gold_lookup, sample_size=5, max_new_tokens=128):
    sample_size = min(sample_size, len(queries))
    if sample_size == 0:
        return {}
    sample_queries = queries[:sample_size]
    student_texts = [
        learner.generate_answer(q, use_teacher=False, max_new_tokens=max_new_tokens)
        for q in sample_queries
    ]
    teacher_texts = [
        learner.generate_answer(q, use_teacher=True, max_new_tokens=max_new_tokens)
        for q in sample_queries
    ]
    gold_texts = [gold_lookup.get(q) for q in sample_queries]
    if any(g is None for g in gold_texts):
        gold_texts = None
    return compute_metrics(student_texts, teacher_texts, gold_texts=gold_texts)


def _build_gold_lookup(split_path, folds=10):
    lookup = {}
    for i in range(folds):
        dataset = loadPubmedQA(split_path, split=i)
        for example in dataset.values():
            query = example.get("QUESTION")
            answer = example.get("LONG_ANSWER") or example.get("final_decision")
            if query and answer:
                lookup[query] = answer
    return lookup


if __name__ == "__main__":
    train_model()
