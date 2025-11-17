import logging
import hydra
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from eval import evaluate_model
from sagelearner import SageLearner
from dataset import PubMedQADataset
from collator import KDCollator
from data.prepare_dataset import setIndex


@hydra.main(config_path=".conf/", config_name="config", version_base="1.3")
def train_model(cfg):
    # Setup indices
    indicies = setIndex(cfg)
    vector_index = indicies[0]  # Use the first index for retrieval
    
    # Initialize learner
    learner = SageLearner(cfg, vector_index)
    
    # Create dataset
    dataset = PubMedQADataset(
        split_path=cfg.data.split_path,
        folds=10,
        include_gold=True
    )
    
    if len(dataset) == 0:
        raise ValueError("No queries loaded from PubMedQA splits.")
    
    logging.info(f"Loaded {len(dataset)} queries from dataset")
    
    # Create collator with prompt builder
    def prompt_builder(query):
        return learner._build_prompt(query)
    
    collator = KDCollator(
        tokenizer=learner.stu_tok,
        prompt_builder=prompt_builder,
        max_length=cfg.model.get('max_length', 512)
    )
    
    # Create DataLoader
    batch_size= cfg.train.batch_size
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.train.get('num_workers', 0),
        pin_memory=cfg.train.get('pin_memory', True)
    )
    
    # Training configuration
    max_new_tokens = getattr(cfg.train, "max_gen_tokens", 128)
    eval_sample_size = getattr(cfg.train.logging, "eval_sample_size", 5)
    
    # Training loop
    for epoch in trange(cfg.train.epochs, desc=f"Training:"):
        logging.info(f"Epoch {epoch+1}/{cfg.train.epochs}")
        learner.student.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Batch Training")
        
        for batch in pbar:
            result = learner.train_step_autoregressive(batch)
            loss = result['avg_loss']
            token_gen = result['num_tokens']
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            logging.info(f"Batch: {num_batches}/{batch_size} loss: {loss:.4f} token_gen: {token_gen}")
            pbar.set_postfix({'loss': f'{loss:.4f}', 'token_gen': token_gen})
        
        avg_loss = total_loss / num_batches
        logging.info(f"Average Train Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{cfg.train.epochs} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate model at the end of each epoch
        if (epoch + 1) % cfg.train.logging.eval_every_steps == 0:
            learner.student.eval()
            metrics = evaluate_model(
                learner,
                dataset,
                sample_size=eval_sample_size,
                max_new_tokens=max_new_tokens,
            )
            logging.info(f"Evaluation Metrics at epoch {epoch+1}: {metrics}")
            print(f"Evaluation Metrics at epoch {epoch+1}: {metrics}")
            learner.student.train()
        
        # Save checkpoint
        if (epoch + 1) % cfg.train.logging.save_every_steps == 0:
            learner.save_student()
            logging.info(f"Student model saved at epoch {epoch+1}.")
            print(f"Student model saved at epoch {epoch+1}.")
    
    # Save final model
    learner.save_student()
    logging.info(f"Student model saved. Path: {cfg.train.logging.save_dir}")
    print(f"Student model saved. Path: {cfg.train.logging.save_dir}")


if __name__ == "__main__":
    train_model()