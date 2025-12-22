import hydra
import time 
import os
import logging
import torch

from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm, trange
from eval import evaluate_model
from sagelearner import SageLearner
from logging import getLogger
from dataset import PubMedQADataset
from collator import KDCollator
from tools import is_main_process, get_rank, get_world_size, setup_ddp, clean_up_ddp
# from data.prepare_dataset import setIndex --- IGNORE(legacy) ---
from data.setup_dataset import setIndex

@hydra.main(config_path=".conf/", config_name="config", version_base="1.3")
def train_model(cfg):
    verbose = cfg.get('verbose', False)
    use_ddp = cfg.get('use_ddp', False)
    # Setup logging

    if is_main_process():
        log = getLogger("mainLearner")
    else:
        log = getLogger("lowerLearner")
        log.setLevel(logging.ERROR)
    
    class MainFilter(logging.Filter):
        def filter(self, record):
            return dist.get_rank() == 0
    mf = MainFilter()
    log.addFilter(mf)
    log.info("="*30)
    log.info("Starting RAG Student-Teacher")
    log.info("-"*30)
    # Setup indices - automatically loads from disk if they exist
    log.info("Setting up RAG indices...")
    force_rebuild = getattr(cfg.rag, 'force_rebuild_indices', False)

    if use_ddp:
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

        setup_ddp(rank, local_rank, world_size)
        dist.barrier()
    else:
        rank = 0
        world_size = None
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Setting up indices
    try:
        if is_main_process():
            indices = setIndex(cfg, force_rebuild=force_rebuild)
        if use_ddp:
            dist.barrier()
        if not is_main_process():
            indices = setIndex(cfg, force_rebuild=False)
        
        log.info(f"RAG indices ready ({len(indices)} indices)")
    except Exception as e:
        log.error(f"Failed to setup indices: {e}")
        if dist.is_initialized(): 
            log.info("Destroying process group due to setup failure.")
            clean_up_ddp()
        raise
    
    vector_index = indices[0]  # Use the first index for retrieval
    
    # Initialize learner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    log.info("Initializing SageLearner...")
    learner = SageLearner(cfg, vector_index, rank=rank, world_size=world_size)
    log.info("SageLearner initialized")

    
    # Create dataset
    dataset = PubMedQADataset(
        split_path=cfg.data.split_path,
        folds=cfg.data.get('folds', 1),
        include_gold=True
    )
    
    if len(dataset) == 0:
        raise ValueError("No queries loaded from PubMedQA splits.")
    
    log.info(f"Loaded {len(dataset)} queries from dataset")
    
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
    smapler = None
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=cfg.train.get('num_workers', 0)
    )

    total_batches = train_loader.__len__()
    
    # Training configuration
    max_new_tokens = getattr(cfg.train, "max_gen_tokens", 128)
    eval_sample_size = getattr(cfg.train.logging, "eval_sample_size", 5)
    log.info("-"*30)
    log.info("Starting Training")
    log.info("-"*30)
    log.info("Training Configuration")
    log.info(">"*30)
    log.info(f"Epochs: {cfg.train.epochs}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Train size: {len(dataset)}")
    log.info(f"Max generation tokens: {max_new_tokens}")
    log.info(f"Learning rate: {cfg.train.optim.lr}")
    log.info(f"Eval sample size: {eval_sample_size}")
    log.info(f"Save directory: {cfg.train.logging.save_dir}")
    log.info("-"*30)

    start_time = time.perf_counter()
    # Training loop
    for epoch in trange(cfg.train.epochs, desc=f"Training:", disable=(not is_main_process() or verbose)):
        log.info(f"Epoch {epoch+1}/{cfg.train.epochs}")
        learner.student.train()
        sampler.set_epoch(epoch) if sampler else None
        
        total_loss = 0.0
        num_batches = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Batch Training", disable=(not is_main_process() and not verbose))
        
        for batch in pbar:
            start = time.perf_counter()
            result = learner.train_step_autoregressive(batch)
            elapsed = time.perf_counter() - start
            loss = result['avg_loss']
            token_gen = result['num_tokens']
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            log.info(f"Batch: {num_batches}/{total_batches} loss: {loss:.4f} token_gen: {token_gen} time: {elapsed:.2f}s") if verbose else None
            pbar.set_postfix({'loss': f'{loss:.4f}', 'token_gen': token_gen})
        
        avg_loss = total_loss / num_batches

        if use_ddp:
            # Gather losses from all processes
            avg_loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / world_size

        log.info(f"Average Train Loss: {avg_loss:.4f} for epoch {epoch+1}")
        # print(f"Epoch {epoch+1}/{cfg.train.epochs} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate model at the end of each epoch
        if (epoch + 1) % cfg.train.logging.eval_every_steps == 0 :
            if is_main_process():
                learner.student.eval()
                metrics = evaluate_model(
                    learner,
                    dataset,
                    sample_size=eval_sample_size,
                    max_new_tokens=max_new_tokens,
                )
                log.info(f"Evaluation Metrics at epoch {epoch+1}: {metrics}")
                learner.student.train()

            if use_ddp:
                dist.barrier()

        # Save checkpoint
        if (epoch + 1) % cfg.train.logging.save_every_steps == 0:
            learner.save_student()
            log.info(f"Student model saved at epoch {epoch+1}.")

    elapsed_time = time.perf_counter() - start_time
    log.info(f"Training completed in {elapsed_time/60:.2f} minutes.")
    log.info("="*30)
    # Save final model
    learner.save_student()
    log.info(f"Student model saved. Path: {cfg.train.logging.save_dir}")

    if use_ddp:
        clean_up_ddp()


if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt as k:
        logging.log.warning("Training interrupted by user.")
    finally:
        if dist.is_initialized():
            clean_up_ddp()
    
