import json
import os
import shutil
from typing import List, Optional
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate as Instantiate
from accelerate.logging import get_logger
import logging

from llama_index.core import (
    Document, 
    VectorStoreIndex, 
    StorageContext,
    load_index_from_storage
)
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = get_logger(__name__)

READY_FILE = ".READY"

class RAGIndexManager:
    """Manages creation, persistence, and loading of RAG indices."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_path = cfg.rag.client.path
        self.dataset_name = cfg.data.name
        self.num_indices = cfg.num_pairs
    
    def get_or_create_indices(self, force_rebuild: bool = False) -> List[VectorStoreIndex]:
        """
        Get existing indices or create new ones if they don't exist.
        Returns:
            List of VectorStoreIndex objects
        """
        if self.dataset_name != "pubmed_qa":
            raise ValueError("This script only supports PubMedQA dataset.")
        
        indices = []
        
        for i in range(self.num_indices):
            index_path = os.path.join(self.base_path, f'train_index_{i}')
            
            # Check if index exists and can be loaded
            if not force_rebuild and self._index_exists(index_path):
                logger.info(f"Loading existing index {i} from {index_path}")
                try:
                    index = self._load_index(index_path, i)
                    indices.append(index)
                    logger.info(f"✓ Successfully loaded index {i}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load index {i}: {e}. Rebuilding...")
            
            # Create new index
            logger.info(f"Building new index {i} at {index_path}")
            index = self._build_index(index_path, i)
            indices.append(index)
            logger.info(f"✓ Successfully built and persisted index {i}")

        
        logger.info(f"✅ All {len(indices)} indices ready at {os.path.abspath(self.base_path)}")
        return indices
    
    def _index_exists(self, index_path: str) -> bool:
        """Check if a persisted index exists. Checks for ready file"""
        return (os.path.isdir(index_path) and os.path.isfile(os.path.join(index_path, ".READY")))
    
    def _load_index(self, index_path: str, index_id: int) -> VectorStoreIndex:
        """Load a persisted index from disk."""
        # Set up the vector store
        client_cfg = OmegaConf.merge(
            self.cfg.rag.client, 
            OmegaConf.create({"path": index_path})
        )
        client = Instantiate(client_cfg)
        collection = client.get_or_create_collection(name=self.dataset_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        # Load storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=index_path
        )
        
        # Load the index
        embed_model = Instantiate(self.cfg.rag.embedder)
        index = load_index_from_storage(
            storage_context,
            embed_model=embed_model
        )
        
        return index
    
    def _build_index(self, index_path: str, index_id: int) -> VectorStoreIndex:
        """Build a new index from scratch."""
        # Clean up existing directory

        if os.path.isdir(index_path) and os.path.exists(index_path):
            shutil.rmtree(index_path)
        os.makedirs(index_path, exist_ok=True)
        
        # Load dataset for this index
        dataset_cl = self._load_dataset(index_id)
        docs = self._create_documents(dataset_cl)
        
        logger.info(f"Creating index with {len(docs)} documents")
        
        # Create index
        index = self._create_index(docs, index_path)
        
        # Persist to disk
        index.storage_context.persist(persist_dir=index_path)
        #file to flag as ready 
        with open(os.path.join(index_path, READY_FILE), 'w') as f:
            f.write("ready")
        return index
    
    def _load_dataset(self, split_id: int) -> dict:
        """Load dataset for a specific split."""
        split_file = os.path.join(
            self.cfg.data.split_path, 
            f"pqal_fold{split_id}/dev_set.json" # Double check this path
        )
        
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset)} examples from {split_file}")
        return dataset
    
    def _create_documents(self, dataset: dict) -> List[Document]:
        """Convert dataset to LlamaIndex documents."""
        docs = []
        for pmid, ex in dataset.items():
            # Combine contexts into single text
            text = " ".join(ex.get("CONTEXTS", []))
            
            # Create document with metadata
            doc = Document(
                text=text,
                metadata={
                    "id": pmid,
                    "query": ex.get("QUESTION", ""),
                    "answer": ex.get("final_decision", ""),
                    "long_ans": ex.get("LONG_ANSWER", "")
                }
            )
            docs.append(doc)
        
        return docs
    
    def _create_index(self, docs: List[Document], persist_path: str) -> VectorStoreIndex:
        """Create a vector store index."""

        embed_model = Instantiate(self.cfg.rag.embedder)
        
        client_cfg = OmegaConf.merge(
            self.cfg.rag.client, 
            OmegaConf.create({"path": persist_path})
        )
        client = Instantiate(client_cfg)
        collection = client.get_or_create_collection(name=self.dataset_name)
        

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        return index
    
    def rebuild_all(self):
        """Force rebuild all indices."""
        logger.info("Force rebuilding all indices...")
        return self.get_or_create_indices(force_rebuild=True)
    
    def clear_all(self):
        """Delete all persisted indices."""
        if os.path.isdir(self.base_path) and os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
            logger.info(f"Cleared all indices from {self.base_path}")
        else:
            logger.info("No indices to clear")


def setIndex(cfg, force_rebuild: bool = False) -> List[VectorStoreIndex]:
    """
    Main entry point for getting/creating indices.
    Returns:
        List of VectorStoreIndex objects
    """
    manager = RAGIndexManager(cfg)
    return manager.get_or_create_indices(force_rebuild=force_rebuild)


@hydra.main(config_path="../.conf", config_name="config", version_base="1.3")
def main(cfg):
    """
    Use --force-rebuild to rebuild all indices
    """
    import sys
    
    force_rebuild = '--force-rebuild' in sys.argv
    
    if force_rebuild:
        logger.info("Force rebuild flag detected")
    
    manager = RAGIndexManager(cfg)
    
    if '--clear' in sys.argv:
        manager.clear_all()
        logger.info("Indices cleared. Exiting.")
        return
    
    indices = manager.get_or_create_indices(force_rebuild=force_rebuild)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RAG Index Preparation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total indices: {len(indices)}")
    logger.info(f"Storage path: {os.path.abspath(cfg.rag.client.path)}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()