import json
import os
import shutil
from typing import List
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate as Instantiate

from llama_index.core import (Document, VectorStoreIndex, StorageContext)
from llama_index.vector_stores.chroma import ChromaVectorStore




def setIndex(cfg)->List[VectorStoreIndex]:
    dataset = cfg.data.name
    num_indicies = cfg.num_pairs
    if dataset != "pubmed_qa":
        raise ValueError("This script only supports PubMedQA dataset.")

    # Collection paths for multiple indicies
    collection_paths = []
    base_path = cfg.rag.chroma_client.path
    for i in range(num_indicies):
        path = os.path.join(base_path, f'train_index_{i}')
        collection_paths.append(path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    indicies = []
    for i, cpath in enumerate(collection_paths):
        dataset_cl = loadPubmedQA(cfg.data.split_path, split=i)
        docs = getDocs(dataset_cl)
        print(f"Building index {i+1}/{num_indicies} at {cpath}...")
        index = createIndexer(cfg.rag, dataset, docs, cpath)
        index.storage_context.persist(persist_dir=cpath)
        indicies.append(index)

    ANSI_GREEN = "\u001b[32m"
    ANSI_RESET = "\u001b[0m"
    print(ANSI_GREEN, "✅ Indicies built and persisted to: \n", os.path.abspath(base_path), ANSI_RESET)

    return indicies


def createIndexer(cfg, dataset, docs, persist_path)->VectorStoreIndex:
    embed = Instantiate(cfg.embedder)
    client_cfg = OmegaConf.merge(cfg.chroma_client, OmegaConf.create({"path": persist_path}))
    client = Instantiate(client_cfg)
    collection = client.get_or_create_collection(name=dataset)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed
    )
    return index

def getDocs(dataset_cl)->list:
    docs = [
        Document(
            text= " ".join(ex["CONTEXTS"]), 
            metadata={"id": pubid, "query": ex["QUESTION"], "answer": ex["final_decision"], "long_ans": ex["LONG_ANSWER"]}
        )
    for pubid, ex  in dataset_cl.items()
    ]
    return docs

def loadPubmedQA(split_path, split=None)->list:
    if split is None:
        split_file = os.path.join(split_path, "pqal_fold0/train_set.json")
    else:
        split_file = os.path.join(split_path, f"pqal_fold{split}/dev_set.json")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, 'r') as f:
        dataset_cl = json.load(f)
    return dataset_cl


@hydra.main(config_path="../.conf", config_name="config", version_base="1.3")
def main(cfg):
    setIndex(cfg)


if __name__ == "__main__":
    main()
