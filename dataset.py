import json
import os
from torch.utils.data import Dataset
from typing import List, Dict, Optional


class PubMedQADataset(Dataset):
    
    def __init__(self, split_path: str, folds: int = 10, include_gold: bool = True):
        """
        Args:
            split_path: Path to the split directory
            folds: Number of folds to load
            include_gold: Whether to include gold answers in the dataset
        """
        self.queries = []
        self.gold_answers = []
        self.pubmed_ids = []
        
        for i in range(folds):
            fold_path = os.path.join(split_path, f"pqal_fold{i}/dev_set.json")
            with open(fold_path, 'r') as f:
                dataset_cl = json.load(f)
                for pubid, ex in dataset_cl.items():
                    self.queries.append(ex["QUESTION"])
                    self.pubmed_ids.append(pubid)
                    
                    if include_gold:
                        answer = ex.get("LONG_ANSWER") or ex.get("final_decision", "")
                        self.gold_answers.append(answer)
        
        self.include_gold = include_gold
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = {
            'query': self.queries[idx],
            'pubmed_id': self.pubmed_ids[idx]
        }
        if self.include_gold:
            item['gold_answer'] = self.gold_answers[idx]
        return item
    
    def get_queries(self) -> List[str]:
        """Return list of all queries."""
        return self.queries
    
    def get_gold_lookup(self) -> Dict[str, str]:
        """Return dictionary mapping queries to gold answers."""
        if not self.include_gold:
            return {}
        return {q: a for q, a in zip(self.queries, self.gold_answers)}