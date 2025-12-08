"""
Improved dataset splitting with customizable ratios and stratification.

Features:
- Customizable train/dev/test ratios
- Stratified splitting to maintain label distribution
- K-fold cross-validation with configurable K
- Deterministic results with seed
- Validation of splits
"""

import json
import os
import random
import shutil
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


class DatasetSplitter:
    """Handles dataset splitting with stratification and k-fold CV."""
    
    def __init__(self, seed: int = 0):
        self.seed = seed
        random.seed(seed)
    
    def stratified_split(
        self, 
        dataset: Dict, 
        ratios: List[float],
        label_key: str = 'final_decision'
    ) -> List[Dict]:
        """
        Split dataset into multiple subsets while maintaining label distribution.
        
        Args:
            dataset: Dictionary with pmid as keys
            ratios: List of ratios for each split (e.g., [0.7, 0.15, 0.15] for train/dev/test)
            label_key: Key to use for stratification
            
        Returns:
            List of dataset dictionaries
        """
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
        
        # Group by label
        label2pmids = defaultdict(list)
        for pmid, info in dataset.items():
            label = info.get(label_key, 'unknown')
            label2pmids[label].append(pmid)
        
        # Split each label group according to ratios
        label2splits = {}
        for label, pmids in label2pmids.items():
            random.shuffle(pmids)
            label2splits[label] = self._split_by_ratios(pmids, ratios)
        
        # Combine splits across labels
        num_splits = len(ratios)
        output = [dict() for _ in range(num_splits)]
        
        for label, splits in label2splits.items():
            for i, split_pmids in enumerate(splits):
                for pmid in split_pmids:
                    output[i][pmid] = dataset[pmid]
        
        return output
    
    def _split_by_ratios(self, items: List, ratios: List[float]) -> List[List]:
        """Split a list according to given ratios."""
        n = len(items)
        splits = []
        start_idx = 0
        
        for i, ratio in enumerate(ratios):
            if i == len(ratios) - 1:
                # Last split gets remaining items
                end_idx = n
            else:
                end_idx = start_idx + int(n * ratio)
            
            splits.append(items[start_idx:end_idx])
            start_idx = end_idx
        
        return splits
    
    def create_kfold_cv(
        self, 
        dataset: Dict, 
        k: int = 10,
        label_key: str = 'final_decision'
    ) -> List[Dict]:
        """
        Create k-fold cross-validation splits with stratification.
        
        Args:
            dataset: Dictionary with pmid as keys
            k: Number of folds
            label_key: Key to use for stratification
            
        Returns:
            List of k dataset dictionaries (dev sets)
        """
        # Create k equal folds
        ratios = [1.0 / k] * k
        return self.stratified_split(dataset, ratios, label_key)
    
    def combine_folds(self, folds: List[Dict], exclude_idx: int) -> Dict:
        """Combine all folds except the one at exclude_idx."""
        combined = {}
        for i, fold in enumerate(folds):
            if i != exclude_idx:
                combined.update(fold)
        return combined
    
    def validate_splits(self, splits: List[Dict], original: Dict):
        """Validate that splits are correct."""
        # Check all items are present
        total_items = sum(len(s) for s in splits)
        assert total_items == len(original), \
            f"Split size mismatch: {total_items} != {len(original)}"
        
        # Check no duplicates across splits
        all_pmids = set()
        for split in splits:
            split_pmids = set(split.keys())
            assert len(all_pmids & split_pmids) == 0, "Duplicate items across splits"
            all_pmids.update(split_pmids)
        
        # Check label distribution
        for i, split in enumerate(splits):
            label_counts = defaultdict(int)
            for info in split.values():
                label_counts[info.get('final_decision', 'unknown')] += 1
            print(f"Split {i} - Size: {len(split)}, Labels: {dict(label_counts)}")


def save_split_config(output_dir: str, config: Dict):
    """Save splitting configuration for reproducibility."""
    config_path = os.path.join(output_dir, 'split_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved split configuration to {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Split dataset with customizable ratios')
    parser.add_argument('split_name', choices=['pqal', 'pqaa'], 
                        help='Dataset to split')
    parser.add_argument('--test-ratio', type=float, default=0.5,
                        help='Ratio for test set (default: 0.5)')
    parser.add_argument('--k-folds', type=int, default=10,
                        help='Number of CV folds (default: 10)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--output-dir', type=str, default='../data',
                        help='Output directory (default: ../data)')
    
    args = parser.parse_args()
    
    splitter = DatasetSplitter(seed=args.seed)
    
    if args.split_name == 'pqal':
        # Load original dataset
        dataset = json.load(open('../data/ori_pqal.json'))
        print(f"Loaded {len(dataset)} samples from ori_pqal.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Split into CV and test sets
        cv_ratio = 1.0 - args.test_ratio
        cv_set, test_set = splitter.stratified_split(
            dataset, 
            ratios=[cv_ratio, args.test_ratio]
        )
        
        print(f"\nCreated CV set ({len(cv_set)} samples) and test set ({len(test_set)} samples)")
        
        # Save test set
        test_path = os.path.join(args.output_dir, 'test_set.json')
        with open(test_path, 'w') as f:
            json.dump(test_set, f, indent=2)
        print(f"Saved test set to {test_path}")
        
        # Create k-fold CV splits
        cv_folds = splitter.create_kfold_cv(cv_set, k=args.k_folds)
        splitter.validate_splits(cv_folds, cv_set)
        
        # Save each fold
        for i in range(args.k_folds):
            fold_dir = os.path.join(args.output_dir, f'pqal_fold{i}')
            
            # Clean and create directory
            if os.path.isdir(fold_dir):
                shutil.rmtree(fold_dir)
            os.makedirs(fold_dir)
            
            # Save dev set (validation for this fold)
            dev_path = os.path.join(fold_dir, 'dev_set.json')
            with open(dev_path, 'w') as f:
                json.dump(cv_folds[i], f, indent=2)
            
            # Save train set (all other folds combined)
            train_set = splitter.combine_folds(cv_folds, exclude_idx=i)
            train_path = os.path.join(fold_dir, 'train_set.json')
            with open(train_path, 'w') as f:
                json.dump(train_set, f, indent=2)
            
            print(f"Fold {i}: train={len(train_set)}, dev={len(cv_folds[i])}")
        
        # Save configuration
        save_split_config(args.output_dir, {
            'dataset': 'pqal',
            'total_samples': len(dataset),
            'test_ratio': args.test_ratio,
            'cv_ratio': cv_ratio,
            'k_folds': args.k_folds,
            'seed': args.seed,
            'test_size': len(test_set),
            'cv_size': len(cv_set)
        })
    
    elif args.split_name == 'pqaa':
        # Load original dataset
        dataset = json.load(open('../data/ori_pqaa.json'))
        print(f"Loaded {len(dataset)} samples from ori_pqaa.json")
        
        # Split with custom ratio (default 200k train, rest dev)
        train_ratio = min(200000 / len(dataset), 0.95)  # Cap at 95%
        dev_ratio = 1.0 - train_ratio
        
        train_set, dev_set = splitter.stratified_split(
            dataset,
            ratios=[train_ratio, dev_ratio]
        )
        
        print(f"Created train set ({len(train_set)} samples) and dev set ({len(dev_set)} samples)")
        
        # Save splits
        train_path = os.path.join(args.output_dir, 'pqaa_train_set.json')
        with open(train_path, 'w') as f:
            json.dump(train_set, f, indent=2)
        
        dev_path = os.path.join(args.output_dir, 'pqaa_dev_set.json')
        with open(dev_path, 'w') as f:
            json.dump(dev_set, f, indent=2)
        
        print(f"Saved train set to {train_path}")
        print(f"Saved dev set to {dev_path}")
        
        # Save configuration
        save_split_config(args.output_dir, {
            'dataset': 'pqaa',
            'total_samples': len(dataset),
            'train_ratio': train_ratio,
            'dev_ratio': dev_ratio,
            'seed': args.seed,
            'train_size': len(train_set),
            'dev_size': len(dev_set)
        })


if __name__ == "__main__":
    main()