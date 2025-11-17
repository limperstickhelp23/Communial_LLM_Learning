import torch
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class KDCollator: 
    def __init__(self, tokenizer, prompt_builder=None, max_length: int = 512):
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of queries into tensors.
        
        Args:
            batch: List of dicts with 'query' and optionally 'gold_answer'
        
        Returns:
            Dictionary with tokenized inputs for student and teacher
        """
        queries = [item['query'] for item in batch]
        
        # Tokenize student inputs (just the queries)
        student_inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Build teacher prompts if prompt_builder is provided
        if self.prompt_builder:
            teacher_prompts = [self.prompt_builder(q) for q in queries]
            teacher_inputs = self.tokenizer(
                teacher_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            # If no prompt builder, teacher uses same inputs as student
            teacher_inputs = student_inputs
        
        result = {
            'student_input_ids': student_inputs['input_ids'],
            'student_attention_mask': student_inputs['attention_mask'],
            'teacher_input_ids': teacher_inputs['input_ids'],
            'teacher_attention_mask': teacher_inputs['attention_mask'],
            'queries': queries
        }
        
        # Include gold answers if available
        if 'gold_answer' in batch[0]:
            result['gold_answers'] = [item['gold_answer'] for item in batch]
        
        return result