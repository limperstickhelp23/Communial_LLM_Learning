import os
import torch
from accelerate import Accelerator
from tools import load_student, load_teacher, kl_div_loss


class SageLearner:

    def __init__(self, cfg, vector_index, **kwargs):
        self.cfg = cfg
        self.vector_index = vector_index
        self.top_k = kwargs.get('top_k', cfg.model.teacher.top_k)
        self.prompt_template = kwargs.get('prompt_template', cfg.model.teacher.prompt_template)

        self.student = kwargs.get('student_model', None)
        self.teacher = kwargs.get('teacher_model', None)
        self.stu_tok = kwargs.get('stu_tokenizer', None)
        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)

        self.instantiate_models()
        self.optimizer = kwargs.get('optimizer', torch.optim.AdamW(
            self.student.parameters(), 
            lr=self.cfg.train.optim.lr
        ))
        print("mixed_precision:", cfg.train.get('mixed_precision', 'no'))
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train.get('gradient_accumulation_steps', 1),
            mixed_precision=cfg.train.get('mixed_precision', 'no'),  # 'fp16', 'bf16', or 'no'
        )
        # Log device info
        self.accelerator.print(f"Student model on: {self.accelerator.device}")
        self.accelerator.print(f"Number of processes: {self.accelerator.num_processes}")

    def instantiate_models(self):
        if self.student is None or self.stu_tok is None:
            self.student, self.stu_tok = load_student(self.cfg.model.student)
            if self.stu_tok.pad_token_id is None:
                self.stu_tok.pad_token = self.stu_tok.eos_token
        if self.teacher is None:
            self.teacher = load_teacher(self.cfg.model.teacher)

    def train_step_autoregressive(self, batch: dict) -> dict:
        """
        Args:
            batch: Dictionary with student/teacher inputs and attn masks
        
        Returns:
            Dictionary with:
                - avg_loss: Combined loss across all generation steps
                - num_tokens: Number of tokens generated
        """

        max_gen_tokens = getattr(self.cfg.train, "max_gen_tokens", 128)
        
        batch_size = batch['student_input_ids'].shape[0]
        # batch['teacher_input_ids'] = batch['teacher_input_ids'].to(self._teacher_device)
        # batch['teacher_attention_mask'] = batch['teacher_attention_mask'].to(self._teacher_device)

        # batch['student_input_ids'] = batch['student_input_ids'].to(self._student_device)
        # batch['student_attention_mask'] = batch['student_attention_mask'].to(self._student_device)
         
        # Track which sequences have finished (encountered EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self._student_device)
        
        total_loss = 0.0
        num_tokens_generated = 0
        
        # if torch.backends.mps.is_available():
        #     torch.mps.empty_cache()
        
        for step in range(max_gen_tokens):
            # Get teacher's next token prediction
            token_loss, teacher_next_token = self.train_step(batch, finished, False)
            
            total_loss += token_loss
            # if self._teacher_device != self._student_device:
            #     student_next_tokens = teacher_next_tokens.to(self._student_device)
            # else:
            student_next_token = teacher_next_token
            
            
            num_tokens_generated += 1
            
            # Update finished status
            is_eos = (teacher_next_token.squeeze(-1) == self.stu_tok.eos_token_id)
            finished = finished | is_eos
            
            # Stop if all sequences are finished
            if finished.all():
                break
            
            # Append teacher's token to both sequences for next iteration
            batch['teacher_input_ids'] = torch.cat([batch['teacher_input_ids'], teacher_next_token], dim=1)
            batch['teacher_attention_mask'] = torch.cat([
                batch['teacher_attention_mask'],
                torch.ones((batch_size, 1), device=self.accelerator.device)
            ], dim=1)
            
            batch['student_input_ids'] = torch.cat([batch['student_input_ids'], student_next_token], dim=1)
            batch['student_attention_mask'] = torch.cat([
                batch['student_attention_mask'],
                torch.ones((batch_size, 1), device=self.accelerator.device)
            ], dim=1)
        
        # Backward pass on accumulated loss
        avg_loss = total_loss / num_tokens_generated
        self.optimizer.zero_grad()
        self.accelerator.backward(avg_loss)
        # avg_loss.backward()
        # Gradient clipping
        if hasattr(self.cfg.train.optim, 'max_grad_norm'):
            self.accelerator.clip_grad_norm_(
                self.student.parameters(),
                self.cfg.train.optim.max_grad_norm
            )

        self.optimizer.step()
        
        # if torch.backends.mps.is_available():
        #     torch.mps.empty_cache()
            
        return {
            'total_loss': avg_loss.detach().item(),
            'num_tokens': num_tokens_generated,
            'avg_loss': avg_loss.detach().item()
        }
    
    def train_step(self, batch: dict, finished=None, backwards=True) -> torch.Tensor:
        """
        Process a batch of queries for next Token.
        
        Args:
            batch: Dictionary containing:
                - student_input_ids
                - student_attention_mask
                - teacher_input_ids
                - teacher_attention_mask
            finished:
                - track finished queries.
            backwards:
                - whether to perform backward pass
        Returns:
            Mean loss across the batch
            last token predicted by teacher
        """
        # Move inputs to appropriate devices
        teacher_inputs = {
            'input_ids': batch['teacher_input_ids'],#.to(self._teacher_device),
            'attention_mask': batch['teacher_attention_mask']#.to(self._teacher_device)
        }
        student_inputs = {
            'input_ids': batch['student_input_ids'],#.to(self._student_device),
            'attention_mask': batch['student_attention_mask']#.to(self._student_device)
        }

        # Get teacher logits (no gradients needed)
        with torch.no_grad():
            teacher_outputs = self.teacher(**teacher_inputs)
            t_logits = teacher_outputs.logits  # (batch_size, seq_len, vocab_size)
            
        # Get student logits
        student_outputs = self.student(**student_inputs)
        s_logits = student_outputs.logits  # (batch_size, seq_len, vocab_size)
        
        s_last_logits = s_logits[:, -1, :]
        t_last_logits = t_logits[:, -1, :]
        teacher_next_token = t_last_logits.argmax(dim=-1, keepdim=True).to(self._teacher_device) # greedy sample.

        if t_logits.device != self._student_device:
            t_logits = t_logits.to(self._student_device)

        # Shape: (batch_size, vocab_size)
        loss = kl_div_loss(
            s_last_logits,
            t_last_logits,
            self.cfg.train.loss.temperature,
        )

        # Mask out finished sequences
        if finished != None and finished.any():
            # Create a mask for active sequences
            active_mask = (~finished).float()
            loss = loss * active_mask.mean()
        
        # Backward pass
        if backwards:
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            # loss.backward()
            # Gradient clipping
            if hasattr(self.cfg.train.optim, 'max_grad_norm'):
                self.accelerator.clip_grad_norm_(
                    self.student.parameters(), 
                    self.cfg.train.optim.max_grad_norm
                )
        
            self.optimizer.step()
        
        return loss , teacher_next_token

    def teacher_ans(self, query: str):
        prompt = self._build_prompt(query)
        t_tokenized = self.stu_tok(prompt, return_tensors="pt").to(self._teacher_device)
        t_logits = self.teacher(**t_tokenized)
        return t_logits

    def student_ans(self, query: str):
        s_tokenized = self.stu_tok(query, return_tensors="pt").to(self._student_device)
        s_logits = self.student(**s_tokenized)
        return s_logits

    def save_student(self):
        save_dir = self.cfg.train.logging.save_dir
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.student)
        if self.accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            unwrapped_model.save_pretrained(save_dir)
            self.stu_tok.save_pretrained(save_dir)

    def generate_answer_batch(self, queries: list[str], use_teacher: bool = False, 
                             max_new_tokens: int = 128) -> list[str]:
        """
        Generate answers for a batch of queries.
        
        Args:
            queries: List of query strings
            use_teacher: Whether to use teacher model
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            List of generated answer strings
        """
        if use_teacher:
            prompts = [self._build_prompt(q) for q in queries]
        else:
            prompts = queries
        
        inputs = self.stu_tok(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(self._teacher_device if use_teacher else self._student_device)
        
        model = self.teacher if use_teacher else self.student
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.stu_tok.pad_token_id,
                eos_token_id=self.stu_tok.eos_token_id,
            )
        
        return [
            self.stu_tok.decode(ids, skip_special_tokens=True) 
            for ids in output_ids
        ]

    def generate_answer(self, query: str, use_teacher: bool = False, 
                       max_new_tokens: int = 128) -> str:
        prompt = self._build_prompt(query) if use_teacher else query
        inputs = self.stu_tok(prompt, return_tensors="pt").to(
            self._teacher_device if use_teacher else self._student_device
        )
        model = self.teacher if use_teacher else self.student
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.stu_tok.pad_token_id,
                eos_token_id=self.stu_tok.eos_token_id,
            )
        return self.stu_tok.decode(output_ids[0], skip_special_tokens=True)

    def _build_prompt(self, query: str) -> str:
        """Build RAG prompt for a single query."""
        nodes = self.retriever.retrieve(query)
        context_blocks = []
        for node in nodes:
            if hasattr(node, "node") and node.node is not None:
                context_blocks.append(node.node.get_content())
            else:
                print(f"NO CONTEXT for {query}")
                context_blocks.append(str(node))
        context_text = "\n\n".join(context_blocks)
        return self.prompt_template.format(context=context_text, question=query)
    
    def build_prompt_batch(self, queries: list[str]) -> list[str]:
        """Build RAG prompts for a batch of queries."""
        return [self._build_prompt(q) for q in queries]

    @property
    def _student_device(self):
        return next(self.student.parameters()).device

    @property
    def _teacher_device(self):
        return next(self.teacher.parameters()).device