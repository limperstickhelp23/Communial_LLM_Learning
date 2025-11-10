import os
import torch
from tools import load_student, load_teacher, contrastive_loss

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
        self.optimizer = kwargs.get('optimizer', torch.optim.AdamW(self.student.parameters(), lr=self.cfg.train.optim.lr))

    def instantiate_models(self):
        if self.student is None or self.stu_tok is None:
            self.student, self.stu_tok = load_student(self.cfg.model.student)
            if self.stu_tok.pad_token_id is None:
                # Llama tokenizers usually do not define a PAD token; reuse EOS for generate()
                self.stu_tok.pad_token = self.stu_tok.eos_token
        if self.teacher is None:
            self.teacher = load_teacher(self.cfg.model.teacher)

    def teacher_ans(self, query: str):
        prompt = self._build_prompt(query)
        t_tokenized = self.stu_tok(prompt, return_tensors="pt").to(self._teacher_device)
        t_logits = self.teacher(**t_tokenized)
        self.t_logits = t_logits
        return t_logits

    def student_ans(self, query: str):
        prompt = self._build_prompt(query)
        s_tokenized = self.stu_tok(prompt, return_tensors="pt").to(self._student_device)
        s_logits = self.student(**s_tokenized)
        self.s_logits = s_logits
        return s_logits

    def train_step(self, query: str) -> torch.Tensor:
        with torch.no_grad():
            t_logits = self.teacher_ans(query)

        s_logits = self.student_ans(query)

        loss = contrastive_loss(
            s_logits.logits,
            t_logits.logits,
            self.cfg.train.loss.temperature,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def save_student(self):
        save_dir = self.cfg.train.logging.save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.student.save_pretrained(save_dir)

    def generate_answer(self, query: str, use_teacher: bool = False, max_new_tokens: int = 128) -> str:
        prompt = self._build_prompt(query)
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
        nodes = self.retriever.retrieve(query)
        context_blocks = []
        for node in nodes:
            if hasattr(node, "node") and node.node is not None:
                context_blocks.append(node.node.get_content())
            else:
                context_blocks.append(str(node))
        context_text = "\n\n".join(context_blocks)
        return self.prompt_template.format(context=context_text, question=query)

    @property
    def _student_device(self):
        return next(self.student.parameters()).device

    @property
    def _teacher_device(self):
        return next(self.teacher.parameters()).device
