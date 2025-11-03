import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tools import load_student

class SageLearner:
    
    def __init__(self, cfg, vector_index, **kwargs):
        self.cfg = cfg
        self.vector_index = vector_index
        # Initialize other components as needed
        self.student = kwargs.get('student_model', None)
        self.teacher = kwargs.get('teacher_model', None)
        self.stok = kwargs.get('stu_tokenizer', None)
        self.top_k = kwargs.get('top_k', cfg.model.teacher.top_k)
        self.prompt_template = kwargs.get('prompt_template', cfg.model.teacher.prompt_template)
        self.instantiate_models()

    def instantiate_models(self):
        if self.student is None or self.stu_tok is None:
            self.student, self.stu_tok = load_student(self.cfg.model.student)
        if self.teacher is None:
            self.teacher = self.cfg.model.teacher.ollama.model

    def teacher_ans(self, query: str) -> dict:
        context = self.vector_index.retriever(query, top_k=self.top_k)
        prompt = self.prompt_template.format(context=context, question=query)
        response = ollama.response(self.teacher, prompt)
        self.t_response = response
        return response
    
    def student_ans(self, inputs: dict) -> dict:
        # TODO: Implement student answer generation
        pass
