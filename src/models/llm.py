import os

from peft import (
    PeftConfig,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class MistralModel:
    instructions = (
        "Eres un sistema experto de preguntas y respuestas en el que se confía en todo el mundo.\n"
        "Siempre responde a la pregunta utilizando la información del contexto proporcionado, "
        "y no conocimiento previo.\n"
        "Algunas reglas a seguir:\n"
        "1. Nunca hagas referencia directa al contexto proporcionado en tu respuesta.\n"
        "2. Evita afirmaciones como 'Basándote en el contexto, ...' o "
        "'La información del contexto ...' o cualquier cosa similar.\n"
        "3. La respuesta debe estar escrita en Español\n"
        "4. Primero proporciona un razonamiento para encontrar la respuesta de manera concisa y sin dar información no necesaria.\n"
        "5. En el razonamiento, si necesitas copiar y pegar algunas frases del contexto, inclúyelas entre ##begin_quote## y ##end_quote##. "
        "Esto significaría que lo que esté fuera de ##begin_quote## y ##end_quote## no se copia y pega directamente del contexto.\n"
        "6. Termina tu respuesta con la respuesta final en la forma <ANSWER>: $respuesta, la respuesta debe ser concisa."
    )
    context_format = (
        "La información del contexto está a continuación.\n"
        "<contexto>\n"
        "{context_str}\n"
        "<contexto>\n"
        "Dada la información del contexto y sin conocimiento previo, "
        "responde la pregunta.\n"
        "Pregunta: {query_str}\n"
        "Respuesta: "
    )

    def __init__(self, model_base: str, pre_trained_model: str | None = None) -> None:
        self.model_base_name = model_base
        self.pre_trained_model = pre_trained_model

    def load_model(self) -> None:
        if self.pre_trained_model is not None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_base_name,
                device_map="auto",
                trust_remote_code=False,
                revision="main",
            )

            self.config = PeftConfig.from_pretrained(self.pre_trained_model)
            self.model = PeftModel.from_pretrained(model, self.pre_trained_model)

            # load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_base_name,
                use_fast=True,
            )
        else:
            model_name = self.model_base_name
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=False,
                revision="main",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def ask_model(self, question: str, context: str) -> str:
        self.model.eval()
        def format_prompt(instruction: str, context: str) -> str:
            return f"""<s>[INST] {instruction} \n{context} \n[/INST]\n""" + "</s>"
        prompt = format_prompt(
            instructions=self.instructions,
            context=self.context_format.format(context_str=context, query_str=question),
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            input_ids=inputs["input_ids"].to("cuda"),
            max_new_tokens=280,
        )
        return self.tokenizer.batch_decode(outputs)[0]
