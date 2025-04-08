from pathlib import Path

from src.models.llm import MistralModel
from src.repositories.chunk_strategies import ChunkingStrategySentence
from src.repositories.data_extraction import TextExtractor


def main() -> None:
    here = Path(__name__).parent
    text_extractor = TextExtractor(
        root_path=here,
        strategy="docling",
        chunk_strategy=ChunkingStrategySentence(),
        path_pdf_file="assets/CONTRATO_AP000000718.pdf",
    )
    index = text_extractor.get_index()

    question = "¿Cuál es el plazo para que el Arrendatario pague el precio de venta del Equipo al Arrendador después de la expiración del Plazo Básico?"
    mistral_model = MistralModel(
        model_base="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        pre_trained_model="isaiasgutierrezcruz/qa_documents_ft_val",
    )
    mistral_model.load_model()
    mistral_model.ask_model(question=question, context=index.get_context(question))


if __name__ == "__main__":
    main()
