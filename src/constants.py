from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    """Constants for the data set paths."""

    path_file_pdf_data_set = "assets/CONTRATO_AP000000718.pdf"
    extracted_text = "assets/extracted_text.pkl"
    embeddings_size = 768
    embeddings_path = "assets/embeddings.ann"


CONSTANTS = Constants()
