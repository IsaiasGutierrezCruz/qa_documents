import random
from pathlib import Path
from typing import Literal

from annoy import AnnoyIndex
from llama_index.core.schema import Document, TextNode
from llama_index.readers.docling import DoclingReader

from src.constants import CONSTANTS
from src.models.embeddings import get_embeddings, normalize_vector, store_embeddings
from src.repositories.chunk_strategies import ChunkingStrategy
from src.utils.file_management import load_data, save_data


class Index:
    """Class to index the text."""

    def __init__(self, db: AnnoyIndex, map_context: dict[int, str]) -> None:
        self.db = db
        self.map_context = map_context

    def get_context(self, question: int) -> str:
        """Get the context of the question."""
        similar_context = self.db.get_nns_by_vector(
            normalize_vector(get_embeddings([question]))[0],
            2,
        )
        return "\n".join([self.map_context[i] for i in similar_context])


class TextExtractor:
    """
    Class to extract text from a file using a specific strategy of OCR.

    Parameters
    ----------
    file_source : Path
        The path to the file to extract text from.
    strategy : Literal["docling"]
        The strategy to use for extracting text.
    """

    def __init__(
        self,
        root_path: Path,
        strategy: Literal["docling"],
        chunk_strategy: ChunkingStrategy,
        path_pdf_file: Path,
    ) -> None:
        self.root_path = root_path
        self.strategy = strategy
        self.chunk_strategy = chunk_strategy
        self.path_pdf_file = path_pdf_file

        self.chunks: list[TextNode] = []
        self.data: list[Document] = []

    def _extract_text(self) -> list[Document]:
        """
        Extract text from a file using a specific strategy of OCR.

        Only supported strategy "docling" for now.

        Parameters
        ----------
        file_source : str
            The path to the file to extract text from.
        strategy : Literal["docling"]
            The strategy to use for extracting text.

        Returns
        -------
            list[Document]: A list of documents containing the extracted text.
        """
        if (self.root_path / CONSTANTS.extracted_text).is_file():
            self.data = load_data(self.root_path / CONSTANTS.extracted_text)
        else:
            pdf_path = self.root_path / self.path_pdf_file
            if pdf_path.is_file():
                if self.strategy == "docling":
                    reader = DoclingReader()
                    data = reader.load_data(pdf_path)
                else:
                    msg = f"Strategy {self.strategy} not supported"
                    raise ValueError(msg)
            else:
                msg = f"File {pdf_path} does not exist"
                raise FileNotFoundError(msg)
            save_data(data, self.root_path / CONSTANTS.extracted_text)

            self.data = data

    def _get_chucks(self) -> None:
        """Get chunks from the text."""
        self._extract_text()
        if self.data is None:
            msg = "Data is not loaded"
            raise ValueError(msg)
        self.chunks = self.chunk_strategy.get_chunks(self.data)
        random.seed(42)
        random.shuffle(self.chunks)

    def get_index(self) -> tuple[AnnoyIndex, dict[int, str]]:
        """Get the index of the chunks."""
        self._get_chucks()
        contexts = [text.text for text in self.chunks]
        map_context = dict(zip(range(len(contexts)), contexts, strict=True))
        if not (self.root_path / CONSTANTS.embeddings_path).is_file():
            embeddings = get_embeddings(contexts)
            store_embeddings(embeddings, path=self.root_path / CONSTANTS.embeddings_path)
        ann = AnnoyIndex(CONSTANTS.embeddings_size, "angular")
        _ = ann.load(str(self.root_path / CONSTANTS.embeddings_path))
        return Index(db=ann, map_context=map_context)
