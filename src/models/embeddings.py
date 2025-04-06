import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2", cache_folder="assets"
)


def get_embeddings(sentences: list[str]):
    return model.encode(sentences)

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v  # Or handle as a zero vector appropriately
    return v / norm


def store_embeddings(
    embeddings: np.ndarray,
    path: str,
    number_of_trees: int = 10,
) -> None:
    embedding_size = embeddings.shape[1]

    ann = AnnoyIndex(embedding_size, metric="angular")

    for idx, item_vector in enumerate(embeddings):
        id_item = idx
        ann.add_item(id_item, normalize_vector(item_vector))

    ann.build(number_of_trees)
    ann.save(str(path))