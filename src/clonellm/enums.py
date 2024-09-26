from enum import Enum


class RagVectorStore(str, Enum):
    """Names of vector stores."""

    Chroma = "chroma"
    FAISS = "faiss"
    InMemory = "in-memory"
