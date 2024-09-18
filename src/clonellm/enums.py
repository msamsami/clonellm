from enum import Enum


class RagVectorStore(str, Enum):
    Chroma = "chroma"
    FAISS = "faiss"
