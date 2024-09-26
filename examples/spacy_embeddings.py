import asyncio
from typing import Any

import spacy
import spacy.cli
from langchain_core.embeddings import Embeddings
from spacy.language import Language

# !pip install spacy


class SpacyEmbeddings(Embeddings):
    """A class that uses spaCy to generate embeddings for the given input.

    Args:
        model (str): The spaCy model to use, e.g., 'en_core_web_md'.
        **kwargs: Additional keyword arguments supported by spaCy's `load` function.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self._spacy_kwargs = kwargs
        self._internal_init()

    def _internal_init(self):
        self._nlp: Language
        try:
            self._nlp = spacy.load(self.model, **self._spacy_kwargs)
        except OSError:
            print(f"spaCy model '{self.model}' not found. Downloading now...")
            spacy.cli.download(self.model)
            self._nlp = spacy.load(self.model, **self._spacy_kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Call spaCy's model to generate embeddings for a list of documents.

        Args:
            texts (list[str]): The list of texts to embed.

        Returns:
            list[list[float]]: List of embeddings, one for each text.
        """
        return [self._nlp(text).vector.tolist() for text in texts]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously generate embeddings for a list of documents using spaCy's model.

        Args:
            texts (List[str]): The list of texts to embed.

        Returns:
            list[list[float]]: List of embeddings, one for each text.
        """
        return await asyncio.to_thread(self.embed_documents, texts)

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: Embedding for the text.
        """
        return self._nlp(text).vector.tolist()

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously generate embedding for a single query text.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: Embedding for the text.
        """
        return await asyncio.to_thread(self.embed_query, text)

    def __repr__(self) -> str:
        return f"SpacyEmbeddings<(model='{self.model}')>"
