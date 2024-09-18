from typing import Any, Optional, cast

from langchain_core.embeddings import Embeddings
from litellm import aembedding, all_embedding_models, embedding

from ._base import LiteLLMMixin

__all__ = ("LiteLLMEmbeddings",)


class LiteLLMEmbeddings(LiteLLMMixin, Embeddings):
    """A class that uses LiteLLM to call an LLM's API to generate embeddings for the given input.

    Args:
        model (str): The embedding model to use.
        api_key (Optional[str]): The API key to use. Defaults to None.
        dimensions (Optional[int]): The number of dimensions the resulting output embeddings should have. Defaults to None.
        **kwargs: Additional keyword arguments supported by the `litellm.embedding` and `litellm.aembedding` functions.

    """

    def __init__(self, model: str, api_key: Optional[str] = None, dimensions: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(model, api_key, **kwargs)
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Call out to LLM's embedding endpoint for embedding a list of documents.

        Args:
            texts (list[str]): The list of texts to embed.

        Returns:
            list[list[float]]: List of embeddings, one for each text.
        """
        response = embedding(
            model=self.model, input=texts, api_key=self.api_key, dimensions=self.dimensions, **self._litellm_kwargs
        )
        return [r["embedding"] for r in response.data]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Call out to LLM's embedding endpoint async for embedding a list of documents.

        Args:
            texts (list[str]): The list of texts to embed.

        Returns:
            list[list[float]]: List of embeddings, one for each text.
        """
        response = await aembedding(
            model=self.model, input=texts, api_key=self.api_key, dimensions=self.dimensions, **self._litellm_kwargs
        )
        return [r["embedding"] for r in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Call out to LLM's embedding endpoint for embedding query text.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: Embedding for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        """Call out to LLM's embedding endpoint async for embedding query text.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: Embedding for the text.
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0]

    @property
    def all_embedding_models(self) -> list[str]:
        """
        Returns the names of supported embedding models.
        """
        return cast(list[str], all_embedding_models)

    def __repr__(self) -> str:
        return (
            "LiteLLMEmbeddings<("
            + f"model='{self.model}'"
            + (f", dimensions={self.dimensions}" if self.dimensions else "")
            + ")>"
        )
