import functools
import json
import logging
import sys
import uuid
from importlib.util import find_spec
from operator import itemgetter
from typing import Any, AsyncIterator, Iterator, Optional, cast

from langchain.text_splitter import CharacterTextSplitter, TextSplitter
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.vectorstores import FAISS, Chroma, InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from litellm import models_by_provider
from pydantic import BaseModel

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._base import LiteLLMMixin
from ._prompt import (
    context_prompt,
    history_prompt,
    question_prompt,
    summarize_context_prompt,
    user_profile_prompt,
)
from .enums import RagVectorStore
from .memory import clear_session_history, get_session_history, get_session_history_size
from .models import UserProfile

logging.getLogger("langchain_core").setLevel(logging.ERROR)

__all__ = ("CloneLLM",)


class CloneLLM(LiteLLMMixin):
    """Creates an LLM clone of a user based on provided user profile and related context.

    Args:
        model (str): Name of the language model.
        documents (list[Document | str]): List of documents or strings related to cloning user to use for LLM context.
        embedding (Optional[Embeddings]): The embedding function to use for RAG. Defaults to None for no embedding, i.e., a summary of `documents` is used for RAG.
        vector_store (Optional[str | RagVectorStore]): The vector store to use for embedding-based retrieval. Defaults to None for "in-memory" vector store.
        user_profile (Optional[UserProfile | dict[str, Any] | str]): The profile of the user to be cloned by the language model. Defaults to None.
        memory (Optional[bool | int]): Maximum number of messages in conversation memory. Defaults to None (or 0) for no memory. -1 or `True` means infinite memory.
        api_key (Optional[str]): The API key to use. Defaults to None.
        **kwargs (Any): Additional keyword arguments supported by the `langchain_community.chat_models.ChatLiteLLM` class.

    """

    _VECTOR_STORE_COLLECTION_NAME = "clonellm"
    _FROM_CLASS_METHOD_KWARG = "_from_class_method"
    _DEFAULT_VECTOR_STORE = RagVectorStore.InMemory
    _TEXT_SPLITTER_CHUNK_SIZE = 2000

    def __init__(
        self,
        model: str,
        documents: list[Document | str],
        embedding: Optional[Embeddings] = None,
        vector_store: Optional[str | RagVectorStore] = None,
        user_profile: Optional[UserProfile | dict[str, Any] | str] = None,
        memory: Optional[bool | int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.embedding = embedding
        self.vector_store = vector_store
        self.documents = documents
        self.user_profile = user_profile
        self.memory = memory

        from_class_method: Optional[dict[str, Any]] = kwargs.pop(self._FROM_CLASS_METHOD_KWARG, None)
        super().__init__(model, api_key, **kwargs)
        self._internal_init(from_class_method)

    @property
    def _vector_store(self) -> str:
        return (self.vector_store or self._DEFAULT_VECTOR_STORE).lower().strip()

    def _check_dependencies(self) -> None:
        if self.embedding:
            if self._vector_store == RagVectorStore.Chroma:
                if not find_spec("chromadb"):
                    raise ImportError(
                        "Could not import chromadb. "
                        "Please install CloneLLM with `pip install clonellm[chroma]` "
                        "to use Chroma vector store for RAG."
                    )
            elif self._vector_store == RagVectorStore.FAISS:
                if not find_spec("faiss"):
                    raise ImportError(
                        "Could not import faiss-cpu. "
                        "Please install CloneLLM with `pip install clonellm[faiss]` "
                        "to use FAISS vector store for RAG."
                    )
            elif self._vector_store == RagVectorStore.InMemory:
                pass
            else:
                raise ValueError(f"Unsupported vector store '{self.vector_store}' provided.")

    def _internal_init(self, from_class_method: Optional[dict[str, Any]] = None) -> None:
        self._splitter: TextSplitter
        self.context: str
        self.db: VectorStore
        self._is_fitted: bool = False
        self._session_id: str = ""

        self._check_dependencies()
        self._litellm_kwargs.update({f"{self._llm_provider}_api_key": self.api_key})
        self._llm = ChatLiteLLM(model=self.model, model_name=self.model, **self._litellm_kwargs)
        if self.embedding:
            self._splitter = CharacterTextSplitter(chunk_size=self._TEXT_SPLITTER_CHUNK_SIZE, chunk_overlap=100)
        self.clear_memory()

        if from_class_method:
            for attr, value in from_class_method.items():
                setattr(self, attr, value)

    @classmethod
    def from_persist_directory(
        cls,
        model: str,
        chroma_persist_directory: str,
        embedding: Optional[Embeddings] = None,
        user_profile: Optional[UserProfile | dict[str, Any] | str] = None,
        memory: Optional[bool | int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Self:
        """Creates an instance of CloneLLM by loading a Chroma vector store from a persistent directory.

        Args:
            model (str): Name of the language model.
            chroma_persist_directory (str): Directory path to the persisted Chroma vector store.
            embedding (Optional[Embeddings]): The embedding function to use for Chroma store. Defaults to None for no embedding, i.e., a summary of `documents` is used for RAG.
            user_profile (Optional[UserProfile | dict[str, Any] | str]): The profile of the user to be cloned by the language model. Defaults to None.
            memory (Optional[bool | int]): Maximum number of messages in conversation memory. Defaults to None (or 0) for no memory. -1 or `True` means infinite memory.
            api_key (Optional[str]): The API key to use. Defaults to None.
            **kwargs (Any): Additional keyword arguments supported by the `langchain_community.chat_models.ChatLiteLLM` class.

        Returns:
            CloneLLM: An instance of CloneLLM with Chroma-based retrieval.
        """
        kwargs.update(
            {
                cls._FROM_CLASS_METHOD_KWARG: {
                    "db": Chroma(
                        collection_name=cls._VECTOR_STORE_COLLECTION_NAME,
                        embedding_function=embedding,
                        persist_directory=chroma_persist_directory,
                    ),
                    "_is_fitted": True,
                }
            }
        )
        return cls(
            model=model,
            documents=[],
            embedding=embedding,
            vector_store=RagVectorStore.Chroma,
            user_profile=user_profile,
            memory=memory,
            api_key=api_key,
            **kwargs,
        )

    @classmethod
    def from_context(
        cls,
        model: str,
        context: str,
        user_profile: Optional[UserProfile | dict[str, Any] | str] = None,
        memory: Optional[bool | int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Self:
        """Creates an instance of CloneLLM using a summarized context string instead of documents.

        Args:
            model (str): Name of the language model.
            context (str): Pre-summarized context string for the language model.
            user_profile (Optional[UserProfile | dict[str, Any] | str]): The profile of the user to be cloned by the language model. Defaults to None.
            memory (Optional[bool | int]): Maximum number of messages in conversation memory. Defaults to None (or 0) for no memory. -1 or `True` means infinite memory.
            api_key (Optional[str]): The API key to use. Defaults to None.
            **kwargs (Any): Additional keyword arguments supported by the `langchain_community.chat_models.ChatLiteLLM` class.

        Returns:
            CloneLLM instance using the provided context.
        """
        kwargs.update({cls._FROM_CLASS_METHOD_KWARG: {"context": context, "_is_fitted": True}})
        return cls(
            model=model,
            documents=[],
            user_profile=user_profile,
            memory=memory,
            api_key=api_key,
            **kwargs,
        )

    def _get_documents(self, documents: Optional[list[Document | str]] = None) -> list[Document]:
        if not (documents or self.documents):
            raise ValueError("No documents provided")
        documents_ = []
        for i, doc in enumerate(documents or self.documents):
            if not isinstance(doc, (Document, str)):
                raise ValueError(f"item at index {i} is not a valid Document or a string")
            documents_.append(Document(page_content=doc) if isinstance(doc, str) else doc)
        return documents_

    def _get_summarized_context(self, documents: list[Document]) -> str:
        documents_str = "\n\n".join([doc.page_content for doc in documents])
        chain: RunnableSerializable[Any, str] = (
            {"input": RunnablePassthrough()} | summarize_context_prompt | self._llm | StrOutputParser()
        )
        return chain.invoke(documents_str)

    def fit(self) -> Self:
        """Fits the CloneLLM instance by processing the provided documents.

        Embeds the documents for retrieval using the selected vector store or generates a summarized context.

        Returns:
            CloneLLM: Fitted CloneLLM instance.
        """
        documents = self._get_documents()
        if self.embedding:
            documents = self._splitter.split_documents(documents)
            if self._vector_store == RagVectorStore.Chroma:
                self.db = Chroma.from_documents(documents, self.embedding, collection_name=self._VECTOR_STORE_COLLECTION_NAME)
            elif self._vector_store == RagVectorStore.FAISS:
                self.db = FAISS.from_documents(documents, self.embedding)
            elif self._vector_store == RagVectorStore.InMemory:
                self.db = InMemoryVectorStore.from_documents(documents, self.embedding)
        else:
            self.context = self._get_summarized_context(documents)
        self._is_fitted = True
        return self

    async def _aget_summarized_context(self, documents: list[Document]) -> str:
        documents_str = "\n\n".join([doc.page_content for doc in documents])
        chain: RunnableSerializable[Any, str] = (
            {"input": RunnablePassthrough()} | summarize_context_prompt | self._llm | StrOutputParser()
        )
        return await chain.ainvoke(documents_str)

    async def afit(self) -> Self:
        """Asynchronously fits the CloneLLM instance by processing the provided documents.

        Embeds the documents for retrieval or generates a summarized context.

        Returns:
            CloneLLM: Fitted CloneLLM instance.
        """
        documents = self._get_documents()
        if self.embedding:
            documents = self._splitter.split_documents(documents)
            if self._vector_store == RagVectorStore.Chroma:
                self.db = await Chroma.afrom_documents(
                    documents, self.embedding, collection_name=self._VECTOR_STORE_COLLECTION_NAME
                )
            elif self._vector_store == RagVectorStore.FAISS:
                self.db = await FAISS.afrom_documents(documents, self.embedding)
            elif self._vector_store == RagVectorStore.InMemory:
                self.db = await InMemoryVectorStore.afrom_documents(documents, self.embedding)
        else:
            self.context = await self._aget_summarized_context(documents)
        self._is_fitted = True
        return self

    def _check_is_fitted(self, from_update: bool = False) -> None:
        if (
            not self._is_fitted
            or (
                self.embedding
                and (
                    not hasattr(self, "db")
                    or self.db is None
                    or ((not hasattr(self, "_splitter") or self._splitter is None) if from_update else False)
                )
            )
            or (not self.embedding and (not hasattr(self, "context") or self.context is None))
        ):
            raise AttributeError("This CloneLLM instance is not fitted yet. Call `fit` using this method.")

    def update(self, documents: list[Document | str]) -> Self:
        """Updates the CloneLLM with additional documents, either embedding them or updating the context.

        Args:
            documents (list[Document | str]): Additional documents to add to the model.

        Returns:
            CloneLLM: Updated CloneLLM instance.
        """
        self._check_is_fitted(from_update=True)
        documents_ = self._get_documents(documents)
        if self.embedding:
            documents_ = self._splitter.split_documents(documents_)
            self.db.add_documents(documents_)
        else:
            self.context = self._get_summarized_context(documents_)
        return self

    async def aupdate(self, documents: list[Document | str]) -> Self:
        """Asynchronously updates the CloneLLM with additional documents, either embedding them or updating the context.

        Args:
            documents (list[Document | str]): Additional documents to add to the model.

        Returns:
            CloneLLM: Updated CloneLLM instance.
        """
        self._check_is_fitted(from_update=True)
        documents_ = self._get_documents(documents)
        if self.embedding:
            documents_ = self._splitter.split_documents(documents_)
            await self.db.aadd_documents(documents_)
        else:
            self.context = await self._aget_summarized_context(documents_)
        return self

    @property
    def _user_profile(self) -> str:
        if isinstance(self.user_profile, BaseModel):
            return self.user_profile.model_dump_json(indent=4, exclude_none=True)
        elif isinstance(self.user_profile, dict):
            return json.dumps(self.user_profile, indent=4, default=str)
        return str(self.user_profile)

    def _get_retriever(self, k: int = 1) -> VectorStoreRetriever:
        return self.db.as_retriever(search_kwargs={"k": k})

    def _get_rag_chain(self) -> RunnableSerializable[Any, str]:
        prompt = context_prompt.copy()
        if self.user_profile:
            prompt += user_profile_prompt.format_messages(user_profile=self._user_profile)
        prompt += question_prompt

        context = self._get_retriever() if self.embedding else lambda x: self.context
        return {"context": context, "input": RunnablePassthrough()} | prompt | self._llm | StrOutputParser()

    def _get_rag_chain_with_history(self) -> RunnableWithMessageHistory:
        prompt = context_prompt
        if self.user_profile:
            prompt += user_profile_prompt.format_messages(user_profile=self._user_profile)
        prompt += history_prompt
        prompt += question_prompt

        context = itemgetter("input") | self._get_retriever() if self.embedding else lambda x: self.context
        first_step = RunnablePassthrough.assign(context=context)  # type: ignore[arg-type]
        rag_chain = first_step | prompt | self._llm | StrOutputParser()

        if not self.memory:
            max_memory_size = 0
        elif (isinstance(self.memory, bool) and self.memory) or self.memory == -1:
            max_memory_size = -1
        else:
            max_memory_size = int(self.memory)

        get_session_history_ = functools.partial(get_session_history, max_memory_size=max_memory_size)

        return RunnableWithMessageHistory(
            rag_chain,  # type: ignore[arg-type]
            get_session_history_,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_parser=StrOutputParser(),
        )

    def invoke(self, prompt: str) -> str:
        """Invokes the cloned language model and generates a response based on the given prompt.

        This method uses the underlying language model to simulate responses as if coming from the cloned user profile.

        Args:
            prompt (str): Input prompt for the cloned language model.

        Returns:
            str: The generated response from the language model as the cloned user.
        """
        self._check_is_fitted()
        if self.memory:
            rag_chain_with_history = self._get_rag_chain_with_history()
            response = rag_chain_with_history.invoke({"input": prompt}, config={"configurable": {"session_id": self._session_id}})
            return cast(str, response)
        rag_chain = self._get_rag_chain()
        return rag_chain.invoke(prompt)

    async def ainvoke(self, prompt: str) -> str:
        """Asynchronously invokes the cloned language model and generates a response based on the given prompt.

        This method uses the underlying language model to simulate responses as if coming from the cloned user profile.

        Args:
            prompt (str): Input prompt for the cloned language model.

        Returns:
            str: The generated response from the language model as the cloned user.
        """
        self._check_is_fitted()
        if self.memory:
            rag_chain_with_history = self._get_rag_chain_with_history()
            response = await rag_chain_with_history.ainvoke(
                {"input": prompt}, config={"configurable": {"session_id": self._session_id}}
            )
            return cast(str, response)
        rag_chain = self._get_rag_chain()
        return await rag_chain.ainvoke(prompt)

    def stream(self, prompt: str) -> Iterator[str]:
        """Streams responses from the cloned language model for a given prompt, returning the output in chunks.

        Args:
            prompt (str): Input prompt for the cloned language model.

        Returns:
            Iterator[str]: An iterator over the streamed response chunks from the cloned language model.
        """
        self._check_is_fitted()
        if self.memory:
            rag_chain_with_history = self._get_rag_chain_with_history()
            for chunk in rag_chain_with_history.stream(
                {"input": prompt}, config={"configurable": {"session_id": self._session_id}}
            ):
                yield chunk
        else:
            rag_chain = self._get_rag_chain()
            for chunk in rag_chain.stream(prompt):
                yield chunk

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """Asynchronously streams responses from the cloned language model for a given prompt, returning the output in chunks.

        Args:
            prompt (str): Input prompt for the cloned language model.

        Returns:
            AsyncIterator[str]: An asynchronous iterator over the streamed response chunks from the cloned language model.
        """
        self._check_is_fitted()
        if self.memory:
            rag_chain_with_history = self._get_rag_chain_with_history()
            async for chunk in rag_chain_with_history.astream(
                {"input": prompt}, config={"configurable": {"session_id": self._session_id}}
            ):
                yield chunk
        else:
            rag_chain = self._get_rag_chain()
            async for chunk in rag_chain.astream(prompt):
                yield chunk

    @property
    def memory_size(self) -> int:
        """
        Returns the size of the conversation memory (number of stored messages).
        """
        return get_session_history_size(self._session_id)

    def clear_memory(self) -> None:
        """
        Clears the conversation memory for the cloned language model.
        """
        clear_session_history(self._session_id)
        self._session_id = str(uuid.uuid4())

    def reset_memory(self) -> None:
        """
        Resets the conversation memory by clearing all stored history.
        """
        self.clear_memory()

    @property
    def models_by_provider(self) -> dict[str, list[str]]:
        """
        Returns the available models grouped by their providers.
        """
        return cast(dict[str, list[str]], models_by_provider)

    def __repr__(self) -> str:
        return f"CloneLLM<(model='{self.model}'" + (f", memory={self.memory}" * (self.memory is not None)) + ")>"
