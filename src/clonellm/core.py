from __future__ import annotations
import json
import logging
from typing import Any, AsyncIterator, Iterator, Optional
from typing_extensions import Self
import uuid

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import CharacterTextSplitter, TextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.vectorstores import Chroma

from ._base import LiteLLMMixin
from ._prompt import context_prompt, user_profile_prompt, history_prompt, contextualize_question_prompt, question_prompt
from ._typing import UserProfile
from .embed import LiteLLMEmbeddings
from .memory import get_session_history, clear_session_history

logging.getLogger("langchain_core").setLevel(logging.ERROR)

__all__ = ("CloneLLM",)


class CloneLLM(LiteLLMMixin):
    """Creates an LLM clone of a user based on provided user profile and related context.

    Args:
        model (str): The name of the language model to use for text completion.
        documents (list[Document | str]): List of documents related to cloning user to use as context for the language model.
        embedding (LiteLLMEmbeddings | Embeddings): The embedding function to use for RAG.
        user_profile (Optional[UserProfile | dict[str, Any] | str]): The profile of the user to be cloned by the language model. Defaults to None.
        memory (Optional[bool]): Whether to enable the conversation memory (history). Defaults to None for no memory.
        api_key (Optional[str]): The API key to use. Defaults to None.
        **kwargs: Additional keyword arguments supported by the `langchain_community.chat_models.ChatLiteLLM` class.

    """

    __is_fitted: bool = False
    _splitter: TextSplitter
    _session_id: str
    db: Chroma

    _CHROMA_COLLECTION_NAME = "clonellm"

    def __init__(
        self,
        model: str,
        documents: list[Document | str],
        embedding: LiteLLMEmbeddings | Embeddings,
        user_profile: Optional[UserProfile | dict[str, Any] | str] = None,
        memory: Optional[bool] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key, **kwargs)
        self.embedding = embedding
        self.documents = documents
        self.user_profile = user_profile
        self.memory = memory
        self._internal_init()

    def _internal_init(self) -> None:
        self._litellm_kwargs.update({f"{self._llm_provider}_api_key": self.api_key})
        self._llm = ChatLiteLLM(model=self.model, model_name=self.model, **self._litellm_kwargs)
        self._splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        self._session_id = str(uuid.uuid4())
        self.clear_memory()

    @classmethod
    def from_persist_directory(
        cls,
        persist_directory: str,
        model: str,
        embedding: LiteLLMEmbeddings | Embeddings,
        user_profile: Optional[UserProfile | dict[str, Any] | str] = None,
        memory: Optional[bool] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Self:
        cls.db = Chroma(
            collection_name=cls._CHROMA_COLLECTION_NAME, embedding_function=embedding, persist_directory=persist_directory
        )
        cls.__is_fitted = True
        return cls(
            model=model, documents=[], embedding=embedding, user_profile=user_profile, memory=memory, api_key=api_key, **kwargs
        )

    def _get_documents(self, documents: Optional[list[Document | str]] = None) -> list[Document]:
        documents_ = []
        for i, doc in enumerate(documents or self.documents):
            if not isinstance(doc, (Document, str)):
                raise ValueError(f"item at index {i} is not a valid Document or a string")
            documents_.append(Document(page_content=doc) if isinstance(doc, str) else doc)
        return documents_

    def fit(self) -> Self:
        documents = self._get_documents()
        documents = self._splitter.split_documents(documents)
        self.db = Chroma.from_documents(documents, self.embedding, collection_name=self._CHROMA_COLLECTION_NAME)
        self.__is_fitted = True
        return self

    async def afit(self) -> Self:
        documents = self._get_documents()
        documents = self._splitter.split_documents(documents)
        self.db = await Chroma.afrom_documents(documents, self.embedding, collection_name=self._CHROMA_COLLECTION_NAME)
        self.__is_fitted = True
        return self

    def _check_is_fitted(self, from_update: bool = False) -> None:
        if not self.__is_fitted or self.db is None or ((self._splitter is None) if from_update else False):
            raise AttributeError("This CloneLLM instance is not fitted yet. Call `fit` using this method.")

    def update(
        self,
        documents: list[Document | str],
    ) -> Self:
        self._check_is_fitted(from_update=True)
        documents_ = self._get_documents(documents)
        documents_ = self._splitter.split_documents(documents_)
        self.db.add_documents(documents_)
        return self

    async def aupdate(
        self,
        documents: list[Document | str],
    ) -> Self:
        self._check_is_fitted(from_update=True)
        documents_ = self._get_documents(documents)
        documents_ = self._splitter.split_documents(documents_)
        await self.db.aadd_documents(documents_)
        return self

    @property
    def _user_profile(self) -> str:
        if isinstance(self.user_profile, UserProfile):
            return self.user_profile.model_dump_json(exclude_none=True)
        elif isinstance(self.user_profile, dict):
            return json.dumps(self.user_profile, default=str)
        return str(self.user_profile)

    def _get_retriever(self, k: int = 1) -> VectorStoreRetriever:
        return self.db.as_retriever(search_kwargs={"k": k})

    def _get_rag_chain(self) -> RunnableSerializable[Any, str]:
        prompt = context_prompt.copy()
        if self.user_profile:
            prompt += user_profile_prompt.format_messages(user_profile=self._user_profile)
        prompt += question_prompt
        return {"context": self._get_retriever(), "input": RunnablePassthrough()} | prompt | self._llm | StrOutputParser()

    def _get_rag_chain_with_history(self) -> RunnableWithMessageHistory:
        contextualize_system_prompt = contextualize_question_prompt + history_prompt + question_prompt
        history_aware_retriever = create_history_aware_retriever(self._llm, self._get_retriever(), contextualize_system_prompt)

        prompt = context_prompt
        if self.user_profile:
            prompt += user_profile_prompt.format_messages(user_profile=self._user_profile)
        prompt += history_prompt
        prompt += question_prompt
        question_answer_chain = create_stuff_documents_chain(self._llm, prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            output_parser=StrOutputParser(),
        )

    def invoke(self, prompt: str) -> str:
        self._check_is_fitted()
        if self.memory:
            rag_chain_with_history = self._get_rag_chain_with_history()
            response = rag_chain_with_history.invoke({"input": prompt}, config={"configurable": {"session_id": self._session_id}})
            return response["answer"]  # type: ignore[no-any-return]
        rag_chain = self._get_rag_chain()
        return rag_chain.invoke(prompt)

    async def ainvoke(self, prompt: str) -> str:
        self._check_is_fitted()
        if self.memory:
            rag_chain_with_history = self._get_rag_chain_with_history()
            response = await rag_chain_with_history.ainvoke(
                {"input": prompt}, config={"configurable": {"session_id": self._session_id}}
            )
            return response["answer"]  # type: ignore[no-any-return]
        rag_chain = self._get_rag_chain()
        return await rag_chain.ainvoke(prompt)

    def stream(self, prompt: str) -> Iterator[str]:
        self._check_is_fitted()
        if self.memory:
            rag_chain_with_history = self._get_rag_chain_with_history()
            iterator = rag_chain_with_history.stream({"input": prompt}, config={"configurable": {"session_id": self._session_id}})
            for chunk in iterator:
                if "answer" in chunk:
                    yield chunk["answer"]
                else:
                    yield ""
        rag_chain = self._get_rag_chain()
        for chunk in rag_chain.stream(prompt):
            yield chunk

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        self._check_is_fitted()
        if self.memory:
            rag_chain_with_history = self._get_rag_chain_with_history()
            async for chunk in rag_chain_with_history.astream(
                {"input": prompt}, config={"configurable": {"session_id": self._session_id}}
            ):
                yield chunk["answer"]
        rag_chain = self._get_rag_chain()
        async for chunk in rag_chain.astream(prompt):
            yield chunk

    def clear_memory(self) -> None:
        clear_session_history(self._session_id)
        self._session_id = str(uuid.uuid4())

    def __repr__(self) -> str:
        return f"CloneLLM<(model='{self.model}', memory={self.memory})>"
