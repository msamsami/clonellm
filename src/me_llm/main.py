from __future__ import annotations
import json
from typing import Optional, Self

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, TextSplitter

from ._base import LiteLLMMixin
from ._prompt import context_prompt, user_profile_prompt, question_prompt
from ._typing import UserProfile
from .embed import LiteLLMEmbeddings

__all__ = ("MeLLM",)


class MeLLM(LiteLLMMixin):
    _splitter: TextSplitter = None
    _documents: list[Document] = None
    db: Chroma = None

    _CHROMA_COLLECTION_NAME = "mellm"

    def __init__(
        self,
        model: str,
        documents: list[Document],
        embedding: LiteLLMEmbeddings | Embeddings,
        user_profile: Optional[UserProfile | dict | str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, api_key, **kwargs)
        self.embedding = embedding
        self.documents = documents
        self.user_profile = user_profile
        self._internal_init()

    def _internal_init(self):
        api_key_kwarg = {f"{self._llm_provider}_api_key": self.api_key}
        self._llm = ChatLiteLLM(model=self.model, model_name=self.model, **api_key_kwarg, **self._litellm_kwargs)
        self._splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

    def _check_documents(self):
        for i, doc in enumerate(self.documents):
            if not isinstance(doc, Document):
                raise ValueError(f"document at index {i} is not a valid Document")

    @property
    def _user_profile(self) -> str:
        if isinstance(self.user_profile, UserProfile):
            return self.user_profile.model_dump_json()
        elif isinstance(self.user_profile, dict):
            return json.dumps(self.user_profile)
        return str(self.user_profile)

    def fit(self) -> Self:
        self._check_documents()
        self._documents = None
        self._documents = self._splitter.split_documents(self.documents)
        self.db = Chroma.from_documents(self._documents, self.embedding, collection_name=self._CHROMA_COLLECTION_NAME)
        return self

    async def afit(self) -> Self:
        self._check_documents()
        self._documents = None
        self._documents = self._splitter.split_documents(self.documents)
        self.db = await Chroma.afrom_documents(self._documents, self.embedding, collection_name=self._CHROMA_COLLECTION_NAME)
        return self

    def update(self): ...

    async def aupdate(self): ...

    def _get_rag_chain(self) -> RunnableSequence:
        prompt = context_prompt + user_profile_prompt.format_messages(user_profile=self._user_profile) + question_prompt
        return (
            {"context": self.db.as_retriever(search_kwargs={"k": 1}), "question": RunnablePassthrough()}
            | prompt
            | self._llm
            | StrOutputParser()
        )

    def completion(self, prompt: str) -> str:
        rag_chain = self._get_rag_chain()
        return rag_chain.invoke(prompt)

    async def acompletion(self, prompt: str) -> str:
        docs = await self.db.asimilarity_search(prompt, k=1)
        context = docs[0].page_content
        prompts = context_prompt + user_profile_prompt + question_prompt
        messages = await prompts.aformat_messages(context=context, user_profile=self._user_profile, question=prompt)
        return await self._llm.ainvoke(messages)

    def __repr__(self) -> str:
        return f"MeLLM<(model='{self.model})>"
