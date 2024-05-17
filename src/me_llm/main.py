from typing import Optional, Self

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from litellm import completion, acompletion

from ._base import LiteLLMMixin
from ._prompt import personal_info_prompt, context_prompt, question_prompt
from ._typing import PersonalInfo
from .embed import LiteLLMEmbeddings

__all__ = ("MeLLM",)


class MeLLM(LiteLLMMixin):
    _splitter: CharacterTextSplitter = None
    _documents: list[Document] = None
    db: Chroma = None

    def __init__(
        self,
        documents: list[Document],
        embedding: LiteLLMEmbeddings | Embeddings,
        personal_info: Optional[PersonalInfo] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, api_key, **kwargs)
        self.embedding = embedding
        self.documents = documents
        self.personal_info = personal_info
        self._internal_init()

    def _internal_init(self):
        self._splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    def _check_documents(self):
        for i, doc in enumerate(self.documents):
            if not isinstance(doc, Document):
                raise ValueError(f"document at index {i} is not a valid Document")

    def fit(self) -> Self:
        self._check_documents()
        self._documents = None
        self._documents = self._splitter.split_documents(self.documents)
        self.db = Chroma.from_documents(self._documents, self.embedding)
        return self

    async def afit(self) -> Self:
        self._check_documents()
        self._documents = None
        self._documents = self._splitter.split_documents(self.documents)
        self.db = await Chroma.afrom_documents(self._documents, self.embedding)
        return self

    def update(self): ...

    async def aupdate(self): ...

    def completion(self, prompt: str) -> str:
        docs = self.db.similarity_search(prompt, k=1)
        context = docs[0].page_content
        prompts = (context_prompt + personal_info_prompt + question_prompt).format_messages(
            context=context, personal_info=self.personal_info.model_dump(), input=prompt
        )
        messages = [{"role": msg.type.replace("human", "user"), "content": msg.content} for msg in prompts]
        return completion(model=self.model, api_key=self.api_key, messages=messages, **self._litellm_kwargs)

    async def acompletion(self, prompt: str) -> str:
        docs = await self.db.asimilarity_search(prompt, k=1)
        context = docs[0].page_content
        prompts = context_prompt + personal_info_prompt + question_prompt
        prompts = await prompts.aformat_messages(context=context, personal_info=self.personal_info.model_dump(), input=prompt)
        messages = [{"role": msg.type.replace("human", "user"), "content": msg.content} for msg in prompts]
        return await acompletion(model=self.model, api_key=self.api_key, messages=messages, **self._litellm_kwargs)

    def __repr__(self) -> str:
        return f"MeLLM<(model='{self.model})>"
