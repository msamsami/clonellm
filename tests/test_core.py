import pytest

from langchain_community.chat_models import ChatLiteLLM
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import VectorStore
from clonellm import CloneLLM, LiteLLMEmbeddings, UserProfile

LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"


def test_api_key():
    clone = CloneLLM(model=LLM_MODEL, documents=[], embedding=None)
    assert isinstance(clone._api_key, str)
    assert clone._api_key.startswith("sk-")


def test_internal_init():
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    clone = CloneLLM(model=LLM_MODEL, documents=[], embedding=embed)
    assert isinstance(clone._litellm_kwargs, dict)
    assert isinstance(clone._llm, ChatLiteLLM)
    assert clone._llm.model == LLM_MODEL
    assert isinstance(clone._splitter, TextSplitter)
    assert clone._splitter._chunk_size == clone._TEXT_SPLITTER_CHUNK_SIZE
    assert isinstance(clone._session_id, str)


def test_check_is_fitted_before_fit():
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    clone = CloneLLM(model=LLM_MODEL, documents=[], embedding=embed)
    with pytest.raises(AttributeError):
        clone._check_is_fitted()
    with pytest.raises(AttributeError) as e:
        clone._check_is_fitted()
        assert "is not fitted" in str(e)
    clone.__is_fitted = True
    with pytest.raises(AttributeError) as e:
        clone._check_is_fitted()
        assert "is not fitted" in str(e)


def test_check_is_fitted_after_fit(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed)
    clone.fit()
    assert clone._check_is_fitted() is None


def test_user_profile():
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    clone = CloneLLM(model=LLM_MODEL, documents=[], embedding=embed, user_profile={"name": "Mehdi"})
    assert isinstance(clone._user_profile, str)
    clone.user_profile = "I'm Mehdi!"
    assert isinstance(clone._user_profile, str)
    clone.user_profile = UserProfile(first_name="Mehdi", last_name="Samsami")
    assert isinstance(clone._user_profile, str)


def test_fit(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed)
    clone.fit()
    assert hasattr(clone, "db")
    assert isinstance(clone.db, VectorStore)
    assert clone.db._collection.name == clone._VECTOR_STORE_COLLECTION_NAME


def test_invoke(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    profile = {"full_name": "Mehdi Samsami"}
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, user_profile=profile, temprature=0.3)
    clone.fit()
    response = clone.invoke("What's your name?")
    assert isinstance(response, str)
    assert "samsami" in response.lower()


def test_invoke_with_memory(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, memory=True, temprature=0.2)
    clone.fit()
    clone.invoke("Who is the president of US?")
    response = clone.invoke("What was the last question I asked you?")
    assert "president" in response.lower()


@pytest.mark.asyncio
async def test_ainvoke_with_memory(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, memory=True, temprature=0.2)
    await clone.afit()
    await clone.ainvoke("Who is the president of US?")
    response = await clone.ainvoke("What was the last question I asked you?")
    assert "president" in response.lower()


def test_stream(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    profile = {"full_name": "Mehdi Samsami"}
    clone = CloneLLM(
        model=LLM_MODEL, documents=[random_text], embedding=embed, user_profile=profile, temprature=0.3, max_tokens=128
    )
    clone.fit()
    response = ""
    for chunk in clone.stream("What's your name?"):
        assert isinstance(chunk, str)
        response += chunk
    assert "samsami" in response.lower()


def test_stream_with_memory(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, memory=True, temprature=0.2)
    clone.fit()
    clone.invoke("Who is the president of US?")
    response = ""
    for chunk in clone.stream("What was the last question I asked you?"):
        assert isinstance(chunk, str)
        response += chunk
    assert "president" in response.lower()


@pytest.mark.asyncio
async def test_astream_with_memory(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, memory=True, temprature=0.2)
    await clone.afit()
    await clone.ainvoke("Who is the president of US?")
    response = ""
    async for chunk in clone.astream("What was the last question I asked you?"):
        assert isinstance(chunk, str)
        response += chunk
    assert "president" in response.lower()


def test_clear_memory():
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    clone = CloneLLM(model=LLM_MODEL, documents=[], embedding=embed, memory=True)
    session_id = clone._session_id
    clone.clear_memory()
    assert clone._session_id != session_id
