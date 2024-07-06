import pytest
from langchain.text_splitter import TextSplitter
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.vectorstores import VectorStore

from clonellm import CloneLLM, LiteLLMEmbeddings, UserProfile
from clonellm.memory import get_session_history

LLM_MODEL = "gpt-3.5-turbo"
LLM_SETTINGS = dict(temprature=0.1, max_tokens=32)
EMBEDDING_MODEL = "text-embedding-3-small"
GENERIC_PROFILE = dict(first_name="Mehdi", last_name="Samsami")
GENERIC_CONTEXT = "I'm Mehdi Samsami."
GENERIC_PROMPT = "What football team do you support?"
embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)


def test_api_key():
    clone = CloneLLM(model=LLM_MODEL, documents=[])
    assert isinstance(clone._api_key, str)
    assert clone._api_key.startswith("sk-")


def test_internal_init(mock_find_spec):
    mock_find_spec.return_value = True
    clone = CloneLLM(model=LLM_MODEL, documents=[], embedding=embed)
    assert isinstance(clone._litellm_kwargs, dict)
    assert isinstance(clone._llm, ChatLiteLLM)
    assert clone._llm.model == LLM_MODEL
    assert isinstance(clone._splitter, TextSplitter)
    assert clone._splitter._chunk_size == clone._TEXT_SPLITTER_CHUNK_SIZE
    assert isinstance(clone._session_id, str)


def test_internal_init_with_missing_dependencies(mock_find_spec):
    mock_find_spec.return_value = None
    with pytest.raises(ImportError, match="Could not import"):
        CloneLLM(model=LLM_MODEL, documents=[], embedding=embed)


def test_internal_init_without_embedding():
    clone = CloneLLM(model=LLM_MODEL, documents=[])
    assert isinstance(clone._litellm_kwargs, dict)
    assert isinstance(clone._llm, ChatLiteLLM)
    assert clone._llm.model == LLM_MODEL
    assert not hasattr(clone, "_splitter")
    assert isinstance(clone._session_id, str)


def test_check_is_fitted_before_fit():
    clone = CloneLLM(model=LLM_MODEL, documents=[], embedding=embed)
    with pytest.raises(AttributeError):
        clone._check_is_fitted()
    with pytest.raises(AttributeError, match="is not fitted"):
        clone._check_is_fitted()
    clone.__is_fitted = True
    with pytest.raises(AttributeError, match="is not fitted"):
        clone._check_is_fitted()


def test_check_is_fitted_before_fit_without_embedding():
    clone = CloneLLM(model=LLM_MODEL, documents=[])
    with pytest.raises(AttributeError):
        clone._check_is_fitted()
    with pytest.raises(AttributeError, match="is not fitted"):
        clone._check_is_fitted()
    clone.__is_fitted = True
    with pytest.raises(AttributeError, match="is not fitted"):
        clone._check_is_fitted()


def test_check_is_fitted_after_fit(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed)
    clone.fit()
    assert clone._check_is_fitted() is None


def test_check_is_fitted_after_fit_without_embedding(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text])
    clone.fit()
    assert clone._check_is_fitted() is None


def test_user_profile():
    clone = CloneLLM(model=LLM_MODEL, documents=[], embedding=embed, user_profile=GENERIC_PROFILE)
    assert isinstance(clone._user_profile, str)
    clone.user_profile = GENERIC_CONTEXT
    assert isinstance(clone._user_profile, str)
    clone.user_profile = UserProfile(first_name=GENERIC_PROFILE["first_name"], last_name=GENERIC_PROFILE["last_name"])
    assert isinstance(clone._user_profile, str)


def test_fit(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[], embedding=embed)
    with pytest.raises(ValueError, match="No documents provided"):
        clone.fit()
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed)
    clone.fit()
    assert not hasattr(clone, "context")
    assert hasattr(clone, "db")
    assert isinstance(clone.db, VectorStore)
    assert clone.db._collection.name == clone._VECTOR_STORE_COLLECTION_NAME


def test_fit_without_embedding(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[])
    with pytest.raises(ValueError, match="No documents provided"):
        clone.fit()
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text])
    clone.fit()
    assert not hasattr(clone, "db")
    assert hasattr(clone, "context")
    assert isinstance(clone.context, str)


def test_from_context():
    clone = CloneLLM.from_context(model=LLM_MODEL, context=GENERIC_CONTEXT)
    assert not hasattr(clone, "db")
    assert hasattr(clone, "context")
    assert isinstance(clone.context, str)
    with pytest.raises(ValueError, match="No documents provided"):
        clone.fit()


def test_invoke(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, user_profile=GENERIC_PROFILE, **LLM_SETTINGS)
    clone.fit()
    response = clone.invoke("What's your name?")
    assert isinstance(response, str)
    assert bool(response)
    assert any(name.lower() in response.lower() for name in GENERIC_PROFILE.values())


def test_invoke_without_embedding(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], user_profile=GENERIC_PROFILE, **LLM_SETTINGS)
    clone.fit()
    response = clone.invoke("What's your name?")
    assert isinstance(response, str)
    assert bool(response)
    assert any(name.lower() in response.lower() for name in GENERIC_PROFILE.values())


def test_invoke_with_memory(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, memory=True, **LLM_SETTINGS)
    clone.fit()
    clone.invoke(GENERIC_PROMPT)
    assert get_session_history(clone._session_id).messages[-2].content == GENERIC_PROMPT


def test_invoke_with_memory_without_embedding(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], memory=True, **LLM_SETTINGS)
    clone.fit()
    clone.invoke(GENERIC_PROMPT)
    assert get_session_history(clone._session_id).messages[-2].content == GENERIC_PROMPT


@pytest.mark.asyncio
async def test_ainvoke_with_memory(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, memory=True, **LLM_SETTINGS)
    await clone.afit()
    await clone.ainvoke(GENERIC_PROMPT)
    assert get_session_history(clone._session_id).messages[-2].content == GENERIC_PROMPT


def test_stream(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, user_profile=GENERIC_PROFILE, **LLM_SETTINGS)
    clone.fit()
    response = ""
    for chunk in clone.stream("What's your name?"):
        assert isinstance(chunk, str)
        response += chunk
    assert any(name.lower() in response.lower() for name in GENERIC_PROFILE.values())


def test_stream_without_embedding(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], user_profile=GENERIC_PROFILE, **LLM_SETTINGS)
    clone.fit()
    response = ""
    for chunk in clone.stream("What's your name?"):
        assert isinstance(chunk, str)
        response += chunk
    assert any(name.lower() in response.lower() for name in GENERIC_PROFILE.values())


def test_stream_with_memory(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, memory=True, **LLM_SETTINGS)
    clone.fit()
    for chunk in clone.stream(GENERIC_PROMPT):
        assert isinstance(chunk, str)
    assert get_session_history(clone._session_id).messages[-2].content == GENERIC_PROMPT


def test_stream_with_memory_without_embedding(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], memory=True, **LLM_SETTINGS)
    clone.fit()
    for chunk in clone.stream(GENERIC_PROMPT):
        assert isinstance(chunk, str)
    assert get_session_history(clone._session_id).messages[-2].content == GENERIC_PROMPT


@pytest.mark.asyncio
async def test_astream_with_memory(random_text):
    clone = CloneLLM(model=LLM_MODEL, documents=[random_text], embedding=embed, memory=True, **LLM_SETTINGS)
    await clone.afit()
    async for chunk in clone.astream(GENERIC_PROMPT):
        assert isinstance(chunk, str)
    assert get_session_history(clone._session_id).messages[-2].content == GENERIC_PROMPT


def test_memory_size():
    clone = CloneLLM.from_context(model=LLM_MODEL, context=GENERIC_CONTEXT, user_profile=GENERIC_PROFILE, memory=True)
    assert clone.memory_size == 0
    clone.invoke("What's your first name?")
    assert clone.memory_size == 2
    clone.invoke("What's your last name?")
    assert clone.memory_size == 4


def test_clear_memory():
    clone = CloneLLM(model=LLM_MODEL, documents=[], embedding=embed, memory=True)
    session_id = clone._session_id
    clone.clear_memory()
    assert clone._session_id != session_id
    assert not get_session_history(session_id).messages
    assert clone.memory_size == 0
