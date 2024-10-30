import pytest
from litellm import (
    azure_embedding_models,
    bedrock_embedding_models,
    cohere_embedding_models,
    open_ai_embedding_models,
    vertex_embedding_models,
)
from litellm.exceptions import AuthenticationError, BadRequestError

from clonellm import LiteLLMEmbeddings

API_KEY = "TEST_API_KEY"
EMBEDDING_MODEL = "text-embedding-3-small"


@pytest.mark.parametrize(
    ("provider", "models"),
    [
        ("azure", azure_embedding_models),
        ("cohere", cohere_embedding_models),
        ("vertex_ai", vertex_embedding_models),
        ("bedrock", bedrock_embedding_models),
        ("openai", open_ai_embedding_models),
    ],
)
def test_llm_provider(provider: str, models: list[str] | dict[str, str]):
    models_ = list(models.values()) if isinstance(models, dict) else models
    for model in models_:
        assert LiteLLMEmbeddings(model)._llm_provider == provider


def test_api_key():
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    assert isinstance(embed._api_key, str)
    assert embed._api_key.startswith("sk-")


def test_invalid_api_key():
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL, api_key="invalid-key")
    with pytest.raises(AuthenticationError):
        embed.embed_documents(["text"])


def test_invalid_model():
    with pytest.raises(BadRequestError):
        LiteLLMEmbeddings(model="non-existent-model").embed_documents(["text"])


def test_embed_documents(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    documents = [random_text for _ in range(10)]
    embeddings = embed.embed_documents(documents)
    assert isinstance(embeddings, list)
    for item in embeddings:
        assert isinstance(item, list)
        for v in item:
            isinstance(v, float)


@pytest.mark.asyncio
async def test_aembed_documents(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    documents = [random_text for _ in range(10)]
    embeddings = await embed.aembed_documents(documents)
    assert isinstance(embeddings, list)
    for item in embeddings:
        assert isinstance(item, list)
        for v in item:
            isinstance(v, float)


@pytest.mark.parametrize("dimensions", [256, 512, 1024])
def test_embed_documents_with_dimensions(dimensions: int, random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL, dimensions=dimensions)
    documents = [random_text for _ in range(10)]
    embeddings = embed.embed_documents(documents)
    for item in embeddings:
        assert len(item) == dimensions


def test_embed_documents_empty_list():
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    with pytest.raises(BadRequestError):
        embed.embed_documents([])


@pytest.mark.asyncio
async def test_aembed_documents_empty_list():
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    with pytest.raises(BadRequestError):
        await embed.aembed_documents([])


def test_embed_query(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    embedding = embed.embed_query(random_text)
    assert isinstance(embedding, list)
    for v in embedding:
        assert isinstance(v, float)


@pytest.mark.asyncio
async def test_aembed_query(random_text):
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    embedding = await embed.aembed_query(random_text)
    assert isinstance(embedding, list)
    for v in embedding:
        assert isinstance(v, float)


def test_all_embedding_models():
    embed = LiteLLMEmbeddings(model=EMBEDDING_MODEL)
    assert isinstance(embed.all_embedding_models, list)
    for model in embed.all_embedding_models:
        assert isinstance(model, str)
