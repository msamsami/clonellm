<p align="center">
    <img src="https://raw.githubusercontent.com/msamsami/clonellm/main/docs/assets/images/logo.png" alt="Logo" width="250" />
</p>
<h1 align="center">
    CloneLLM
</h1>
<p align="center">
    <p align="center">Create an AI clone of yourself using LLMs.</p>
</p>

<h4 align="center">
    <a href="https://pypi.org/project/clonellm/" target="_blank">
        <img src="https://img.shields.io/badge/release-v0.3.0-green" alt="Latest Release">
    </a>
    <a href="https://pypi.org/project/clonellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/clonellm.svg" alt="PyPI Version">
    </a>
    <a target="_blank">
        <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python Versions">
    </a>
    <a target="_blank">
        <img src="https://img.shields.io/pypi/l/clonellm" alt="PyPI License">
    </a>
</h4>

## Introduction
CloneLLM is a minimal Python package that enables you to create an AI clone of yourself using LLMs. Built on top of LiteLLM and LangChain, CloneLLM utilizes Retrieval-Augmented Generation (RAG) to tailor AI responses as if you are answering the questions.

You can input texts and documents about yourself — including personal information, professional experience, educational background, etc. — which are then embedded into a vector space for dynamic retrieval. This AI clone can act as a virtual assistant or digital representation, capable of handling queries and tasks in a manner that reflects your own knowledge, tone, style, and mannerisms.

## Installation
Before installing CloneLLM, make sure you have Python 3.10 or newer installed on your machine.

### PyPi
```bash
pip install clonellm
```

### Poetry
```bash
poetry add clonellm
```

### uv
```bash
uv add clonellm
```

## Usage

### Getting started
You can set up a clone of yourself using CloneLLM in just a few lines of code.

**Step 1**. Gather documents that contain relavant information about you. These documents form the base from which your AI clone will learn to mimic your tone, style, and expertise.
```python
from langchain_core.documents import Document

documents = [
    Document(page_content="My name is Mehdi Samsami."),
    open("about_me.txt", "r").read(),
]
```

**Step 2**. Initialize a clone with your documents and your preferred LLM.
```python
from clonellm import CloneLLM

clone = CloneLLM(model="gpt-4o", documents=documents)
```

**Step 3**. Configure environment variables to store API keys for LLM model.
```bash
export OPENAI_API_KEY=sk-...
```

**Step 4**. Fit the clone to the data (documents).
```python
clone.fit()
```

**Step 5**. Invoke the clone to ask questions.
```python
clone.invoke("What's your name?")

# Response: My name is Mehdi Samsami. How can I help you?
```

### Models
At its core, CloneLLM utilizes LiteLLM for interactions with various LLMs. This is why you can choose from 100+ LLMs from many different providers, including Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate, etc.

### Document loaders
You can use LangChain's document loaders to seamlessly import data from various sources into `Document` format. Take, for example, text and HTML loaders:
```python
# !pip install unstructured
from langchain_community.document_loaders import TextLoader, UnstructuredHTMLLoader

documents = TextLoader("cv.txt").load() + UnstructuredHTMLLoader("linkedin.html").load()
```

Or JSON loader:
```python
# !pip install jq
from langchain_community.document_loaders import JSONLoader

documents = JSONLoader(
    file_path='chat.json',
    jq_schema='.messages[].content',
    text_content=False
).load()
```

### RAG
In the basic usage described above, documents are summarized to create a static context for interacting with the LLM. This is the default behavior where the `embedding` and `vector_store` parameters are not specified. For a more advanced usage, you can specify an embedding model and a vector store to implement a RAG-based question-answering system. In this scenario, the documents are embedded and stored in the vector store, allowing them to serve as a dynamic retrieval context for each prompt.

#### Embeddings
With `LiteLLMEmbeddings`, CloneLLM allows you to utilize embedding models from a variety of providers supported by LiteLLM:
```python
from clonellm import CloneLLM, LiteLLMEmbeddings
import os

os.environ["OPENAI_API_KEY"] = "openai-api-key"

embedding = LiteLLMEmbeddings(model="text-embedding-3-small", dimensions=1024)
clone = CloneLLM(model="gpt-4o-mini", documents=documents, embedding=embedding)
```

Additionally, you can select any preferred embedding model from LangChain's extensive range. Take, for example, the Hugging Face embedding:
```python
# !pip install --upgrade --quiet sentence_transformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from clonellm import CloneLLM
import os

os.environ["COHERE_API_KEY"] = "cohere-api-key"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
clone = CloneLLM(model="command-xlarge-beta", documents=documents, embedding=embedding)
```

Or, the Llama-cpp embedding:
```python
# !pip install --upgrade --quiet llama-cpp-python
from langchain_community.embeddings import LlamaCppEmbeddings
from clonellm import CloneLLM
import os

os.environ["OPENAI_API_KEY"] = "openai-api-key"

embedding = LlamaCppEmbeddings(model_path="ggml-model-q4_0.bin")
clone = CloneLLM(model="gpt-4o-mini", documents=documents, embedding=embedding)
```

#### Vector store
Currently, CloneLLM supports the following vector stores:
- [Chroma](https://github.com/chroma-core/chroma)
- [FAISS](https://github.com/facebookresearch/faiss)
- [InMemory](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html) (default)

When an embedding model is specified (via the `embedding` parameter), the dynamic context retrieval is enabled and the selected vector store will be initialized and used to store the document embeddings.
```python
# !pip install clonellm[faiss]
from clonellm import CloneLLM, LiteLLMEmbeddings, RagVectorStore
import os

os.environ["OPENAI_API_KEY"] = "openai-api-key"

embedding = LiteLLMEmbeddings(model="text-embedding-3-small")
clone = CloneLLM(model="gpt-4o", documents=documents, embedding=embedding, vector_store=RagVectorStore.FAISS)
```

### User profile
Create a personalized profile using CloneLLM's `UserProfile`, which allows you to feed detailed personal information into your clone for more customized interactions:
```python
from clonellm import UserProfile
from clonellm.models import PersonalityTraits

profile = UserProfile(
    first_name="Mehdi",
    last_name="Samsami",
    city="Shiraz",
    country="Iran",
    expertise=["Data Science", "AI/ML", "Data Analytics"],
    personality_traits=PersonalityTraits(
        openness=0.8,
        conscientiousness=0.7,
        extraversion=0.6,
        agreeableness=0.9,
        neuroticism=0.5,
    ),
)
```

Or simply define your profile using Python dictionaries:
```python
profile = {
    "full_name": "Mehdi Samsami",
    "age": 28,
    "location": "Shiraz, Iran",
    "expertise": ["Data Science", "AI/ML", "Data Analytics"],
    "languages": ["English", "Persian"],
    "tone": "Friendly",
}
```

Finnaly:
```python
# !pip install clonellm[chroma]
from clonellm import CloneLLM
import os

os.environ["ANTHROPIC_API_KEY"] = "anthropic-api-key"

clone = CloneLLM(
    model="claude-3-5-sonnet-latest",
    documents=documents,
    embedding=embedding,
    vector_store=RagVectorStore.Chroma,
    user_profile=profile,
)
```

### Conversation history (memory)
Enable the memory feature to allow your clone to access to the history of conversation. This is simply done by setting `memory` argument to `True` or -1 for infinite memory or an integer greater than zero for a fixed size of memory:
```python
from clonellm import CloneLLM
import os

os.environ["HUGGINGFACE_API_KEY"] = "huggingface-api-key"

clone = CloneLLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    documents=documents,
    embedding=embedding,
    memory=10,  # Enable memory with maximum size of 10
)
```

Use the `memory_size` attribute to get the current length of conversation history, i.e., the size of clone memory:
```
print(clone.memory_size)
# 6
```

If you needed to clear the history of the conversation, i.e., the clone memory, at any time, you can easily call either of the `reset_memory()` and `clear_memory()` methods.
```python
clone.clear_memory()
# clone.reset_memory()
```

### Streaming
CloneLLM supports streaming responses from the LLM, allowing for real-time processing of text as it is being generated, rather than receiving the whole output at once.
```python
from clonellm import CloneLLM, LiteLLMEmbeddings
import os

os.environ["VERTEXAI_PROJECT"] = "hardy-device-28813"
os.environ["VERTEXAI_LOCATION"] = "us-central1"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"

embedding = LiteLLMEmbeddings(model="text-embedding-005")
clone = CloneLLM(model="gemini-1.5-flash", documents=documents, embedding=embedding)

for chunk in clone.stream("Describe yourself in 100 words"):
    print(chunk, end="", flush=True)
```

### Async
CloneLLM provides asynchronous counterparts to its core methods, `afit`, `ainvoke`, and `astream`, enhancing performance in asynchronous programming contexts.

#### `ainvoke`
```python
import asyncio
from clonellm import CloneLLM, LiteLLMEmbeddings
from langchain_core.documents import Document
import os

os.environ["OPENAI_API_KEY"] = "openai-api-key"

async def main():
    documents = [...]
    embedding = LiteLLMEmbeddings(model="text-embedding-3-large")
    clone = CloneLLM(model="gpt-4o", documents=documents, embedding=embedding)
    await clone.afit()
    response = await clone.ainvoke("Tell me about your skills?")
    return response

response = asyncio.run(main())
print(response)
```

#### `astream`
```python
import asyncio
from clonellm import CloneLLM, LiteLLMEmbeddings
from langchain_core.documents import Document
import os

os.environ["OPENAI_API_KEY"] = "openai-api-key"

async def main():
    documents = [...]
    embedding = LiteLLMEmbeddings(model="text-embedding-3-small")
    clone = CloneLLM(model="gpt-4o-mini", documents=documents, embedding=embedding)
    await clone.afit()
    async for chunk in clone.astream("How comfortable are you with remote work?"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## Support Us
If you find CloneLLM useful, please consider showing your support in one of the following ways:

- ⭐ **Star our GitHub repository:** This helps increase the visibility of our project.
- 💡 **Contribute:** Submit pull requests to help improve the codebase, whether it's adding new features, fixing bugs, or improving documentation.
- 📰 **Share:** Post about CloneLLM on LinkedIn or other social platforms.

Thank you for your interest in CloneLLM. We look forward to seeing you create your digital twin!
