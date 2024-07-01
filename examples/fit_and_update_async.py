import asyncio

from langchain_community.document_loaders import (
    DirectoryLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from clonellm import CloneLLM, LiteLLMEmbeddings

# !pip install jq
# !pip install pypdf
# !pip install unstructured


async def main():
    documents = [
        Document(page_content=open("about_me.txt", "r").read()),
        open("bio.txt", "r").read(),
    ]
    documents += await TextLoader("history.txt").aload()
    documents += await UnstructuredHTMLLoader("linkedin.html", strategy="fast").aload()
    documents += await UnstructuredMarkdownLoader("README.md").aload()
    documents += await PyPDFLoader("my_cv.pdf").aload()

    embedding = LiteLLMEmbeddings(model="text-embedding-3-small")
    clone = CloneLLM(model="gpt-4o", documents=documents, embedding=embedding)
    await clone.afit()

    new_documents = await JSONLoader(file_path="chat.json", jq_schema=".messages[].content", text_content=False).aload()
    new_documents += await DirectoryLoader("docs/", glob="**/*.md").aload()
    await clone.aupdate(new_documents)

    # Handle conversation
    ...


if __name__ == "__main__":
    asyncio.run(main())
