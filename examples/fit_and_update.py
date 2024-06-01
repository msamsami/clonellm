from clonellm import CloneLLM, LiteLLMEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

# !pip install jq
# !pip install pypdf
# !pip install unstructured


def main():
    documents = [
        Document(page_content=open("about_me.txt", "r").read()),
        open("bio.txt", "r").read(),
    ]
    documents += TextLoader("history.txt").load()
    documents += UnstructuredHTMLLoader("linkedin.html", strategy="fast").load()
    documents += UnstructuredMarkdownLoader("README.md").load()
    documents += PyPDFLoader("my_cv.pdf").load()

    embedding = LiteLLMEmbeddings(model="text-embedding-3-small")
    clone = CloneLLM(model="gpt-4o", documents=documents, embedding=embedding)
    clone.fit()

    new_documents = JSONLoader(file_path="chat.json", jq_schema=".messages[].content", text_content=False).load()
    new_documents += DirectoryLoader("docs/", glob="**/*.md").load()
    clone.update(new_documents)

    # Handle conversation
    ...


if __name__ == "__main__":
    main()
