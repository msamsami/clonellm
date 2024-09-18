import asyncio

from langchain_core.documents import Document

from clonellm import CloneLLM, LiteLLMEmbeddings, RagVectorStore

RESET_MEMORY_COMMANDS = ["reset memory", "clear memory", "clear your memory", "forget everything"]
EXIT_COMMANDS = ["exit", "quit"]


async def main():
    documents = [Document(page_content=open("bio.txt").read())]
    embedding = LiteLLMEmbeddings(model="embed-english-light-v3.0")
    clone = CloneLLM(
        model="command-nightly", documents=documents, embedding=embedding, vector_store=RagVectorStore.FAISS, memory=-1
    )
    await clone.afit()

    while True:
        prompt = input("Question: ")
        if prompt.strip().lower() in EXIT_COMMANDS:
            break

        print("Response: ", end="", flush=True)

        if any(cmd in prompt.lower() for cmd in RESET_MEMORY_COMMANDS):
            clone.clear_memory()
            print("Done!", end="\n\n", flush=True)
            continue

        response = await clone.ainvoke(prompt)
        print(response, end="\n\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
