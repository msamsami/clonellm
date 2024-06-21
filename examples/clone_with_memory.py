from clonellm import CloneLLM, LiteLLMEmbeddings
from langchain_core.documents import Document

RESET_MEMORY_COMMANDS = ["reset memory", "clear memory", "clear your memory", "forget everything"]
EXIT_COMMANDS = ["exit", "quit"]


def main():
    documents = [Document(page_content=open("data/about_me.txt").read())]
    embedding = LiteLLMEmbeddings(model="embed-english-light-v3.0")
    clone = CloneLLM(model="command-nightly", documents=documents, embedding=embedding, memory=-1)
    clone.fit()

    while True:
        prompt = input("Question: ")
        if prompt.strip().lower() in EXIT_COMMANDS:
            break

        print("Response: ", end="", flush=True)

        if any(cmd in prompt.lower() for cmd in RESET_MEMORY_COMMANDS):
            clone.clear_memory()
            print("Done!", end="\n\n", flush=True)
            continue

        response = clone.invoke(prompt)
        print(response, end="\n\n", flush=True)


if __name__ == "__main__":
    main()
