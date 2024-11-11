from clonellm import CloneLLM
from examples.const import EXIT_COMMANDS


def main() -> None:
    documents = [open("about_me.txt").read()]
    clone = CloneLLM(model="gpt-3.5-turbo", documents=documents)
    clone.fit()

    while True:
        prompt = input("Question: ")
        if prompt.strip().lower() in EXIT_COMMANDS:
            break
        response = clone.invoke(prompt)
        print("Response:", response, end="\n\n")


if __name__ == "__main__":
    main()
