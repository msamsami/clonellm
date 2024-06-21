import datetime

from clonellm import CloneLLM, LiteLLMEmbeddings, UserProfile
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# !pip install pypdf
# !pip install unstructured

RESET_MEMORY_COMMANDS = ["reset memory", "clear memory", "clear your memory", "forget everything"]
EXIT_COMMANDS = ["exit", "quit"]
MAX_MEMORY_SIZE = 20


def main():
    documents = [open("about_me.txt").read()]
    documents += PyPDFLoader("my_cv.pdf").load()
    documents += DirectoryLoader("docs/", glob="**/*.md").load()

    embedding = LiteLLMEmbeddings(model="text-embedding-3-small", dimensions=1024, timeout=90, caching=True)

    profile = UserProfile(
        first_name="Jane",
        middle_name="Eleanor",
        last_name="Doe",
        prefix="Ms.",
        birth_date=datetime.date(1996, 6, 30),
        gender="Female",
        city="Springfield",
        state="Illinois",
        country="USA",
        phone_number="+1234567890",
        email="jane.doe@example.com",
        education_experience=[
            {"Degree": "Bachelor's", "Field": "Computer Science", "Institution": "State University", "Year": 2018},
            {"Degree": "Master's", "Field": "Data Science", "Institution": "Tech College", "Year": 2021},
        ],
        expertise=["Python", "Data Analysis", "Data Science", "Machine Learning"],
        home_page="http://www.jane-doe.com",
        github_page="http://www.github.com/jane_doe",
        linkedin_page="http://www.linkedin.com/in/jane_doe",
    )

    clone = CloneLLM(
        model="gpt-4o",
        documents=documents,
        embedding=embedding,
        user_profile=profile,
        memory=MAX_MEMORY_SIZE,
        request_timeout=5,
        temperature=0.3,
        max_tokens=256,
        max_retries=3,
    )
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

        for chunk in clone.stream(prompt):
            print(chunk, end="", flush=True)

        print("\n")


if __name__ == "__main__":
    main()
