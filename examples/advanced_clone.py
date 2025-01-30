import datetime

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from clonellm import CloneLLM, LiteLLMEmbeddings, RagVectorStore, UserProfile
from clonellm.models import CommunicationSample, PersonalityTraits
from examples.const import EXIT_COMMANDS, RESET_MEMORY_COMMANDS

# !pip install clonellm[chroma]
# !pip install pypdf
# !pip install unstructured

MAX_MEMORY_SIZE = 20


def main() -> None:
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
        personality_traits=PersonalityTraits(
            openness=0.8,
            conscientiousness=0.7,
            extraversion=0.6,
            agreeableness=0.7,
            neuroticism=0.6,
        ),
        education_experience=[
            {"Degree": "Bachelor's", "Field": "Computer Science", "Institution": "State University", "Year": 2018},
            {"Degree": "Master's", "Field": "Data Science", "Institution": "Tech College", "Year": 2021},
        ],
        expertise=["Python", "Data Analysis", "Data Science", "Machine Learning"],
        communication_samples=[
            CommunicationSample(
                context="Interview",
                audience_type="Interviewer",
                formality_level=0.5,
                content="Hello, I am a Data Scientist and ML Engineer who is looking for a new job.",
            )
        ],
        home_page="http://www.jane-doe.com",
        github_page="http://www.github.com/jane_doe",
        linkedin_page="http://www.linkedin.com/in/jane_doe",
    )

    clone = CloneLLM(
        model="gpt-4o",
        documents=documents,
        embedding=embedding,
        vector_store=RagVectorStore.Chroma,
        user_profile=profile,
        memory=MAX_MEMORY_SIZE,
        system_prompts=["Keep your responses brief and concise, and always respond in first person."],
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
