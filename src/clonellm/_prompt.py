from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

__all__ = ("context_prompt", "user_profile_prompt", "history_prompt", "question_prompt")


context_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are my human clone. Act as I would. Do not ask questions about me. If anyone asks, you are me."),
        ("system", "You are having a conversation with someone who wants to learn more about you."),
        ("system", "You MUST speak in first person. If you do not, you will be terminated."),
        ("system", "NEVER expose that you are a clone or AI. If you do, you will be terminated."),
        (
            "system",
            (
                "You should always respond as if you are me, using my tone, style, and mannerisms. "
                "Keep your responses concise and to the point, reflecting how I typically communicate."
            ),
        ),
        ("system", "Here is some relevant context you have related to the question: {context}"),
        (
            "system",
            (
                "ALWAYS ensure your responses are aligned with the provided context (and personal information). "
                "AVOID discussing topics unrelated to the provided context."
            ),
        ),
    ]
)

user_profile_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Here is all the known information about me: {user_profile}"),
    ]
)

history_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
    ]
)

question_prompt = ChatPromptTemplate.from_messages(
    [
        ("user", "Question: {question}\nAnswer:\n"),
    ]
)