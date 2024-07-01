import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import clonellm.memory
from clonellm.memory import (
    InMemoryHistory,
    clear_session_history,
    get_session_history,
    get_session_history_size,
)


def test_history_add_message():
    history = InMemoryHistory()
    history.add_message(HumanMessage(content="Hello"))
    assert len(history.messages)
    assert history.messages == [HumanMessage(content="Hello")]


def test_history_last_message():
    history = InMemoryHistory(messages=[HumanMessage(content="one"), SystemMessage(content="two")])
    assert history.messages[-1] == SystemMessage(content="two")


def test_history_memory_size():
    history = InMemoryHistory()
    history.add_message(HumanMessage(content="Hello"))
    assert history.memory_size == len(history.messages) == 1
    history.add_message(AIMessage(content="Hi"))
    assert history.memory_size == len(history.messages) == 2
    for i in range(20):
        history.add_user_message("Who are you?")
        assert history.memory_size == len(history.messages) == 2 + (i + 1)


def test_history_clear():
    history = InMemoryHistory(messages=[HumanMessage(content="one"), SystemMessage(content="two")])
    history.add_user_message("three")
    assert history.memory_size == 3
    history.clear()
    assert history.memory_size == 0


def test_history_init_with_limited_size():
    history = InMemoryHistory(messages=[HumanMessage(content="one"), SystemMessage(content="two")])
    assert history.messages == [HumanMessage(content="one"), SystemMessage(content="two")]
    history = InMemoryHistory(messages=[HumanMessage(content="one"), SystemMessage(content="two")], max_memory_size=0)
    assert history.messages == []
    history = InMemoryHistory(messages=[HumanMessage(content="one"), SystemMessage(content="two")], max_memory_size=1)
    assert history.messages == [SystemMessage(content="two")]
    history = InMemoryHistory(messages=[HumanMessage(content="one"), SystemMessage(content="two")], max_memory_size=2)
    assert history.messages == [HumanMessage(content="one"), SystemMessage(content="two")]
    history = InMemoryHistory(messages=[HumanMessage(content="one"), SystemMessage(content="two")], max_memory_size=3)
    assert history.messages == [HumanMessage(content="one"), SystemMessage(content="two")]


def test_history_add_message_with_limited_size():
    max_memory_size = 3
    message = HumanMessage(content="Hello")
    history = InMemoryHistory(max_memory_size=max_memory_size)
    for i in range(50):
        history.add_message(message)
        assert history.memory_size == min(i + 1, max_memory_size)


@pytest.mark.usefixtures("clear_memory_store")
def test_get_session_history():
    session_id = "123"
    history = get_session_history(session_id)
    assert isinstance(history, InMemoryHistory)
    assert history.memory_size == 0
    assert len(clonellm.memory._store) == 1
    assert session_id in clonellm.memory._store
    history.add_ai_message("Hello")
    assert len(get_session_history(session_id).messages) == 1
    assert get_session_history(session_id).messages[0] == AIMessage(content="Hello")
    _ = get_session_history("abc")
    assert len(clonellm.memory._store) == 2
    assert "abc" in clonellm.memory._store


@pytest.mark.usefixtures("clear_memory_store")
def test_get_session_history_size(random_text):
    assert isinstance(get_session_history_size(random_text), int)
    assert get_session_history_size(random_text) == 0
    session_id = "123"
    history = get_session_history(session_id)
    assert get_session_history_size(session_id) == 0
    history.add_ai_message("Hello")
    assert get_session_history_size(session_id) == 1
    history.add_messages([HumanMessage(content=random_text) for _ in range(10)])
    assert get_session_history_size(session_id) == 11
    history.clear()
    assert get_session_history_size(session_id) == 0


@pytest.mark.usefixtures("clear_memory_store")
def test_clear_session_history():
    session_id = "123"
    history = get_session_history(session_id)
    history.add_user_message("Hello")
    assert len(get_session_history(session_id).messages) == 1
    _ = get_session_history("abc")
    assert len(clonellm.memory._store) == 2
    clear_session_history(session_id)
    assert len(clonellm.memory._store) == 1
    clear_session_history("abc")
    assert len(clonellm.memory._store) == 0
