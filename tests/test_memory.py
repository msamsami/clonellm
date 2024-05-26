from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import clonellm.memory
from clonellm.memory import InMemoryHistory, get_session_history, clear_session_history


def test_history_add_message():
    history = InMemoryHistory()
    history.add_message(HumanMessage(content="Hello"))
    assert len(history.messages)
    assert history.messages == [HumanMessage(content="Hello")]


def test_history_last_message():
    history = InMemoryHistory(messages=[HumanMessage(content="one"), SystemMessage(content="two")])
    assert history.messages[-1] == SystemMessage(content="two")


def test_history_clear():
    history = InMemoryHistory(messages=[HumanMessage(content="one"), SystemMessage(content="two")])
    history.add_user_message("three")
    assert len(history.messages) == 3
    history.clear()
    assert len(history.messages) == 0


def test_get_session_history():
    session_id = "123"
    history = get_session_history(session_id)
    assert isinstance(history, InMemoryHistory)
    assert len(history.messages) == 0
    assert len(clonellm.memory._store) == 1
    assert session_id in clonellm.memory._store
    history.add_ai_message("Hello")
    assert len(get_session_history(session_id).messages) == 1
    assert get_session_history(session_id).messages[0] == AIMessage(content="Hello")
    _ = get_session_history("abc")
    assert len(clonellm.memory._store) == 2
    assert "abc" in clonellm.memory._store


def test_clear_session_history():
    clonellm.memory._store = {}
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
