import random
import string

from dotenv import load_dotenv
import pytest

import clonellm.memory


@pytest.fixture(autouse=True, scope="session")
def _load_dotenv():
    load_dotenv()


@pytest.fixture
def random_text(request) -> str:
    length = request.param if hasattr(request, "param") and request.param else 20
    return "".join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits + " ", k=length))


@pytest.fixture
def clear_memory_store():
    """Fixture to execute asserts before and after a test is run"""
    # Setup
    clonellm.memory._store = {}

    yield

    # Teardown
    clonellm.memory._store = {}
