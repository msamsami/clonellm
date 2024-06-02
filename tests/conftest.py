import random
import string

from dotenv import load_dotenv
import pytest


@pytest.fixture(autouse=True, scope="session")
def _load_dotenv():
    load_dotenv()


@pytest.fixture
def random_text(request) -> str:
    length = request.param if hasattr(request, "param") and request.param else 20
    return "".join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits + " ", k=length))
