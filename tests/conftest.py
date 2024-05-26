from dotenv import load_dotenv
import pytest


@pytest.fixture(autouse=True, scope="session")
def _load_dotenv():
    load_dotenv()
