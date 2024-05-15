from abc import ABCMeta
from typing import Optional

from litellm import get_llm_provider, get_api_key


class LiteLLMMixin(metaclass=ABCMeta):
    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key

    @property
    def _api_key(self):
        if self.api_key:
            return self.api_key
        else:
            custom_llm_provider = get_llm_provider(model=self.model)[1]
            return get_api_key(llm_provider=custom_llm_provider, dynamic_api_key=self.api_key)
