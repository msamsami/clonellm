from abc import ABCMeta
from typing import Mapping, Optional

from litellm import get_llm_provider, get_api_key

__all__ = ("LiteLLMMixin",)


class LiteLLMMixin(metaclass=ABCMeta):
    _litellm_kwargs: Mapping = {}

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs) -> None:
        self.model = model
        self.api_key = api_key
        self._litellm_kwargs = kwargs

    @property
    def _llm_provider(self) -> str:
        return get_llm_provider(model=self.model)[1]

    @property
    def _api_key(self):
        if self.api_key:
            return self.api_key
        else:
            return get_api_key(llm_provider=self._llm_provider, dynamic_api_key=self.api_key)
