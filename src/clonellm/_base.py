from abc import ABCMeta
from typing import Any, Optional, cast

from litellm import get_api_key, get_llm_provider

__all__ = ("LiteLLMMixin",)


class LiteLLMMixin(metaclass=ABCMeta):
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        self.model = model
        self.api_key = api_key
        self._litellm_kwargs: dict[str, Any] = kwargs

    @property
    def _llm_provider(self) -> str:
        return cast(str, get_llm_provider(model=self.model)[1])

    @property
    def _api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        else:
            return cast(Optional[str], get_api_key(llm_provider=self._llm_provider, dynamic_api_key=self.api_key))
