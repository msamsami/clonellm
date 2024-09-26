from __future__ import annotations

import datetime
from typing import Any, Optional

from pydantic import BaseModel

__all__ = ("UserProfile",)


class UserProfile(BaseModel):
    first_name: str
    middle_name: Optional[str] = None
    last_name: str
    preferred_name: Optional[str] = None
    prefix: Optional[str] = None
    birth_date: Optional[datetime.date | str] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    education_experience: Optional[Any] = None
    work_experience: Optional[Any] = None
    expertise: Optional[Any] = None
    home_page: Optional[str] = None
    github_page: Optional[str] = None
    linkedin_page: Optional[str] = None

    @property
    def full_name(self) -> str:
        return " ".join([self.first_name, self.middle_name or "", self.last_name])

    @property
    def age(self) -> Optional[int]:
        return (datetime.date.today() - self.birth_date).days // 365 if isinstance(self.birth_date, datetime.date) else None
