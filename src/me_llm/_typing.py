from __future__ import annotations
import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

__all__ = ("Path", "PersonalInfo")


Path = str | Path


class PersonalInfo(BaseModel):
    first_name: str
    middle_name: Optional[str] = None
    last_name: str
    birth_date: Optional[datetime.date] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    expertise: Optional[list[str]] = None
    home_page: Optional[str] = None
    github_page: Optional[str] = None
    linkedin_page: Optional[str] = None

    @property
    def full_name(self) -> str:
        return " ".join([self.first_name, self.middle_name or "", self.last_name])

    @property
    def age(self) -> Optional[int]:
        return (datetime.date.today() - self.birth_date).days // 365 if self.birth_date else None
