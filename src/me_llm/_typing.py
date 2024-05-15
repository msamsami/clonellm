import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

__all__ = ("Path", "PersonalInfo")


Path = str | Path


class PersonalInfo(BaseModel):
    first_name: str
    middle_name: Optional[str]
    last_name: str
    birth_date: Optional[datetime.date]
    gender: Optional[str]
    city: Optional[str]
    country: Optional[str]
    phone_number: Optional[str]
    email: Optional[str]
    home_page: Optional[str]
    github_page: Optional[str]
    linkedin_page: Optional[str]

    @property
    def full_name(self) -> str:
        return " ".join([self.first_name, self.middle_name or "", self.last_name])

    @property
    def age(self) -> Optional[int]:
        return (datetime.date.today() - self.birth_date).days // 365 if self.birth_date else None
