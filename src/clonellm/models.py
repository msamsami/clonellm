import datetime
from typing import Any

from pydantic import BaseModel, Field

__all__ = ("PersonalityTraits", "CommunicationSample", "UserProfile")


class PersonalityTraits(BaseModel):
    openness: float = Field(ge=0, le=1)
    """
    Openness to experience, 0: not open, 1: very open.
    """
    conscientiousness: float = Field(ge=0, le=1)
    """
    Conscientiousness, 0: not conscientious, 1: very conscientious.
    """
    extraversion: float = Field(ge=0, le=1)
    """
    Extraversion, 0: not extraverted, 1: very extraverted.
    """
    agreeableness: float = Field(ge=0, le=1)
    """
    Agreeableness, 0: not agreeable, 1: very agreeable.
    """
    neuroticism: float = Field(ge=0, le=1)
    """
    Neuroticism, 0: not neurotic, 1: very neurotic.
    """


class CommunicationSample(BaseModel):
    context: str
    """
    Context of the communication, e.g., 'work', 'personal', 'social media', 'interview', 'meeting', etc.
    """
    audience_type: str
    """
    Type of the audience, e.g., 'colleague', 'friend', 'family member', 'client', 'boss', 'interviewer', etc.
    """
    formality_level: float = Field(ge=0, le=1)
    """
    Level of formality, 0: very informal, 1: very formal.
    """
    content: str
    """Communication content."""


class UserProfile(BaseModel):
    first_name: str
    middle_name: str | None = None
    last_name: str
    preferred_name: str | None = None
    prefix: str | None = None
    birth_date: datetime.date | str | None = None
    gender: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    phone_number: str | None = None
    email: str | None = None
    personality_traits: PersonalityTraits | None = None
    education_experience: Any = None
    work_experience: Any = None
    expertise: Any = None
    communication_samples: list[CommunicationSample] | None = None
    home_page: str | None = None
    github_page: str | None = None
    linkedin_page: str | None = None

    @property
    def full_name(self) -> str:
        return " ".join([self.first_name, self.middle_name or "", self.last_name])

    @property
    def age(self) -> int | None:
        return (datetime.date.today() - self.birth_date).days // 365 if isinstance(self.birth_date, datetime.date) else None
