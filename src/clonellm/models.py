import datetime
from typing import Any, Optional

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
    personality_traits: Optional[PersonalityTraits] = None
    education_experience: Optional[Any] = None
    work_experience: Optional[Any] = None
    expertise: Optional[Any] = None
    communication_samples: Optional[list[CommunicationSample]] = None
    home_page: Optional[str] = None
    github_page: Optional[str] = None
    linkedin_page: Optional[str] = None

    @property
    def full_name(self) -> str:
        return " ".join([self.first_name, self.middle_name or "", self.last_name])

    @property
    def age(self) -> Optional[int]:
        return (datetime.date.today() - self.birth_date).days // 365 if isinstance(self.birth_date, datetime.date) else None
