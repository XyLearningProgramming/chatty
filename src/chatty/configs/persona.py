"""Persona configuration models.

Persona describes the author's identity — name, character traits,
and areas of expertise.  Tool configuration lives separately in
``chatty.configs.tools``.
"""

from pydantic import BaseModel, Field


class PersonaConfig(BaseModel):
    """Author persona configuration (identity only)."""

    name: str = Field(description="Author's name")
    character: list[str] = Field(
        default_factory=list,
        description="Key characteristics of the author, "
        "provided to system prompt",
    )
    expertise: list[str] = Field(
        default_factory=list,
        description="Author's areas of expertise, "
        "provided to system prompt",
    )
