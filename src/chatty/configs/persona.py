"""Persona configuration.

Persona describes the author's identity and three layers of knowledge
config: sources (content acquisition), tools (agent actions), and
embed (RAG). This module is the single entry point; supporting types
live in persona_processors, persona_sources, and persona_actions.

Re-exports all persona types so callers can use::

    from chatty.configs.persona import PersonaConfig, KnowledgeSource, ...
"""

from __future__ import annotations

from jinja2 import Template
from pydantic import BaseModel, Field, model_validator

from .persona_actions import EmbedDeclaration, ToolDeclaration
from .persona_processors import (
    ProcessorRef,
    ProcessorWithArgs,
    processor_ref_name,
)
from .persona_sources import KnowledgeSource

__all__ = [
    "EmbedDeclaration",
    "KnowledgeSource",
    "PersonaConfig",
    "ProcessorRef",
    "ProcessorWithArgs",
    "ToolDeclaration",
    "processor_ref_name",
]


class PersonaConfig(BaseModel):
    """Author persona configuration (identity + knowledge layers)."""

    name: str = Field(description="Author's name")
    character: list[str] = Field(
        default_factory=list,
        description="Key characteristics of the author",
    )
    expertise: list[str] = Field(
        default_factory=list,
        description="Author's areas of expertise",
    )

    sources: dict[str, KnowledgeSource] = Field(
        default_factory=dict,
        description="Knowledge sources keyed by id",
    )
    tools: list[ToolDeclaration] = Field(
        default_factory=list,
        description="Tools exposed to the agent",
    )
    embed: list[EmbedDeclaration] = Field(
        default_factory=list,
        description="Embed actions for RAG retrieval",
    )

    @model_validator(mode="after")
    def _validate_references(self) -> PersonaConfig:
        """Ensure all tool and embed references point to real sources."""
        source_ids = set(self.sources)

        for tool in self.tools:
            unknown = set(tool.sources) - source_ids
            if unknown:
                raise ValueError(
                    f"Tool '{tool.name}' references unknown "
                    f"sources: {unknown}"
                )

        for entry in self.embed:
            if entry.source not in source_ids:
                raise ValueError(
                    f"Embed references unknown source: "
                    f"'{entry.source}'"
                )

        return self

    def build_system_prompt(self, system_prompt_template: str) -> str:
        """Render the system prompt from persona fields.

        Args:
            system_prompt_template: Jinja2 template string.

        Returns a ready-to-use system message string.
        """
        if not system_prompt_template:
            raise ValueError(
                "System prompt template is required. "
                "Ensure prompt.system_prompt is configured."
            )

        template = Template(system_prompt_template.strip())

        persona_character = (
            ", ".join(self.character) if self.character else None
        )
        persona_expertise = (
            ", ".join(self.expertise) if self.expertise else None
        )

        return template.render(
            persona_name=self.name,
            persona_character=persona_character,
            persona_expertise=persona_expertise,
        )
