"""Persona configuration models.

Persona describes the author's identity -- name, character traits,
areas of expertise, and knowledge sections.

Knowledge sections are the single source of truth for all persona
knowledge.  Each chat service interprets them differently:

- OneStep (tool-calling agent): sections with ``source_url`` are
  auto-derived into URL tools.
- RAG (retrieval): all sections are embedded for similarity search.
"""

from typing import Any

from jinja2 import Template
from pydantic import BaseModel, Field


class KnowledgeSection(BaseModel):
    """A knowledge section owned by the persona.

    Sections can carry inline ``content``, a ``source_url`` to fetch
    from, or both.  The ``description`` is optional -- if empty, it
    will be auto-generated when used as a tool description.

    Attributes:
        title: Section name (also used as tool name for OneStep).
        description: Optional human-written description.  When empty,
            an auto-generated description is used for tool binding.
        content: Inline text knowledge.
        source_url: URL to fetch content from (fetched by cron /
            at startup and used for both tools and RAG embedding).
        processors: Ordered post-processing steps applied to fetched
            content (e.g. ``'html_head_title_meta'``).
        args: Extra arguments forwarded to the tool builder
            (e.g. ``max_content_length``).
    """

    title: str = Field(..., description="Section name")
    description: str = Field(
        default="",
        description="Optional description; auto-generated if empty",
    )
    content: str = Field(
        default="",
        description="Inline text knowledge",
    )
    source_url: str = Field(
        default="",
        description="URL to fetch content from",
    )
    processors: list[str] = Field(
        default_factory=list,
        description="Ordered post-processing steps for fetched content",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra args forwarded to the tool builder",
    )


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


class PersonaConfig(BaseModel):
    """Author persona configuration (identity + knowledge)."""

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
    sections: list[KnowledgeSection] = Field(
        default_factory=list,
        description="Knowledge sections -- single source of truth for "
        "persona knowledge.  Each chat service interprets them "
        "differently by convention.",
    )

    def build_system_prompt(self, system_prompt_template: str) -> str:
        """Render the system prompt from persona fields.

        Renders the provided Jinja2 template with persona fields (name,
        character, expertise). Default values are handled by the template
        engine itself.

        Args:
            system_prompt_template: Jinja2 template string from PromptConfig.

        Returns a ready-to-use string that can be passed as the system
        message to an LLM.  Callers do not need to know about the
        internal field layout.
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
