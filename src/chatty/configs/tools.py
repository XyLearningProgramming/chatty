"""Tool configuration models.

Tools are a top-level config concern, separate from persona identity.
Each tool is configured uniformly with a type, a name, a description
(visible to the model), and type-specific hidden args (e.g. the URL
for a ``url`` tool).  The model only sees ``name`` and ``description``;
``args`` are baked into the tool at construction time.
"""

from typing import Any

from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    """Configuration for a single agent tool.

    All tools share this uniform schema regardless of type.

    Attributes:
        name: Shown to the model as the tool name.
        description: Shown to the model; should explain when to use it.
        tool_type: Registered builder type, e.g. ``'url'``.
        args: Hidden kwargs baked into the tool instance
            (e.g. ``url``, ``timeout``).  Not exposed to the model.
        processors: Ordered post-processing steps applied to the
            raw tool output (e.g. ``'html_head_title_meta'``).
    """

    name: str = Field(..., description="Tool name shown to the model")
    description: str = Field(
        ..., description="Tool description shown to the model"
    )
    tool_type: str = Field(
        ...,
        description="Registered tool type, e.g. 'url'",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Hidden args baked into the tool "
        "(e.g. url, timeout). Not exposed to the model.",
    )
    processors: list[str] = Field(
        default_factory=list,
        description="Ordered list of content processors to apply "
        "after the tool runs (e.g. 'html_head_title_meta').",
    )
