from typing import Any

from pydantic import BaseModel, Field

# class ToolSchema(BaseModel):
#     name: str = Field(..., description="Schema field name")
#     type: str = Field(
#         ...,
#         description="JSON basic types, eg. \
#                       string, integer, list[string], etc.",
#     )
#     description: str = Field(
#         ...,
#         description="Field description, used in prompt",
#     )


# class ToolExample(BaseModel):
#     """Example input/output for a tool. Input and output should
#     always be of JSON type."""

#     input: dict[str, Any] | None = Field(
#         None,
#         description="Input example for the tool",
#     )
#     output: dict[str, Any] | None = Field(
#         None,
#         description="Expected output example from the tool",
#     )


class PersonaToolConfig(BaseModel):
    """Configuration for a tool available as extra info of the author persona."""

    name: str = Field(..., description="Tool name")
    description: str | None = Field(None, description="Tool description")
    tool_type: str = Field(
        ...,
        description="Predefined type of tool, e.g., 'url', 'doc', 'text', "
        "'command', etc.",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Fixed arguments for the tool, e.g., URL, command, etc.",
    )
    processors: list[str] = Field(
        default_factory=list,
        description="Predefined list of processors to apply to the tool content "
        "after fetching, e.g., 'vectorizer', etc.",
    )
    arg_schema: dict[str, Any] | None = Field(
        None, description="Ordered list of expected input arguments"
    )


class PersonaConfig(BaseModel):
    """Author persona configuration."""

    name: str = Field(description="Author's name")
    character: list[str] = Field(
        default_factory=list,
        description="Key characteristics of the author, provided to system prompt",
    )
    expertise: list[str] = Field(
        default_factory=list,
        description="Author's areas of expertise, provided to system prompt",
    )
    tools: list[PersonaToolConfig] = Field(
        default_factory=list, description="Available tools"
    )
