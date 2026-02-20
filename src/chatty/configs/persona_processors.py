"""Processor reference types for persona config.

Used in YAML to name processors (string or structured with args).
Shared by sources, tools, and embed declarations.
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import BaseModel, Field


class ProcessorWithArgs(BaseModel):
    """A processor reference that carries configuration.

    Example YAML::

        processors:
          - name: truncate
            max_length: 4000
    """

    name: str = Field(description="Registered processor name")
    max_length: int | None = Field(
        default=None,
        description="For truncate processor: maximum character length",
    )


ProcessorRef = Annotated[
    Union[str, ProcessorWithArgs],
    Field(
        description="Processor reference: plain string name or "
        "structured {name, ...args}",
    ),
]


def processor_ref_name(ref: str | ProcessorWithArgs) -> str:
    """Extract the processor name from either form of ``ProcessorRef``."""
    if isinstance(ref, str):
        return ref
    return ref.name
