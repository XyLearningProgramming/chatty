from langchain.tools import Tool


def emit_structured(type: str, payload: dict) -> dict:
    return {"type": type, "payload": payload}


emit_structured_tool = Tool(
    name="emit_structured",
    func=emit_structured,
    description=(
        "Emit any structured data. Arguments with type describing data: "
        "type (str), payload (dict)"
    ),
)
