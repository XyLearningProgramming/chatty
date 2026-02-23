# Project Conventions & Tech Stack Reference

## Tech Stack

| Layer | Tech | Notes |
|---|---|---|
| Language | Python 3.13+ | Use modern syntax: `list[str]`, `str \| None`, not `Optional[List[str]]` |
| Framework | FastAPI | Async-first, `Annotated[X, Depends(...)]` for DI |
| Config | pydantic-settings + YAML | Multi-source: env > .env > yaml > defaults |
| LLM | LangChain + langchain-openai | `ChatOpenAI` as primary LLM wrapper |
| Agent | Direct tool-call loop | Native OpenAI `tools`/`tool_choice` via `astream` |
| HTTP client | httpx | Async with `httpx.AsyncClient` |
| HTML parsing | beautifulsoup4 | For content processors |
| Package mgr | uv | `uv sync`, `uv add`, `uv run` |
| Linting | ruff | Line length 88 |
| Testing | pytest + pytest-asyncio | E2E tests spin up a real server |

## Dependency Injection Pattern

Everything flows through FastAPI's `Depends()` with `Annotated` typing. Factory functions named `get_xxx` produce dependencies.

### Singleton factory

Use the `@singleton` decorator from `chatty.infra` to cache the first return value forever:

```python
from chatty.infra import singleton

@singleton
def get_thing() -> Thing:
    return Thing(...)
```

### Injecting config into a factory

Pull a sub-config via a lambda `Depends`:

```python
from typing import Annotated
from fastapi import Depends
from chatty.configs.config import get_app_config
from chatty.configs.system import LLMConfig

def get_llm(
    config: Annotated[LLMConfig, Depends(lambda: get_app_config().llm)],
) -> ChatOpenAI:
    return ChatOpenAI(base_url=config.endpoint, ...)
```

### Composing dependencies

Chain `get_xxx` functions — FastAPI resolves the graph automatically:

```python
@singleton
def get_chat_service(
    llm: Annotated[BaseLanguageModel, Depends(get_llm)],
    tools_registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
    config: Annotated[AppConfig, Depends(lambda: get_app_config())],
) -> ChatService:
    return SomeChatService(llm, tools_registry, config)
```

### Using a dependency in a route handler

```python
@router.post("/chat")
async def chat(
    chat_request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    ...
```

## Configuration

### Adding a new config field

1. Add the Pydantic model in `configs/system.py` (infra/service settings) or `configs/persona.py` (persona/content settings):

```python
class MyNewConfig(BaseModel):
    my_field: str = Field(default="value", description="What it does")
```

2. Wire it into `AppConfig` in `configs/config.py`:

```python
class AppConfig(BaseSettings):
    my_new: MyNewConfig = Field(default_factory=MyNewConfig)
```

3. Set values in `configs/*.yaml` under the matching key:

```yaml
my_new:
  my_field: "override_value"
```

### Config priority (highest wins)

`CHATTY_` env vars > `.env` file > `configs/*.yaml` > field defaults

Nested env vars use `__` delimiter: `CHATTY_LLM__ENDPOINT=http://...`

## Service / Business Logic Layer

### Abstract service pattern

Define an abstract base in `core/service/models.py`, implement in a sibling module:

```python
class ChatService(ABC):
    @abstractmethod
    def __init__(self, llm, tools_registry, config) -> None: ...

    @abstractmethod
    async def stream_response(self, question: str) -> AsyncGenerator[ServiceStreamEvent, None]: ...
```

### Registering a new service implementation

Add it to the `_known_agents` dict in `core/service/dependency.py`:

```python
_known_agents = {srv.chat_service_name: srv for srv in [OneStepChatService, MyNewService]}
```

Selection happens via `config.chat.agent_name` in YAML.

## Tools System

Tools are plain Pydantic `BaseModel` subclasses (no LangChain `BaseTool`). Each tool exposes:

- `to_openai_tool() -> ToolDefinition` — returns a typed OpenAI tool schema.
- `async execute(**kwargs) -> str` — runs the tool logic.
- `from_declaration(decl, sources, prompt) -> Self` — factory from persona config.

`ToolDefinition` and `FunctionDefinition` are Pydantic models in `core/service/tools/model.py` that mirror the OpenAI function-calling JSON spec.

### Adding a new tool type

1. Create a Pydantic `BaseModel` subclass implementing `to_openai_tool()`, `execute()`, and `from_declaration()`:

```python
class MyTool(BaseModel):
    name: str
    # ...

    def to_openai_tool(self) -> ToolDefinition:
        return ToolDefinition(function=FunctionDefinition(name=self.name, ...))

    async def execute(self, **kwargs: str) -> str:
        ...

    @classmethod
    def from_declaration(cls, declaration, sources, prompt) -> Self:
        ...
```

2. Wire it into `ToolRegistry._build_tools` in `core/service/tools/registry.py`.
3. Configure it in `configs/persona.yaml`:

```yaml
persona:
  tools:
    - name: my tool
      tool_type: my_type
      args:
        key: value
```

### Adding a new content processor

1. Implement the `Processor` protocol in `core/service/tools/processors.py`:

```python
class MyProcessor(Processor):
    processor_name = "my_processor"

    def process(self, content: str) -> str:
        return content.upper()
```

2. Register in `ToolRegistry._known_processors`.
3. Reference by name in persona YAML: `processors: [my_processor]`.

## API Layer

### SSE streaming response pattern

Routes return `StreamingResponse` wrapping an async generator. Events are JSON lines prefixed with `data: `:

```python
async def stream(request, service) -> AsyncGenerator[str, None]:
    async for event in service.stream_response(request.query):
        yield f"data: {json.dumps(jsonable_encoder(event))}\n\n"
```

### Request validation

Use Pydantic models with `Field` constraints:

```python
class ChatRequest(BaseModel):
    query: str = Field(max_length=1024)
    conversation_history: list[ChatMessage] = Field(default_factory=list)
```

## Testing Conventions

- Unit tests in `tests/`, E2E in `tests/e2e/`.
- E2E tests use a session-scoped fixture that auto-starts the server process.
- Run unit: `uv run pytest tests/ -v --ignore=tests/e2e/`
- Run e2e: `uv run pytest tests/e2e/ -v`
- Tests should be simple. Cover most code but don't over-assert on third-party return values.
- Add dev-only deps with `uv add --dev`.

## DB / External Connections (planned)

When adding database or Redis connections, follow the same `get_xxx` + `@singleton` + `Depends` pattern:

```python
@singleton
def get_db_pool(
    config: Annotated[ThirdPartyConfig, Depends(lambda: get_app_config().third_party)],
) -> AsyncEngine:
    return create_async_engine(config.vector_database_uri)

@singleton
def get_redis(
    config: Annotated[ThirdPartyConfig, Depends(lambda: get_app_config().third_party)],
) -> Redis:
    return Redis.from_url(config.redis_uri)
```

Initialize/teardown in the lifespan context manager in `app.py`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    pool = get_db_pool()  # trigger singleton creation
    yield
    await pool.dispose()
```

## Quick Reference

| I want to... | Do this |
|---|---|
| Add a dependency | Write `get_xxx()` function, use `@singleton` if needed |
| Inject into handler | `param: Annotated[Type, Depends(get_xxx)]` |
| Inject sub-config | `Depends(lambda: get_app_config().section)` |
| Add a config field | Pydantic model in `system.py`/`persona.py`, wire in `AppConfig` |
| Add a tool | Pydantic `BaseModel` with `to_openai_tool()` + `execute()` + wire in `ToolRegistry._build_tools` |
| Add a processor | `Processor` protocol + register in `ToolRegistry._known_processors` |
| Add a service | Subclass `ChatService`, add to `_known_agents` |
| Run dev server | `make dev` or `uv run uvicorn chatty.app:app --reload` |
| Add a package | `uv add pkg` (or `uv add --dev pkg` for test-only) |
