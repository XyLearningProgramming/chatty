"""Lifespan dependency injection bridge.

``inject`` lets the FastAPI lifespan declare ``Depends()`` parameters
just like a route handler.  FastAPI's own ``solve_dependencies``
resolves the DAG automatically — no manual ordering required.

Based on https://github.com/fastapi/fastapi/discussions/11742
"""

from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.dependencies.utils import get_dependant, solve_dependencies


def get_app(request: Request) -> FastAPI:
    """Lifespan dependency — returns the ``FastAPI`` application."""
    return request.app


def inject(
    lifespan: Callable[..., Any],
) -> Callable[[FastAPI], Any]:
    """Resolve ``Depends()`` parameters for a lifespan function.

    Usage::

        @inject
        async def lifespan(
            app: FastAPI,
            _db: Annotated[None, Depends(build_db)],
        ):
            yield

    Each dependency generator owns its setup **and** teardown
    (via ``yield``).  ``AsyncExitStack`` runs cleanups in reverse
    resolution order on shutdown.

    ``app.dependency_overrides`` is respected, so tests can swap
    any lifespan dependency.
    """

    @asynccontextmanager
    async def wrapper(app: FastAPI):  # type: ignore[misc]
        request = Request(
            scope={
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/",
                "raw_path": b"/",
                "query_string": b"",
                "root_path": "",
                "headers": ((b"X-Request-Scope", b"lifespan"),),
                "client": ("localhost", 80),
                "server": ("localhost", 80),
                "state": app.state,
                "app": app,
            }
        )
        dependant = get_dependant(path="/", call=partial(lifespan, app))

        async with AsyncExitStack() as stack:
            solved = await solve_dependencies(
                request=request,
                dependant=dependant,
                async_exit_stack=stack,
                embed_body_fields=False,
                dependency_overrides_provider=app,
            )
            ctx = asynccontextmanager(lifespan)
            async with ctx(app, **solved.values):
                yield

    return wrapper
