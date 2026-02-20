"""E2E test configuration and fixtures."""

import multiprocessing
import time
from typing import Generator

import httpx
import pytest
import uvicorn


def is_server_running(port: int = 8080) -> bool:
    """Check if a server is running on the specified port by checking the /health endpoint."""
    try:
        response = httpx.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except (httpx.RequestError, httpx.ConnectTimeout):
        return False


@pytest.fixture(scope="session")
def server_port() -> int:
    """Get the port for the test server."""
    return 8080


def run_server(port: int) -> None:
    """Run the uvicorn server in a separate process."""
    uvicorn.run("chatty.app:app", host="0.0.0.0", port=port, log_level="info")


@pytest.fixture(scope="session")
def chat_server(server_port: int) -> Generator[int, None, None]:
    """
    A session-scoped fixture that starts the chat server if it's not already running.
    It tears down the server process after all tests in the session are complete.

    Returns the port number the server is running on.
    """
    if is_server_running(server_port):
        print(
            f"Server is already running on port {server_port}. Tests will proceed against \
            the existing server."
        )
        yield server_port
        return

    print(f"Starting chat server on port {server_port}...")

    # Start the server as a background process using multiprocessing
    process = multiprocessing.Process(target=run_server, args=(server_port,))
    process.start()

    # Wait for the server to be ready
    server_started = False
    for attempt in range(30):  # 30 seconds timeout
        if is_server_running(server_port):
            print(f"Chat server started successfully on port {server_port}.")
            server_started = True
            break
        time.sleep(1)

        # Check if process crashed
        if not process.is_alive():
            print("Server process crashed during startup.")
            pytest.fail("Server process crashed during startup.", pytrace=False)

    if not server_started:
        print("Server failed to start within timeout period.")
        pytest.fail(
            "Chat server did not start within the timeout period.", pytrace=False
        )

    yield server_port

    print("Tearing down chat server...")
    process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        print("Server did not terminate gracefully, killing it.")
        process.kill()
        process.join()
    print("Chat server torn down.")


@pytest.fixture
def base_url(chat_server: int) -> str:
    """Base URL for the chat API."""
    return f"http://localhost:{chat_server}"


@pytest.fixture
def llm_base_url() -> str:
    """Base URL for the external LLM server (assumed to be running)."""
    return "http://localhost:8000"


@pytest.fixture(scope="session")
def ensure_llm_server() -> None:
    """Ensure the external LLM server is running."""
    if not is_server_running(8000):
        pytest.skip(
            "External LLM server at localhost:8000 is not running. Please start it before running e2e tests."
        )
