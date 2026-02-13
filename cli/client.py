"""API client for the Chatty API with SSE stream parsing."""

import json
import logging
from typing import AsyncIterator

import httpx

from .config import CLIConfig

logger = logging.getLogger(__name__)


class ChatAPIClient:
    """Client for interacting with the Chatty chat API."""

    def __init__(self, config: CLIConfig):
        """Initialize the API client."""
        self.config = config
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout

    async def chat(
        self, query: str, conversation_id: str | None = None
    ) -> AsyncIterator[dict]:
        """Send a chat request and stream events.

        Yields
        ------
        dict
            Parsed JSON event from the SSE stream.
        """
        url = self.config.chat_url
        payload = {"query": query}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        logger.debug(f"Making request to {url} with payload: {payload}")

        try:
            async with self.client.stream(
                "POST", url, json=payload, headers={"Accept": "text/plain"}
            ) as response:
                # Log response headers at DEBUG level
                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response headers: {dict(response.headers)}")

                # Check for errors
                if response.status_code == 429:
                    yield {
                        "type": "error",
                        "message": "Too many requests in flight. Try again later.",
                        "code": "RATE_LIMIT",
                    }
                    return

                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {
                        "type": "error",
                        "message": f"HTTP {response.status_code}: {error_text.decode()}",
                        "code": "HTTP_ERROR",
                    }
                    return

                # Extract conversation_id and trace_id from headers
                conversation_id_header = response.headers.get("X-Chatty-Conversation")
                trace_id_header = response.headers.get("X-Chatty-Trace")

                if conversation_id_header:
                    logger.debug(f"X-Chatty-Conversation: {conversation_id_header}")
                    # Yield a special metadata event with conversation_id
                    yield {
                        "type": "_metadata",
                        "conversation_id": conversation_id_header,
                    }
                if trace_id_header:
                    logger.debug(f"X-Chatty-Trace: {trace_id_header}")

                # Parse SSE stream
                # SSE format: "data: {...}\n\n" (each event ends with double newline)
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    # Process complete events (ending with \n\n)
                    while "\n\n" in buffer:
                        event_block, buffer = buffer.split("\n\n", 1)
                        # Process all lines in this event block
                        for line in event_block.split("\n"):
                            line = line.strip()
                            if not line:
                                continue
                            # SSE data lines start with "data: "
                            if line.startswith("data: "):
                                data_str = line[6:]  # Remove "data: " prefix
                                try:
                                    event = json.loads(data_str)
                                    yield event
                                except json.JSONDecodeError as e:
                                    logger.warning(
                                        f"Failed to parse SSE data: {data_str}, error: {e}"
                                    )

        except httpx.TimeoutException:
            yield {
                "type": "error",
                "message": "Request timed out.",
                "code": "TIMEOUT",
            }
        except httpx.ConnectError as e:
            yield {
                "type": "error",
                "message": f"Connection error: {str(e)}",
                "code": "CONNECTION_ERROR",
            }
        except Exception as e:
            logger.exception("Unexpected error during API request")
            yield {
                "type": "error",
                "message": f"Unexpected error: {str(e)}",
                "code": "UNEXPECTED_ERROR",
            }

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
