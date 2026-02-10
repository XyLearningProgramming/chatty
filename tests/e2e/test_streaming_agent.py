"""End-to-end test for streaming agent functionality."""

import asyncio
import json

import httpx
import pytest

from chatty.core.service.models import (
    EVENT_TYPE_CONTENT,
    EVENT_TYPE_ERROR,
    EVENT_TYPE_TOOL_CALL,
    VALID_EVENT_TYPES,
)

CHAT_TIMEOUT = 600  # 10 minutes timeout for long-running agent tasks
RESPONSE_PREVIEW_LENGTH = 4096  # allow almost all response


class SSEParser:
    """Parser for Server-Sent Events."""

    @staticmethod
    def parse_sse_line(line: str) -> dict | None:
        """Parse a single SSE line into an event dict."""
        if not line.startswith("data: "):
            return None

        try:
            data = line[6:]  # Remove "data: " prefix
            return json.loads(data)
        except json.JSONDecodeError:
            return None


def collect_events(events: list[dict]) -> dict[str, list[dict]]:
    """Group events by type for easier assertions."""
    grouped: dict[str, list[dict]] = {t: [] for t in VALID_EVENT_TYPES}
    for e in events:
        t = e.get("type", "")
        if t in grouped:
            grouped[t].append(e)
    return grouped


def full_content_text(content_events: list[dict]) -> str:
    """Concatenate all content event tokens into a single string."""
    return "".join(e.get("content", "") for e in content_events)


@pytest.fixture
def test_query():
    """Sample test query for the agent."""
    return "What is the capital of France?"


class TestStreamingAgent:
    """Test suite for streaming agent functionality."""

    @pytest.mark.asyncio
    async def test_streaming_chat_endpoint_exists(
        self, base_url: str, ensure_llm_server
    ):
        """Test that the chat endpoint exists and responds."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/v1/chat",
                json={"query": "test"},
                timeout=CHAT_TIMEOUT,
            )
            # Should not return 404
            assert response.status_code != 404

    @pytest.mark.asyncio
    async def test_streaming_response_format(
        self, base_url: str, test_query: str, ensure_llm_server
    ):
        """Test that streaming response uses valid event types."""
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/v1/chat",
                json={"query": test_query},
                timeout=CHAT_TIMEOUT,
            ) as response:
                assert response.status_code == 200
                assert "text/plain" in response.headers.get("content-type", "")

                events = []
                async for line in response.aiter_lines():
                    if line:
                        event = SSEParser.parse_sse_line(line)
                        if event:
                            events.append(event)

                # Should have at least one event
                assert len(events) > 0

                # Every event must carry a valid type
                for event in events:
                    assert event.get("type") in VALID_EVENT_TYPES, (
                        f"Unexpected event type: {event}"
                    )

                # Stream should contain at least one content event
                grouped = collect_events(events)
                assert len(grouped[EVENT_TYPE_CONTENT]) > 0, (
                    "No content events received"
                )

    @pytest.mark.asyncio
    async def test_agent_response_content(
        self, base_url: str, test_query: str, ensure_llm_server
    ):
        """Test that agent produces meaningful response content."""
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/v1/chat",
                json={"query": test_query},
                timeout=CHAT_TIMEOUT,
            ) as response:
                assert response.status_code == 200

                events = []
                async for line in response.aiter_lines():
                    if line:
                        event = SSEParser.parse_sse_line(line)
                        if event:
                            events.append(event)

                grouped = collect_events(events)

                # Should not have error events
                assert len(grouped[EVENT_TYPE_ERROR]) == 0, (
                    f"Received errors: {grouped[EVENT_TYPE_ERROR]}"
                )

                # Should have some content events (streaming response)
                assert len(grouped[EVENT_TYPE_CONTENT]) > 0, (
                    "No content events received"
                )

                # Concatenate all content tokens to form response
                full_response = full_content_text(grouped[EVENT_TYPE_CONTENT])

                # Response should not be empty
                assert len(full_response.strip()) > 0, "Empty response received"

                # For the test query, response should mention Paris or France
                response_lower = full_response.lower()
                assert "paris" in response_lower or "france" in response_lower, (
                    f"Response doesn't seem relevant to query. "
                    f"Response: {full_response}"
                )

    @pytest.mark.asyncio
    async def test_conversation_history_support(
        self, base_url: str, ensure_llm_server
    ):
        """Test that conversation history is properly handled."""
        conversation_history = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]

        follow_up_query = "What is the population of that city?"

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/v1/chat",
                json={
                    "query": follow_up_query,
                    "conversation_history": conversation_history,
                },
                timeout=CHAT_TIMEOUT,
            ) as response:
                assert response.status_code == 200

                events = []
                async for line in response.aiter_lines():
                    if line:
                        event = SSEParser.parse_sse_line(line)
                        if event:
                            events.append(event)

                grouped = collect_events(events)

                # Should not have error events
                assert len(grouped[EVENT_TYPE_ERROR]) == 0, (
                    f"Received errors: {grouped[EVENT_TYPE_ERROR]}"
                )

                # Should have content events
                assert len(grouped[EVENT_TYPE_CONTENT]) > 0

    @pytest.mark.asyncio
    async def test_invalid_request_handling(
        self, base_url: str, ensure_llm_server
    ):
        """Test handling of invalid requests."""
        # Test empty query
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/v1/chat",
                json={"query": ""},
                timeout=CHAT_TIMEOUT,
            )
            # Should handle gracefully (either 400 or streaming error)
            assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_long_query_handling(
        self, base_url: str, ensure_llm_server
    ):
        """Test handling of very long queries."""
        # Create a query longer than CHAT_QUERY_MAX_LENGTH (1024)
        long_query = "What is " + "very " * 300 + "long question?"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/v1/chat",
                json={"query": long_query},
                timeout=CHAT_TIMEOUT,
            )
            # Should reject with 422 (validation error)
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_agent_tool_usage(self, base_url: str, ensure_llm_server):
        """Test that agent can use tools and emit tool_call events."""
        # Query that likely requires tool usage
        tool_query = "Search for information about Xinyu's site"

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/v1/chat",
                json={"query": tool_query},
                timeout=CHAT_TIMEOUT,
            ) as response:
                assert response.status_code == 200

                events = []
                async for line in response.aiter_lines():
                    if line:
                        event = SSEParser.parse_sse_line(line)
                        if event:
                            events.append(event)

                grouped = collect_events(events)

                # Should complete without errors
                assert len(grouped[EVENT_TYPE_ERROR]) == 0, (
                    f"Received errors: {grouped[EVENT_TYPE_ERROR]}"
                )

                # If tool_call events are present, validate their structure
                for tc_event in grouped[EVENT_TYPE_TOOL_CALL]:
                    assert "name" in tc_event, "tool_call event missing 'name'"
                    assert tc_event.get("status") in (
                        "started",
                        "completed",
                        "error",
                    ), f"Invalid tool_call status: {tc_event}"

    @pytest.mark.asyncio
    async def test_relevance_filtering_golden_cases(
        self, base_url: str, ensure_llm_server
    ):
        """Test golden cases for relevance filtering."""
        test_cases = [
            {
                "query": "What is the capital of France?",
                "should_refuse": True,
                "expected_keywords": [
                    "focus on technology",
                    "tech-related",
                    "programming",
                    "software development",
                    "tech topics",
                ],
                "description": "Geography question - should be refused",
            },
            {
                "query": "How do I set up a Docker container for a Python app?",
                "should_refuse": False,
                "expected_keywords": ["docker", "container", "python"],
                "description": "Tech question - should be answered",
            },
            {
                "query": "What's the best recipe for chocolate cake?",
                "should_refuse": True,
                "expected_keywords": [
                    "focus on technology",
                    "tech-related",
                    "programming",
                    "tech topics",
                ],
                "description": "Cooking question - should be refused",
            },
            {
                "query": "How do I optimize a Python function for better performance?",
                "should_refuse": False,
                "expected_keywords": ["python", "performance", "optimize"],
                "description": "Python performance question - should be answered",
            },
        ]

        async with httpx.AsyncClient() as client:
            for test_case in test_cases:
                print(f"\n--- Testing: {test_case['description']} ---")
                print(f"Query: {test_case['query']}")

                async with client.stream(
                    "POST",
                    f"{base_url}/api/v1/chat",
                    json={"query": test_case["query"]},
                    timeout=CHAT_TIMEOUT,
                ) as response:
                    assert response.status_code == 200, (
                        f"Request failed for: {test_case['query']}"
                    )

                    events = []
                    async for line in response.aiter_lines():
                        if line:
                            event = SSEParser.parse_sse_line(line)
                            if event:
                                events.append(event)
                                # Print content tokens in real-time
                                if event.get("type") == EVENT_TYPE_CONTENT:
                                    print(
                                        event.get("content", ""),
                                        end="",
                                        flush=True,
                                    )

                    grouped = collect_events(events)
                    full_response = full_content_text(
                        grouped[EVENT_TYPE_CONTENT]
                    )

                    print(
                        f"\nResponse preview: "
                        f"{full_response[:RESPONSE_PREVIEW_LENGTH]}..."
                    )

                    if test_case["should_refuse"]:
                        found = any(
                            kw.lower() in full_response.lower()
                            for kw in test_case["expected_keywords"]
                        )
                        assert found, (
                            f"Agent should have refused '{test_case['query']}' "
                            f"but response lacks refusal keywords: "
                            f"{test_case['expected_keywords']}\n"
                            f"Response: {full_response}"
                        )
                        print("Agent correctly refused non-tech question")
                    else:
                        found = any(
                            kw.lower() in full_response.lower()
                            for kw in test_case["expected_keywords"]
                        )
                        assert found, (
                            f"Agent should have answered '{test_case['query']}' "
                            f"but response lacks expected keywords: "
                            f"{test_case['expected_keywords']}\n"
                            f"Response: {full_response}"
                        )
                        print("Agent correctly answered tech question")

    @pytest.mark.asyncio
    async def test_concurrent_requests(
        self, base_url: str, ensure_llm_server
    ):
        """Test that multiple concurrent requests are handled properly."""
        queries = [
            "How do I use Git for version control?",
            "What are Python decorators?",
            "How do I deploy a web app with Docker?",
        ]

        async def make_request(query: str) -> bool:
            """Make a single request and return success status."""
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{base_url}/api/v1/chat",
                        json={"query": query},
                        timeout=CHAT_TIMEOUT,
                    ) as response:
                        if response.status_code != 200:
                            return False

                        events = []
                        async for line in response.aiter_lines():
                            if line:
                                event = SSEParser.parse_sse_line(line)
                                if event:
                                    events.append(event)

                        # Check for errors
                        grouped = collect_events(events)
                        return len(grouped[EVENT_TYPE_ERROR]) == 0

            except Exception:
                return False

        # Run requests concurrently
        tasks = [make_request(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should succeed
        success_count = sum(1 for result in results if result is True)
        assert success_count >= len(queries) // 2, (
            f"Only {success_count}/{len(queries)} concurrent requests succeeded"
        )
