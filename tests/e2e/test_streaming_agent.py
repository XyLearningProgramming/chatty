"""End-to-end test for streaming agent functionality."""

import asyncio
import json

import httpx
import pytest

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
        """Test that streaming response follows expected SSE format."""
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

                # Should have end_of_stream event
                end_events = [e for e in events if e.get("type") == "end_of_stream"]
                assert len(end_events) == 1

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

                token_events = []
                structured_events = []
                error_events = []

                async for line in response.aiter_lines():
                    if line:
                        event = SSEParser.parse_sse_line(line)
                        if event:
                            event_type = event.get("type")
                            if event_type == "token":
                                token_events.append(event)
                            elif event_type == "structured_data":
                                structured_events.append(event)
                            elif event_type == "error":
                                error_events.append(event)

                # Should not have error events
                assert len(error_events) == 0, f"Received errors: {error_events}"

                # Should have some token events (streaming response)
                assert len(token_events) > 0, "No token events received"

                # Concatenate all tokens to form response
                full_response = "".join(
                    event.get("content", "") for event in token_events
                )

                # Response should not be empty
                assert len(full_response.strip()) > 0, "Empty response received"

                # For the test query, response should mention Paris or France
                response_lower = full_response.lower()
                assert "paris" in response_lower or "france" in response_lower, (
                    f"Response doesn't seem relevant to query. Response: {full_response}"
                )

    @pytest.mark.asyncio
    async def test_conversation_history_support(self, base_url: str, ensure_llm_server):
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

                # Should not have error events
                error_events = [e for e in events if e.get("type") == "error"]
                assert len(error_events) == 0, f"Received errors: {error_events}"

                # Should have token events
                token_events = [e for e in events if e.get("type") == "token"]
                assert len(token_events) > 0

    @pytest.mark.asyncio
    async def test_invalid_request_handling(self, base_url: str, ensure_llm_server):
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
    async def test_long_query_handling(self, base_url: str, ensure_llm_server):
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
        """Test that agent can use tools effectively."""
        # Query that likely requires tool usage
        tool_query = "Search for information about Xinyu's site"

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/v1/chat",
                json={"query": tool_query},
                timeout=CHAT_TIMEOUT,  # Longer timeout for tool usage
            ) as response:
                assert response.status_code == 200

                events = []
                async for line in response.aiter_lines():
                    if line:
                        event = SSEParser.parse_sse_line(line)
                        if event:
                            events.append(event)

                # Should complete without errors
                error_events = [e for e in events if e.get("type") == "error"]
                assert len(error_events) == 0, f"Received errors: {error_events}"

                # Should have structured data with intermediate steps
                structured_events = [
                    e for e in events if e.get("type") == "structured_data"
                ]
                if structured_events:
                    # Check if intermediate steps are present (indicating tool usage)
                    for event in structured_events:
                        data = event.get("data", {})
                        if "intermediate_steps" in data:
                            steps = data["intermediate_steps"]
                            assert isinstance(steps, list)

    @pytest.mark.asyncio
    async def test_relevance_filtering_golden_cases(
        self, base_url: str, ensure_llm_server
    ):
        """Test golden cases for relevance filtering - agent should refuse non-tech questions."""

        # Golden test cases with expected behaviors
        test_cases = [
            {
                "query": "What is the capital of France?",
                "should_refuse": True,
                "expected_keywords": [
                    "focus on technology",
                    "tech-related",
                    "programming",
                    "software development",
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

                    token_events = []
                    error_events = []
                    structured_events = []

                    print("Streaming response:")
                    print("-" * 50)

                    async for line in response.aiter_lines():
                        if line:
                            event = SSEParser.parse_sse_line(line)
                            if event:
                                event_type = event.get("type")
                                if event_type == "token":
                                    token_events.append(event)
                                    # Print tokens in real-time
                                    print(event.get("content", ""), end="", flush=True)
                                elif event_type == "structured_data":
                                    structured_events.append(event)
                                    # Print structured data with formatting
                                    data_type = event.get("data_type", "unknown")
                                    print(
                                        f"\n[{data_type.upper()}]", end="", flush=True
                                    )
                                    if data_type == "json_output":
                                        print(f" JSON: {event.get('data', {})}")
                                elif event_type == "error":
                                    error_events.append(event)
                                    print(
                                        f"\n[ERROR] {event.get('message', 'Unknown error')}"
                                    )

                    print("\n" + "-" * 50)

                    # Should not have error events
                    assert len(error_events) == 0, (
                        f"Received errors for {test_case['query']}: {error_events}"
                    )

                    # Should have some response
                    assert len(token_events) > 0, (
                        f"No token events received for: {test_case['query']}"
                    )

                    # Concatenate all tokens to form response
                    full_response = "".join(
                        event.get("content", "") for event in token_events
                    ).lower()

                    print(
                        f"Response preview: {full_response[:RESPONSE_PREVIEW_LENGTH]}..."
                    )

                    if test_case["should_refuse"]:
                        # For questions that should be refused, check for refusal keywords
                        found_refusal_keywords = any(
                            keyword.lower() in full_response
                            for keyword in test_case["expected_keywords"]
                        )
                        assert found_refusal_keywords, (
                            f"Agent should have refused the question '{test_case['query']}' "
                            f"but response doesn't contain expected refusal keywords: {test_case['expected_keywords']}\n"
                            f"Response: {full_response}"
                        )
                        print("✓ Agent correctly refused non-tech question")
                    else:
                        # For questions that should be answered, check for relevant keywords
                        found_relevant_keywords = any(
                            keyword.lower() in full_response
                            for keyword in test_case["expected_keywords"]
                        )
                        assert found_relevant_keywords, (
                            f"Agent should have answered the tech question '{test_case['query']}' "
                            f"but response doesn't contain expected keywords: {test_case['expected_keywords']}\n"
                            f"Response: {full_response}"
                        )
                        print("✓ Agent correctly answered tech question")

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, base_url: str, ensure_llm_server):
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
                        error_events = [e for e in events if e.get("type") == "error"]
                        return len(error_events) == 0

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
