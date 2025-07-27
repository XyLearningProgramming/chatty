
golden:
	uv run pytest tests/e2e/test_streaming_agent.py::TestStreamingAgent::test_relevance_filtering_golden_cases -v -s

reload:
	uv run uvicorn chatty.app:app --host 0.0.0.0 --port 8080 --reload
