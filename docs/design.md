# Chat With Me Technical Design

1. Tech stack

API: python, uv, fastapi(async handlers, sse for streaming mode), pydantic, pydantic_setting, slowapi (for rate limiting), requests (wrap around model)
Agent: langchain
DB: sqlalchemy(to postgres, async), redis(to redis, for cache, lock, etc., async), vector db as postgres plugin???
Observalability: logging, prometheus_client, otlp (with span processors writing log and metrics and send span to remote)

2. API Design

3. Chatbot answer workflow, pseudo code

4. Resource (cpu, mem footprint) estimation

5. Implementation Details

4.1. aggressive caching strategy for common questions and similarity to them
4.2. cronjob to trigger refresh of resources, eg. blog rss, and even make stats of common questions and cache it;
4.3. minimize context at best effort, do not overload model
4.4. an outbox pattern with priorities and timeout for each conversation (real user conversation high, system refresh answer low)
4.5. ddd pattern to organize files
4.6. golden test and regression test whenver prompt changes for relevance evaluation step
