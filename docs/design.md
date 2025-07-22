# Technical Design: Chat With Me

## 1. Introduction

### 1.1. Overview

This document provides a detailed technical design for the "Chat With Me" project, a persona-driven chatbot operating under resource constraints. It outlines the architecture, technology stack, and implementation details necessary to deliver the MVP.

### 1.2. Core Objective

The primary goal is to build a chatbot that accurately impersonates a professional persona by leveraging a Retrieval-Augmented Generation (RAG) pipeline. The system must be scalable, efficient, and maintain user privacy.

### 1.3. High-Level Architecture

The system will consist of the following key components:
- A **Backend Service** to handle user requests and orchestrate the RAG pipeline.
- A **Knowledge Ingestion** process to populate the vector database.
- A **Vector Database** to store and retrieve knowledge embeddings for the ingestion.
- A **Model Server** to generate responses.
- A **Caching Layer** to improve performance.

---

## 2. Technology Stack & Resource Estimation

### 2.1. Core Components

| Component             | Technology                               | Rationale                                                                                                                                                           |
| --------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend Service**   | **Python + FastAPI**                     | High-performance, asynchronous, and ideal for I/O-bound operations. A lightweight choice for a resource-constrained environment.                                      |
| **Vector Database**   | **PostgreSQL + pgvector**                | Leverages existing provisioned infrastructure. A robust and scalable solution for vector storage and semantic search.                                                 |
| **Caching**           | **Redis**                                | Industry-standard in-memory data store, perfect for caching frequently accessed data and reducing latency.                                                          |
| **Model Server**      | **Qwen3 0.6B**                           | As specified in the requirements.                                                                                                                                   |
| **Knowledge Ingestion**| **Python + LangChain**        | These libraries provide robust tools for document loading, chunking, and vectorization, simplifying the RAG pipeline implementation.                                 |

### 2.2. Resource Allocation

| Service               | CPU (Approx.) | Memory (Approx.) | Concurrency | Notes                  |
| --------------------- | ------------- | ---------------- | ----------- | ---------------------- |
| **Chatbot Service**   | ~0.5 Core     | ~400 MiB         | N/A         | FastAPI application    |
| **Model Server**      | ~3 Cores      | ~800 MiB         | 1           | Qwen3 0.6B             |
| **Vector Database**   | ~0.5 Core     | ~500 MiB         | N/A         | Shared PostgreSQL      |
| **Redis**             | ~0.25 Core    | ~200 MiB         | N/A         | Shared Redis instance  |

### 2.3. Missing Components & Considerations

| Component                    | Recommendation                   | Rationale                                                                                                                              |
| ---------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Frontend Application**     | **To Be Integrated with Contact Page of My Site**                 | For a simple and effective proof-of-concept.                                                                                           |
| **Deployment & Orchestration** | **Helm Chart**      | As an independent Helm chart for this service alone.            |
| **CI/CD Pipeline**           | **GitHub Actions**| Automates testing and deployment, ensuring code quality, result accuracy, and a streamlined release process.                                             |
| **Monitoring & Logging**     | **Prometheus, Loki, OTPL**| Provides visibility into system health and performance, with logging, metrics, and otlp components integrated into the app.                           |

---

## 3. System Design & Implementation

### 3.1. Unified RAG Pipeline

The entire query-answering process is designed as a single, unified pipeline built with LangChain. This pipeline intelligently routes a user's query through a series of conditional steps—starting with a semantic cache check, followed by a relevance estimation, and finally, the full knowledge retrieval and generation process. This approach ensures maximum efficiency and response quality.

**Conceptual Flow within the LangChain Pipeline:**

1.  **Semantic Cache Branch:** The pipeline first attempts to find a semantically similar question in the cache, as described in the **Semantic Caching Strategy** (Section 3.2). If a high-confidence match is found, it immediately returns the cached answer, ending the process.
2.  **Relevance Check Branch:** If the cache lookup fails, the pipeline routes the query to a relevance-checking sub-chain. If the query is deemed irrelevant, a polite, predefined refusal is returned, and the process stops.
3.  **Full RAG Branch (Default):** If the query passes the relevance check, it is sent to the main RAG sub-chain, which retrieves context from the knowledge base, truncate conversation history if needed, and generates a final answer.
4.  **Cache Update:** The newly generated answer is then automatically added to the dynamic tier of the semantic cache for future use.

**Pseudo-code Illustrating the Unified Pipeline with LangChain:**

This example uses the LangChain Expression Language (LCEL) with `RunnableBranch` to demonstrate how these conditional steps are chained together.

```python
# Pseudo-code for the unified pipeline

from langchain_core.runnables import RunnableBranch, RunnableLambda

# Assume the following components are defined:
# - semantic_cache_lookup: A runnable that returns a cached answer or None (from Sec 3.2).
# - relevance_check_chain: A runnable that returns 'relevant' or 'irrelevant'.
# - full_rag_chain: The main RAG runnable for generation.
# - update_dynamic_cache: A function to save a new Q&A pair.

# 1. Define the function that will be the final step of the RAG chain
def rag_and_cache(input_dict):
    # Run the main RAG chain
    answer = full_rag_chain.invoke(input_dict)
    # Update the dynamic cache with the new answer
    update_dynamic_cache(query=input_dict["query"], answer=answer)
    return answer

# 2. Create the conditional branches
unified_pipeline = RunnableBranch(
    # First, try the semantic cache. The lookup function itself returns the answer if found.
    (lambda x: semantic_cache_lookup(x["query"]) is not None, 
     RunnableLambda(lambda x: semantic_cache_lookup(x["query"]))),
    
    # Second, check for relevance
    (lambda x: "irrelevant" in relevance_check_chain.invoke(x), 
     RunnableLambda(lambda x: "I can only answer questions about my professional experience.")),
    
    # Default to the full RAG pipeline and cache the result
    RunnableLambda(rag_and_cache)
)

# 3. Invoke the single, unified pipeline
final_result = unified_pipeline.invoke({"query": user_query})
```

### 3.2. Outbox Pattern with Priority Queues

To efficiently manage concurrent LLM requests and handle resource constraints, an outbox pattern will be implemented within each process. This component serves as a queue management system that controls access to the limited LLM resources while handling both user-initiated requests and internal pipeline operations.

**Key Components:**

1. **Priority Queue System:**
   - **High Priority:** User-facing chat requests requiring immediate response
   - **Low Priority:** Golden dataset generation and background processing (cronjob-style)

2. **Distributed Concurrency Control:**
   - Uses Redis-based distributed locks to ensure only a configured number of concurrent LLM requests (e.g., 3-5 based on resource limits)
   - Implements exponential backoff for request retries
   - Queue position tracking for user feedback
   - Timeout mechanism for each element in queue

3. **Request Processing Flow:**
   ```python
   # Pseudo-code for outbox pattern
   
   class OutboxManager:
       def __init__(self, redis_client, max_concurrent=3):
           self.redis = redis_client
           self.max_concurrent = max_concurrent
           self.priority_queue = PriorityQueue()
       
       async def enqueue_request(self, request, writer, priority="high"):
           # Add request to priority queue with metadata
           queue_item = {
               "id": generate_uuid(),
               "request": request,
               "priority": priority,
               "timestamp": datetime.utcnow(),
               "timeout": datetime.utcnow(),
               "source": request.source,  # "user" or "pipeline"
               "writer": writer,
           }
           await self.priority_queue.put((priority_value, queue_item))
           
       async def process_queue(self):
           while True:
               # Check available slots using distributed lock
               if await self.acquire_slot():
                   try:
                       # Get highest priority item
                       _, queue_item = await self.priority_queue.get()
                       
                       # Process request
                       result = await self.execute_llm_request(queue_item)
                       
                       # Write back to source
                       await self.write_back_result(queue_item, result)
                       
                   finally:
                       await self.release_slot()
   ```

4. **Write-Back Mechanism:**
   - User requests: Stream results directly to WebSocket/SSE connection
   - Pipeline requests: Store results in vector database

**Integration with Existing Pipeline:**

The outbox pattern integrates seamlessly with the unified RAG pipeline by intercepting LLM calls and routing them through the priority queue system, ensuring fair resource allocation and system stability.

### 3.3. Semantic Caching Strategy

To provide near-instant responses to common queries, a semantic caching layer will be implemented. Unlike a traditional key-value cache, this approach uses the vector database to find and return answers to questions that are semantically similar, not just identical.

**Flow:**

1.  **Vectorize Query:** The incoming user query is vectorized.
2.  **Semantic Search in Cache:** The system searches a dedicated "cached questions" table in the `pgvector` database to find the most similar query vector.
3.  **Threshold Check:** If a cached question is found within a predefined similarity threshold (e.g., cosine similarity > 0.95), the corresponding answer is retrieved from a simple key-value store (like Redis) and returned.
4.  **Cache Miss:** If no sufficiently similar question is found, the query proceeds to the Relevance Estimation step of the RAG pipeline.

**Cache Tiers and Management:**

The cache will be managed in two tiers with distinct admission and eviction policies to ensure both quality and performance.

-   **Tier 1: Golden Questions (Static Cache):**
    -   **Content:** A curated set of high-quality, frequently asked questions and their ideal answers.
    -   **Admission:** Pre-loaded into the cache as part of the deployment process. This set is defined in the project's "golden dataset."
    -   **Eviction:** These entries are permanent and are not subject to eviction.

-   **Tier 2: Auto-Discovered Questions (Dynamic Cache):**
    -   **Admission Policy:** To prevent the cache from being polluted by one-off queries, a new question-answer pair is only admitted to the dynamic cache after the same semantic query has been asked a certain number of times (e.g., 3 times). A simple counter in Redis can track query frequency. That is to say, there is another frequency counter of questions maintained.
    -   **Eviction Policy:** The dynamic cache will be managed with a combination of two policies:
        1.  **Time-to-Live (TTL):** Each entry will have a TTL (e.g., 24 hours) to automatically remove stale entries.
        2.  **Least Frequently Used (LFU):** When memory limits are approached, the system will evict the least frequently used items first. Redis provides built-in support for LFU eviction, as set in redis config.

**Pseudo-code for Semantic Cache Lookup:**

```python
# Pseudo-code for semantic cache lookup

from langchain_community.vectorstores.pgvector import PGVector

# 1. Setup the Vector Store for the cache
cached_questions_vectorstore = PGVector(
    connection_string="postgresql://...",
    collection_name="cached_questions", # Use a dedicated table for cached Q&A
    embedding_function=embedding_function
)

# 2. Search for a similar question
similar_docs = cached_questions_vectorstore.similarity_search_with_score(
    query=user_query,
    k=1
)

# 3. Check if a similar question was found within the threshold
if similar_docs and similar_docs[0][1] > 0.95: # Score represents similarity
    question_id = similar_docs[0][0].metadata["question_id"]
    cached_answer = redis_client.get(question_id)
    if cached_answer:
        # TODO: mark frequency += 1
        return cached_answer

# 4. If no cache hit, proceed to the RAG pipeline...
# TODO: mark frequency += 1; if frequent, then add to cache
```

### 3.4. LLM Response Token Processing and Structured Output

A lightweight token-level state machine is appended to streaming LLM responses, providing brief token buffering and structured output detection during the streaming process.

**Token Processing State Machine:**

```python
# Pseudo-code for token processing state machine
class TokenProcessor:
    def __init__(self):
        self.state = TokenState.STREAMING
        self.buffer = ""
        self.json_buffer = ""
        self.structured_data = None
        
    async def process_token(self, token: str, writer):
        """Process each token from LLM stream"""
            # Check for JSON block start markers
            self.buffer += token
            if "```json" in self.buffer.lower():
                self.state = TokenState.DETECTING_JSON
                self.json_buffer = ""
                # Send buffered tokens before JSON block
                await self._send_content_tokens(writer)
                self.buffer = ""
            else:
                # Send token immediately if buffer gets too long
                if len(self.buffer) > 10:
                    await self._send_content_tokens(writer)
        elif self.state == TokenState.DETECTING_JSON:
            self.json_buffer += token
            if "\n" in token:  # Found newline after ```json
                self.state = TokenState.BUFFERING_JSON
        elif self.state == TokenState.BUFFERING_JSON:
            self.json_buffer += token
            if "```" in token:  # End of JSON block
                await self._parse_and_send_structured_data(writer)
                self.state = TokenState.STREAMING
                
```

**Key Features:**

1. **Minimal Buffering:** Only buffers tokens when detecting structured output patterns
2. **Real-time Streaming:** Most tokens sent immediately to maintain responsiveness  
3. **Pattern Detection:** Uses state machine to detect ```json blocks during streaming
4. **Fallback Handling:** Invalid JSON falls back to regular content streaming
5. **No Persistent Cache:** Processes tokens in real-time without storing responses

### 3.5. API Design

The API is designed to be simple, stateless, and streaming-first. This provides a superior user experience by displaying the response as it is generated. The server will send a stream of Server-Sent Events (SSE), where each event is a JSON object with a defined `type`.

**Endpoint:** `POST /api/v1/chat`

**Request Body:**

```json
{
  "query": "Can you write a blog post about the future of AI?",
  "conversation_history": []
}
```

**Response Stream:**

The response is a stream of JSON objects, sent line by line. The frontend client will listen for these events and process them accordingly.

**Example Stream Sequence:**

1.  **Token Stream:** The natural language response is streamed token by token.
    ```json
    {"type": "token", "content": "Of course"}
    {"type": "token", "content": ", I can"}
    {"type": "token", "content": " certainly"}
    {"type": "token", "content": " write a"}
    {"type": "token", "content": " blog post"}
    {"type": "token", "content": " about that."}
    ```

2.  **Structured Data:** If the response includes structured data, it is sent as a single, complete JSON object. This ensures the frontend receives a valid, parseable object.
    ```json
    {"type": "structured_data", "data": {"title": "The Future of AI", "url": "/blog/ai-future"}}
    ```

3.  **End of Stream:** A final message indicates that the response is complete.
    ```json
    {"type": "end_of_stream"}
    ```

This event-driven approach provides a clear, robust, and easily extensible protocol for communication between the backend and frontend.

### 3.6. Non-Functional Requirements

-   **Statelessness and Privacy:** The service will be stateless. No user login is required, and no conversation data or personally identifiable information (PII) will be stored. This ensures user privacy and simplifies scalability.

-   **Scalability:** The architecture is designed for horizontal scalability. The stateless nature of the backend service allows for multiple instances to be run behind a load balancer.

-   **Conversational Context:** To manage token limits and memory usage, the chatbot will maintain conversation context using a **sliding window** approach. The `conversation_history` from the request body will be truncated to a fixed size by dropping the oldest messages as the conversation grows.

-   **Rate Limiting:** A distributed rate limiter will be implemented using Redis to protect the service from abuse and ensure fair usage.

### 3.7. Configuration Management

Key parameters will be externalized to config files to allow for easy modification without code changes.

**Example `author.yaml`:**

```yaml
system_prompt: "You are a helpful assistant..."
persona_details: "I am a software engineer with 10 years of experience..."
resume_uri: ""
blog_site_rss_uri: ""

**Example `config.yaml`:**

```yaml
model_server_endpoint: "http://localhost:8080/api/v1/chat/completions"
vector_database_uri: "postgresql://user:password@host:port/database"
redis_uri: "redis://localhost:6379"
```

---

### 3.8. File & Directory Structure

To ensure a clean separation of concerns while maintaining simplicity, the project will adopt a pragmatic structure inspired by Domain-Driven Design (DDD). The core logic of the application will be orchestrated within a central `ChatService`, which assembles and executes the RAG pipeline using components from the infrastructure layer.

This structure clearly separates the web-facing API, the core application logic, and the external infrastructure integrations, making the system easy to navigate and maintain.

**Proposed File Structure:**

```
alembic/                    # Database migrations
├── versions/
└── env.py

src/chat_with_me/
├── api/
│   └── v1/
│       └── chat.py         # FastAPI endpoint. Handles HTTP, SSE streaming, and calls ChatService.
├── core/
│   └── config.py           # Loads and provides access to configuration.
├── domain/
│   └── models.py           # Core data structures (e.g., API request/response models).
├── infrastructure/
│   ├── cache.py            # Implements the semantic cache lookup and update logic.
│   ├── llm.py              # Client for interacting with the Qwen model server.
│   └── vector_store.py     # Manages interaction with the pgvector database.
├── services/
│   └── chat_service.py     # **Orchestrator.** Assembles and executes the full RAG pipeline.
└── main.py                 # Application entry point (creates FastAPI app).

configs/
├── author.yaml             # Persona and prompt configuration.
└── config.yaml             # Application and infrastructure configuration.

tests/
├── services/
│   └── test_chat_service.py
└── test_api.py
```

**Component Responsibilities:**

-   **`main.py`**: Initializes the FastAPI application and includes the API routers.
-   **`api/v1/chat.py`**: The thinnest layer. It's responsible for handling the incoming HTTP request, validating the payload, and streaming the response back to the client using Server-Sent Events (SSE). It calls the `ChatService` to get the response stream.
-   **`services/chat_service.py`**: The heart of the application. It imports the necessary building blocks from `infrastructure` (the LLM client, cache, vector store) and assembles the complete, conditional RAG pipeline (the `RunnableBranch` from the pseudo-code). It exposes a simple method for the API layer to call, hiding the complexity of the pipeline.
-   **`infrastructure/`**: Contains all the code that deals with external systems. Each file provides a clean client or interface (e.g., `llm.py` has a function to invoke the model, `cache.py` has a function to perform a cache lookup).
-   **`domain/models.py`**: Holds simple data structures, primarily for defining the shape of data passed between layers, like the API request body.
-   **`core/config.py`**: Handles loading configuration from the `configs/` directory so that it's available to the rest of the application.

---

## 4. Quality Assurance & Testing

### 4.1. Golden Dataset

"Golden datasets" will be created, containing representative sets of questions and their ideal answers. This dataset will serve as the benchmark for evaluating the performance of different steps of the RAG pipeline.

### 4.2. Regression Testing

Automated regression tests will run against the golden dataset on every commit to prevent regressions in response quality.

**CI/CD Integration and Testing Strategy:**

To ensure tests run efficiently, the CI pipeline (GitHub Actions) will use a small, fast, open-source model (`Qwen2-0.5B-Instruct`) instead of the production model. This provides a strong directional signal on quality without high resource usage.

The strategy is as follows:
1.  **Model Caching:** The CI workflow will use a tool like Ollama to run the model. The downloaded model files will be cached using `actions/cache`. This dramatically speeds up subsequent test runs, as the model is restored from the cache instead of being re-downloaded.
2.  **Golden Dataset Execution:** The test suite will execute the golden dataset against the RAG pipeline using the cached `Qwen2-0.5B-Instruct` model.
3.  **Response Evaluation:** Generated answers will be compared against the ideal answers in the dataset using semantic similarity and keyword matching.
4.  **Build Failure:** The CI build will fail if the response quality drops below a predefined threshold, providing immediate feedback and preventing regressions from being merged.

---

## 5. Future Development

The following features are planned for post-MVP releases:

-   **User Accounts:** Introduce user authentication to enable conversation history and personalized context.
-   **Enhanced Functionality:**
    -   Allow users to interrupt chatbot responses.
    -   Enable the chatbot to process and discuss user-provided links.
-   **Dynamic Content:** Fetch content dynamically from social media and other sources to keep the chatbot's knowledge base current.
-   **Advanced Answering Patterns:** Implement a ReAct (Reasoning and Acting) framework to allow the chatbot to dynamically select tools and retrieve new information during a conversation.
-   **Expanded Context Window:** Increase the context window size and implement context compression techniques.
