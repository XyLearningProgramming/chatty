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

### 3.1. Multi-Agent Retrieval and Answering Pipeline

To provide a more dynamic and intelligent response, the system will use a three-agent pipeline. This pipeline separates the concerns of understanding the query's relevance, gathering information, and formulating a final answer.

**Pipeline Flow:**

1.  **Relevancy Check Agent:**
    -   The user's query is first processed by this agent to determine if it is relevant to the chatbot's designated professional persona and knowledge base.
    -   If the query is deemed irrelevant, a predefined polite refusal is returned, and the process stops.

2.  **Retriever Agent (ReAct):**
    -   If the query is relevant, it is passed to a **Retriever Agent** built using a ReAct (Reasoning and Acting) framework.
    -   This agent has access to a set of tools, such as vector store search, web search, or document loaders.
    -   Based on the query, the agent autonomously decides which tools to use and in what sequence to gather the most relevant context. For example, a query about a recent event might trigger a web search, while a query about professional experience would trigger a search of a resume vector store.
    -   The output of this agent is a consolidated block of context.

3.  **Answering Agent:**
    -   The original query and the context retrieved by the Retriever Agent are then passed to an **Answering Agent**.
    -   This agent's responsibility is to synthesize the provided information into a coherent, well-written, and helpful final answer for the user.

**Pseudo-code for the Agent Pipeline:**

```python
# Pseudo-code for the multi-agent pipeline

from langchain.agents import AgentExecutor, create_react_agent
# Assume other necessary components are defined (e.g., llm, tools, prompts)

# 1. Create the Retriever Agent
retriever_agent = create_react_agent(llm, retriever_tools, retriever_prompt)
retriever_executor = AgentExecutor(
    agent=retriever_agent,
    tools=retriever_tools,
    verbose=True,
    max_iterations=8,
    early_stopping_method="generate",
    handle_parsing_errors=True
)

# 2. Define the full pipeline
def run_full_pipeline(user_query: str):
    # Step 1: Check for relevance
    if not relevancy_check_chain.invoke({"query": user_query}):
        return "I can only answer questions about my professional experience."

    # Step 2: Run the retriever agent to get context
    retrieved_context = retriever_executor.invoke({
        "input": user_query
    })

    # Step 3: Run the answering agent with the retrieved context
    final_answer = answerer_chain.invoke({
        "query": user_query,
        "context": retrieved_context["output"]
    })

    return final_answer
```

### 3.2. Customizable Tools for the Retriever Agent

The Retriever Agent's capabilities are extended by a set of customizable tools. Each tool is defined by a name, a description, and a JSON schema for its arguments. This allows for a flexible and extensible system where new tools can be easily added or existing ones modified.

**Tool Definition:**

Each tool is defined with the following properties:
- **Name:** A unique identifier for the tool.
- **Description:** A description of the tool's functionality, which the ReAct agent uses to decide when to use the tool.
- **JSON Schema:** A JSON schema defining the arguments that the tool accepts.

**Example: Web Search Tool**

```json
{
  "name": "web_search",
  "description": "Searches the web for information on a given topic.",
  "json_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query."
      }
    },
    "required": ["query"]
  }
}
```

**Tool Processors and Parsers:**

Each tool can have an associated processor and parser.

- **Processor:** A function that takes the tool's arguments and performs the necessary actions. For example, the `web_search` processor would take the search query, perform a web search, and return the search results.
- **Parser:** A function that takes the output of the processor and formats it into a standardized format that can be used by the agent. This allows for a consistent data structure across different tools.

By using this modular approach, the system can be easily extended with new tools and data sources, making the chatbot more versatile and powerful.

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
            if "
" in token:  # Found newline after ```json
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

To ensure a clean separation of concerns and better scalability, the project will adopt a feature-sliced directory structure.

```
src/
└── chatty/
    ├── __init__.py
    ├── api/
    │   ├── __init__.py
    │   ├── chat.py
    │   └── models.py
    ├── core/
    │   ├── __init__.py
    │   ├── knowledge.py
    │   ├── synthesis.py
    │   ├── memory.py
    │   └── generation.py
    ├── infra/
    │   ├── __init__.py
    │   └── vector_db.py
    └── app.py
```

**Component Responsibilities:**

*   **`app.py`**: Initializes the FastAPI application and includes the feature-level routers from `api`.
*   **`api/chat.py`**: Defines the FastAPI endpoint for chat, handles HTTP requests and responses, and orchestrates calls to the `core` services.
*   **`api/models.py`**: Defines the Pydantic models for the chat API.
*   **`core/knowledge.py`**: Contains the implementation of the Knowledge Agent, including its tools for data retrieval and context gathering.
*   **`core/synthesis.py`**: Contains the implementation of the Synthesis Agent for combining information into coherent responses.
*   **`core/memory.py`**: Implements semantic memory business logic including admission policies, similarity thresholds, frequency tracking, and LFU eviction strategies.
*   **`core/generation.py`**: Handles text generation and token processing state machine for streaming responses.
*   **`infra/vector_db.py`**: Provides an abstraction layer for interacting with the vector database.

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