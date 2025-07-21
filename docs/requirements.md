# Chat With Me: Project Requirements

## 1. Overview

This document outlines the requirements for "Chat With Me," a chatbot designed to impersonate a specific professional persona using a predefined knowledge base. As a proof-of-concept, the MVP will be developed under significant resource constraints, prioritizing core functionality and a clear path for future scaling.

## 2. Core Objective

The primary goal is to create a chatbot that can accurately and professionally represent a user's persona. It will answer questions related to their expertise, experience, and other professional details while politely deflecting queries outside its designated scope.

---

## 3. Minimum Viable Product (MVP) Requirements

### 3.1. Functional Requirements

1.  **Persona-Based Responses:** The chatbot must respond to user queries based on a static, predefined set of personal and professional information.
    *   **Relevance Handling:** The determination of whether a query is relevant will be managed via a sophisticated system prompt. This prompt will be iteratively refined to handle in-scope, out-of-scope, and ambiguous questions effectively.

2.  **Knowledge Base & Semantic Search:** The chatbot will not rely on simple keyword matching. It will be powered by a Retrieval-Augmented Generation (RAG) pipeline.
    *   **Knowledge Ingestion:** A process will be established to ingest, chunk, and convert source documents (e.g., resume, articles) into vector embeddings.
    *   **Vector Storage & Search:** These embeddings will be stored in a vector database to enable fast and accurate semantic search for finding the most relevant context to answer user queries.

3.  **Conversational Context:** The chatbot must maintain the context of a conversation. To manage token limits, it will use a **sliding window** approach, dropping the oldest messages as the conversation grows.

4.  **Structured Data Output:** For certain queries (e.g., requesting a blog post), the chatbot must provide a natural language response followed by a structured JSON object to facilitate frontend rendering. The API response must include a distinct end-of-stream marker.

5.  **Aggressive Caching:** To ensure responsiveness despite minimal model-serving resources, a caching layer will be implemented. The system will cache the generated answers for frequently asked or "classic" questions, bypassing the model entirely for known queries.

### 3.2. Non-Functional Requirements

1.  **Statelessness and Privacy:** The service must be stateless. User login is not required, and no conversation data or personally identifiable information will be stored.

2.  **Scalability:** The architecture must be designed for horizontal scalability, with no dependencies on a single-process or single-instance logic.

3.  **API and Frontend Integration:** The API must be designed with a clear schema to support integration with a separate frontend application.

4.  **Rate Limiting:** A distributed rate limiter will be implemented to manage API usage.

5.  **Configuration Management:** Key parameters—such as the __system prompt__, __persona details__, and __service endpoints__—will be managed in external configuration files to allow for easy modification without code changes.

### 3.3. Quality and Evaluation

1.  **Golden Dataset:** A "golden dataset" containing a representative set of questions and their ideal answers for critial steps in pipeline will be created.
2.  **Regression Testing:** This dataset will be used as a baseline for quality assurance. Automated tests will run against the dataset to evaluate the impact of changes to the prompt, model, or retrieval logic, preventing regressions and objectively measuring performance.

### 3.4. Resources and Constraints

*   **Chatbot Service:**
    *   **CPU:** ~1 core
    *   **Memory:** ~800 MiB
*   **Model Server (Qwen3 0.6B):**
    *   **Endpoint:** `/api/v1/chat/completions`
    *   **CPU:** ~3 cores
    *   **Memory:** ~800 MiB
    *   **Concurrency:** 1
*   **Vector Database (e.g., pgvector):**
    *   **CPU:** ~0.5 core (shared)
    *   **Memory:** ~500 MiB
*   **Backend Infrastructure:**
    *   Access to provisioned Redis and PostgreSQL instances.

---

## 4. Future Development (Post-MVP)

The following features are planned for future releases:

1.  **User Accounts:** Introduce user authentication to enable conversation history and personalized context.
2.  **Enhanced Functionality:**
    *   Allow users to interrupt chatbot responses.
    *   Enable the chatbot to process and discuss user-provided links.
3.  **Dynamic Content:** Fetch content dynamically from social media and other sources to keep the chatbot's knowledge base current.
4.  **Advanced Answering Patterns:** Implement a ReAct (Reasoning and Acting) framework to allow the chatbot to dynamically select tools and retrieve new information during a conversation.
5.  **Expanded Context Window:** Increase the context window size and implement context compression techniques.
