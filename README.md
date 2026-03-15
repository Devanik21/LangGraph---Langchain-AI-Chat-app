# Langgraph Langchain AI Chat APP

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/LangGraph---Langchain-AI-Chat-app?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/LangGraph---Langchain-AI-Chat-app?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Stateful agentic AI with LangGraph — structured multi-step reasoning and tool use in a graph-based workflow.

---

**Topics:** `langchain` · `generative-ai` · `large-language-models` · `deep-learning` · `agentic-ai` · `conversational-ai` · `graph-based-reasoning` · `langgraph` · `llm` · `stateful-agents`

## Overview

This application demonstrates LangGraph in a fully functional, deployable conversational AI context. LangGraph extends LangChain with a directed graph execution model, where each node in the graph is a processing step — an LLM call, a tool invocation, or a conditional router — and edges define the control flow between steps.

The chat interface provides multi-turn conversation with a memory backend that stores the conversation history in a vector store (FAISS or Chroma), enabling semantic retrieval of relevant earlier context rather than simple linear buffering. This is particularly important for long conversations where the full history would exceed the model's context window.

Tool use is a first-class feature: the agent can invoke a Python REPL for computation, a web search tool for current information, a Wikipedia lookup for factual grounding, and a calculator for arithmetic. The tool selection and invocation are handled by the LLM's function-calling interface, with results fed back into the reasoning loop before the final response is generated.

---

## Motivation

LangGraph represents the current frontier of LLM application development: moving beyond single-shot prompt-response patterns toward structured, stateful, multi-step reasoning agents. This project was built to explore and demonstrate these patterns in a real, running application — not just a tutorial notebook.

---

## Architecture

```
User Message
        │
  LangGraph State Machine
  ┌─────────────────────────────┐
  │ Node 1: Intent Classifier   │
  │ Node 2: Tool Router         │
  │ Node 3: Tool Execution      │
  │ Node 4: Response Synthesiser│
  └─────────────────────────────┘
        │
  Vector Memory (FAISS/Chroma)
        │
  Final Response → Streamlit Chat UI
```

---

## Features

### Multi-Turn Conversation with Memory
Conversation history is stored in a vector store with semantic retrieval, allowing the agent to reference relevant earlier context even in long conversations that exceed the model's context window.

### LangGraph State Machine
Structured graph execution: each conversation turn flows through intent classification, optional tool routing, tool execution, and response synthesis nodes.

### Tool-Augmented Reasoning
The agent can invoke web search (SerpAPI / DuckDuckGo), Wikipedia lookup, a Python REPL for computation, and a calculator — selecting tools based on the LLM's assessment of the question.

### Streaming Response Output
LLM response tokens are streamed to the UI character by character, providing low-latency perceived response time even for long outputs.

### Multi-Model Backend Support
Switch between OpenAI GPT-4o, Google Gemini, Anthropic Claude, or a locally running Ollama model via environment variable configuration.

### Conversation Export
Export the full conversation history as a formatted Markdown file or JSON transcript for sharing or review.

### System Prompt Editor
Sidebar text area for customising the agent's system prompt — persona, task focus, language, and response style — without code changes.

### Token Usage Tracking
Real-time token counter in the sidebar tracks prompt and completion tokens per message and cumulative session totals.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **LangChain / LangGraph** | Agent framework | Chains, graphs, memory, tool interfaces |
| **Streamlit** | Chat UI | st.chat_message, st.chat_input for ChatGPT-style interface |
| **FAISS / Chroma** | Vector memory store | Semantic conversation history retrieval |
| **OpenAI / Gemini SDK** | LLM backend | GPT-4o / Gemini function-calling support |
| **SerpAPI / DuckDuckGo** | Web search tool | Real-time web search for current information |
| **python-dotenv** | Env management | API key loading from .env file |

> **Key packages detected in this repo:** `streamlit` · `langchain` · `langchain-google-genai` · `langgraph` · `faiss-cpu` · `unstructured` · `python-docx` · `pypdf` · `pandas`

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JS projects)
- `pip` or `npm` package manager
- Relevant API keys (see Configuration section)

### Installation

```bash
git clone https://github.com/Devanik21/LangGraph---Langchain-AI-Chat-app.git
cd LangGraph---Langchain-AI-Chat-app
python -m venv venv && source venv/bin/activate
pip install langchain langgraph openai google-generativeai faiss-cpu streamlit python-dotenv

# Create .env file
echo 'OPENAI_API_KEY=sk-...' > .env
echo 'SERPAPI_API_KEY=...' >> .env  # optional, for web search

streamlit run app.py
```

---

## Usage

```bash
# Basic chat
streamlit run app.py

# CLI test
python agent_cli.py --query 'What is the capital of France and its current population?'

# Switch model via environment
MODEL=gemini-2.0-flash streamlit run app.py
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `(required)` | OpenAI API key for GPT models |
| `GOOGLE_API_KEY` | `(optional)` | Google API key for Gemini models |
| `MODEL_BACKEND` | `openai` | LLM backend: openai, gemini, ollama |
| `MEMORY_TYPE` | `vector` | Memory backend: buffer, vector, summary |
| `MAX_ITERATIONS` | `10` | Maximum ReAct agent tool-use iterations |

> Copy `.env.example` to `.env` and populate all required values before running.

---

## Project Structure

```
LangGraph---Langchain-AI-Chat-app/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Long-term user memory with persistent vector store (PostgreSQL + pgvector)
- [ ] Multi-agent orchestration: specialist sub-agents for code, research, and creative tasks
- [ ] Document upload and RAG mode: chat over user-provided PDFs
- [ ] Voice I/O integration with Whisper (speech-to-text) and ElevenLabs (text-to-speech)
- [ ] Tracing and observability integration with LangSmith for debugging agent reasoning

---

## Contributing

Contributions, issues, and feature requests are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please follow conventional commit messages and ensure any new code is documented.

---

## Notes

API keys for the selected LLM backend and optional tools (web search) are required. Tool use increases latency and token consumption. The ReAct loop has a configurable maximum iterations limit to prevent runaway tool calls.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Crafted with curiosity, precision, and a belief that good software is worth building well.*
