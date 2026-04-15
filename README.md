# EIA Energy Research Assistant

**Agentic AI Project — Research Assistant**

Author: Srikaran Anand (fsrikar@okstate.edu)  
Oklahoma State University | Agentic AI Systems  
April 2026

---

## Overview

An autonomous AI research assistant that queries the **U.S. Energy Information Administration (EIA) API v2** to answer energy questions with verified data, auto-generated charts, and full observability.

Built with **Streamlit** for rapid deployment and easy demonstration.

# Access [HERE](https://eia-energy-research-assistant.streamlit.app/)

## Features

- **Research Chat** — Ask energy questions with suggested queries, Plotly charts, evaluation badges, and agent trace viewer
- **Telemetry Dashboard** — KPI cards (Total Events, Avg Latency, Cache Hits, Errors), event distribution charts, live EIA data preview
- **Observability** — Full agent trace stream with colored event type badges and JSON detail viewer
- **Architecture** — Visual 10-step pipeline flow, 6 use cases, technology stack
- **MCP Tools** — Interactive tool execution, guardrails testing, LLM-as-Judge evaluation testing

## Architecture

```
User Query → Guardrails → Intent Detection → RAG Retrieval → MCP Tool Selection
    → EIA API v2 Query → Response Synthesis → LLM-as-Judge Eval → Observability Log → Response
```

### ReAct Pattern (Reasoning + Acting)
1. **THINK:** Apply guardrails (input validation, prompt injection detection, topic relevance)
2. **THINK:** Detect intent (trend, compare, forecast, analyze, explore, explain, latest)
3. **ACT:** Retrieve context from RAG knowledge base (8 energy domain chunks)
4. **THINK:** Select MCP tool (query_eia_data, explore_eia_datasets, get_preset_data)
5. **ACT:** Query EIA API v2 with caching, retry, and rate limiting
6. **OBSERVE:** Format data into markdown tables and chart data
7. **THINK:** Synthesize comprehensive response
8. **EVAL:** LLM-as-Judge 3-dimension scoring (factual accuracy, relevance, completeness)

| Phase | Requirement | Implementation |
|-------|------------|----------------|
| 1 | Architecture & Use Cases | ReAct agentic pattern, 6 documented use cases |
| 2 | Agentic RAG | 8 energy knowledge chunks, keyword retrieval |
| 2 | MCP Protocol | 3 standardized tools with JSON Schema parameters |
| 2 | Flow Engineering | Iterative thought → action → observation loops |
| 3 | Guardrails | Topic validation + 14 prompt injection regex patterns |
| 3 | LLM-as-Judge Eval | 3-dimension scoring (accuracy, relevance, completeness) 1-5 scale |
| 3 | Caching | 1hr TTL SQLite cache |
| 3 | Retry | 3 attempts, exponential backoff (1s, 2s, 4s) |
| 3 | Rate Limiting | 200ms minimum between API requests |
| 3 | Observability | Full agent trace logging with event_type + duration_ms |
| 3 | Latency Tracking | Per-step timing in agent traces + dashboard KPIs |

## Quick Start

### Local
```bash
pip install streamlit requests pandas plotly
streamlit run app.py
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | Streamlit, Plotly, Pandas |
| Backend | Python, SQLite |
| AI / Agent | ReAct Pattern, RAG Pipeline, MCP Protocol, LLM-as-Judge |
| Data Source | EIA API v2 — electricity, natural gas, petroleum, renewables, emissions, STEO |

## File Structure

```
eia-streamlit/
├── app.py              # Streamlit application (5 pages)
├── agent.py            # Agentic AI engine (guardrails, RAG, intent, eval, ReAct)
├── eia_api.py          # EIA API v2 integration (caching, retry, rate limiting)
├── storage.py          # SQLite database layer
├── models.py           # Data model definitions
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── ARCHITECTURE.md     # Mermaid architecture diagrams
```

## MCP Tools

| Tool | Description |
|------|------------|
| `query_eia_data` | Query any EIA API v2 endpoint with route, facets, frequency, date range |
| `explore_eia_datasets` | Browse available EIA API routes and dataset metadata |
| `get_preset_data` | Fetch pre-configured datasets (7 presets for common use cases) |

## Use Cases

1. **Price Analysis** — Query electricity and gas prices by state, sector, period
2. **Trend Exploration** — Visualize historical energy trends with auto-generated charts
3. **Renewable Monitoring** — Track solar, wind, hydro generation growth
4. **Emissions Tracking** — Analyze CO2 trends by sector and fuel type
5. **Dataset Discovery** — Explore available EIA datasets and structure
6. **Energy Forecasting** — Access STEO short-term energy outlook projections

## EIA API Key

The app uses an API key that needs registration. To use your own key:

```bash
export EIA_API_KEY=your_key_here
streamlit run app.py
```

Register for a free key at: https://www.eia.gov/opendata/register.php

## License

Academic use — Oklahoma State University.
