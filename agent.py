"""
Agentic AI Engine - EIA Energy Research Assistant
Author: Srikaran Anand (fsrikar@okstate.edu), Oklahoma State University
Course: Agentic AI Systems - Capstone Project (Option 4: Research Assistant)

Architecture: ReAct Pattern (Reasoning + Acting)
  - Guardrails: Input validation, prompt injection detection, topic relevance
  - RAG: Retrieval Augmented Generation over energy domain knowledge
  - Flow Engineering: Iterative Thought → Action → Observation loops
  - MCP: Standardized tool selection (query_eia_data, explore_eia_datasets, get_preset_data)
  - LLM-as-Judge Eval: 3-dimension scoring (factual accuracy, relevance, completeness)
  - Observability: Full agent trace logging
"""

import re
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import storage
import eia_api

logger = logging.getLogger(__name__)


# ─── Energy Domain Constants ───────────────────────────────────────────────────

ALLOWED_TOPICS = [
    "energy", "electricity", "natural gas", "petroleum", "oil", "coal",
    "renewable", "solar", "wind", "nuclear", "hydro", "hydropower",
    "fossil fuel", "carbon", "co2", "emissions", "climate", "greenhouse",
    "power", "fuel", "barrel", "btu", "kwh", "megawatt", "gigawatt",
    "price", "production", "consumption", "generation", "forecast",
    "eia", "department of energy", "doe", "steo", "crude", "gasoline",
    "diesel", "propane", "lng", "biomass", "geothermal", "offshore",
    "refinery", "pipeline", "grid", "utility", "sector", "retail",
    "wholesale", "capacity", "reserve", "storage", "export", "import",
    "trade", "market", "outlook", "trend", "data", "statistics",
    "report", "monthly", "annual", "weekly", "watt", "joule"
]

BLOCKED_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions?",
    r"you\s+are\s+now\s+a?n?\s+\w+",
    r"pretend\s+to\s+be",
    r"jailbreak",
    r"bypass\s+(safety|security|filter|restriction|guardrail)",
    r"act\s+as\s+(?:a|an|if)",
    r"system\s+prompt",
    r"reveal\s+your\s+(instructions?|prompt|system|training)",
    r"forget\s+(your|previous|all)\s+(instructions?|training|context)",
    r"disregard\s+(previous|all|any)\s+instructions?",
    r"override\s+(your|the)\s+(instructions?|rules?|constraints?)",
    r"<\s*script\s*>",
    r"javascript\s*:",
    r"eval\s*\(",
    r"exec\s*\(",
]

# ─── Energy Knowledge Base for RAG ────────────────────────────────────────────

ENERGY_KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "source": "EIA Overview",
        "content": (
            "The U.S. Energy Information Administration (EIA) is the statistical and analytical agency "
            "within the U.S. Department of Energy. EIA collects, analyzes, and disseminates independent "
            "and impartial energy information to promote sound policymaking, efficient markets, and public "
            "understanding of energy and its interaction with the economy and the environment. EIA's API v2 "
            "provides access to time-series energy data covering electricity, petroleum, natural gas, coal, "
            "renewable energy, nuclear, and international energy statistics. Data is available at various "
            "frequencies: annual, monthly, weekly, and hourly."
        ),
    },
    {
        "source": "Electricity Markets",
        "content": (
            "U.S. electricity is generated from multiple fuel sources: natural gas (~40%), coal (~20%), "
            "nuclear (~18%), wind (~10%), hydro (~6%), solar (~4%), and other renewables. The EIA tracks "
            "electricity retail sales (in billion kWh), average retail prices (in cents/kWh), and revenue "
            "(in million dollars) by sector: residential, commercial, industrial, and transportation. "
            "Electricity prices vary significantly by region and season. The residential average U.S. "
            "electricity price is approximately 12-16 cents per kWh. EIA's electric power operational data "
            "covers generation by fuel type, capacity, and net generation by state."
        ),
    },
    {
        "source": "Natural Gas Markets",
        "content": (
            "Natural gas is the largest source of U.S. electricity generation and a major heating fuel. "
            "EIA tracks natural gas production, storage, imports/exports, and prices. Wellhead prices "
            "are measured in dollars per thousand cubic feet (Mcf) or per million British thermal units "
            "(MMBtu). Henry Hub is the primary U.S. natural gas pricing benchmark. The U.S. has become "
            "the world's largest LNG exporter. EIA's natural gas API routes include pri/sum (price summary), "
            "sum/snd (supply and disposition), and stor/wkly (weekly storage). Prices are highly seasonal, "
            "peaking in winter months due to heating demand."
        ),
    },
    {
        "source": "Petroleum & Oil Markets",
        "content": (
            "Petroleum is the largest energy source in the U.S., used mainly for transportation (gasoline, "
            "diesel, jet fuel). The U.S. is the world's largest crude oil producer, exceeding 13 million "
            "barrels per day. West Texas Intermediate (WTI) is the U.S. crude oil benchmark. EIA tracks "
            "crude oil production (petroleum/crd/crpdn), weekly petroleum supply (petroleum/sum/sndw), "
            "and retail gasoline/diesel prices (petroleum/pri/gnd). OPEC+ production decisions significantly "
            "affect world oil prices. The Strategic Petroleum Reserve (SPR) holds emergency crude oil stocks. "
            "U.S. refinery capacity is approximately 17-18 million barrels per day."
        ),
    },
    {
        "source": "Renewable Energy",
        "content": (
            "Renewable energy is the fastest-growing sector of U.S. electricity generation. Solar photovoltaic "
            "(PV) capacity has grown exponentially due to falling costs, with utility-scale solar now cheaper "
            "than new fossil fuel plants in most regions. Wind power is the largest renewable electricity "
            "source in the U.S. EIA fuel type codes: SUN=Solar, WND=Wind, HYC=Conventional Hydroelectric, "
            "GEO=Geothermal, BIO=Biomass. The Inflation Reduction Act (IRA) of 2022 provides major tax "
            "incentives for clean energy investment. EIA's electric power operational data tracks generation "
            "by fuel type across all sectors. Renewable portfolio standards (RPS) in many states mandate "
            "increasing renewable energy percentages."
        ),
    },
    {
        "source": "CO2 Emissions",
        "content": (
            "U.S. energy-related CO2 emissions peaked around 6,000 million metric tons in 2005 and have "
            "declined significantly due to the natural gas boom (displacing coal) and renewable energy growth. "
            "The power sector has seen the largest reductions. EIA tracks CO2 emissions by state, sector, "
            "and fuel type through the co2-emissions API route. Sectors include: residential, commercial, "
            "industrial, transportation, and electric power. Transportation is now the largest CO2-emitting "
            "sector in the U.S. The U.S. has committed to a 50-52% reduction from 2005 levels by 2030 "
            "under the Paris Agreement. Carbon intensity (CO2 per unit of energy) has been steadily declining."
        ),
    },
    {
        "source": "Short-Term Energy Outlook",
        "content": (
            "EIA's Short-Term Energy Outlook (STEO) is a monthly report providing forecasts for energy "
            "markets over the next 24 months. It covers world liquid fuels supply and demand, U.S. energy "
            "price forecasts (crude oil, natural gas, electricity, gasoline), U.S. electricity generation "
            "by fuel, and U.S. energy production and consumption. Key STEO series include: PATC_WORLD "
            "(world petroleum prices), MGASUSGP (U.S. gasoline price), COPRPUS (U.S. crude oil production). "
            "The STEO is EIA's flagship publication for near-term energy market forecasting and is widely "
            "used by energy market participants, policymakers, and analysts."
        ),
    },
    {
        "source": "EIA API Datasets",
        "content": (
            "EIA API v2 major dataset routes: electricity/ (generation, retail sales, operational data), "
            "natural-gas/ (prices, supply, storage, imports/exports), petroleum/ (production, prices, "
            "supply, refinery), coal/ (production, consumption, stocks), nuclear/ (capacity, generation), "
            "renewable-capacity/ (installed capacity by fuel type), international/ (world energy data), "
            "co2-emissions/ (CO2 by state/sector/fuel), steo/ (short-term forecasts), aeo/ (annual energy "
            "outlook). Each route supports filtering by facets (e.g., state, sector, fuel type), "
            "frequency (annual/monthly/weekly/hourly), date range, and pagination. The API requires an "
            "API key (free registration). DEMO_KEY allows limited requests per hour."
        ),
    },
]

# ─── Intent Detection Keywords ─────────────────────────────────────────────────

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "trend": ["trend", "over time", "history", "historical", "change", "growth", "decline", "trajectory"],
    "compare": ["compare", "vs", "versus", "difference", "between", "which is", "better", "higher", "lower"],
    "forecast": ["forecast", "predict", "outlook", "future", "projection", "expect", "steo", "next year"],
    "latest": ["latest", "current", "recent", "now", "today", "this month", "this year", "newest"],
    "explain": ["what is", "what are", "explain", "how does", "why", "define", "meaning", "about"],
    "explore": ["explore", "browse", "what data", "available", "dataset", "routes", "categories"],
    "analyze": ["analyze", "analysis", "breakdown", "deep dive", "insight", "pattern", "correlation"],
}

# Preset selection mapping: (data_type, intent) → preset_name
PRESET_SELECTION_MAP: Dict[Tuple[str, str], str] = {
    ("electricity", "latest"): "electricityRetailSales",
    ("electricity", "trend"): "electricityRetailSales",
    ("electricity", "analyze"): "electricityRetailSales",
    ("electricity", "compare"): "electricityRetailSales",
    ("natural gas", "latest"): "naturalGasPrices",
    ("natural gas", "trend"): "naturalGasPrices",
    ("natural gas", "analyze"): "naturalGasPrices",
    ("petroleum", "latest"): "petroleumPrices",
    ("petroleum", "trend"): "petroleumPrices",
    ("petroleum", "analyze"): "petroleumPrices",
    ("gasoline", "latest"): "petroleumPrices",
    ("gasoline", "trend"): "petroleumPrices",
    ("oil", "latest"): "crudeOilProduction",
    ("oil", "trend"): "crudeOilProduction",
    ("oil", "analyze"): "crudeOilProduction",
    ("crude", "latest"): "crudeOilProduction",
    ("crude", "trend"): "crudeOilProduction",
    ("renewable", "latest"): "renewableGeneration",
    ("renewable", "trend"): "renewableGeneration",
    ("renewable", "analyze"): "renewableGeneration",
    ("renewable", "compare"): "renewableGeneration",
    ("solar", "latest"): "renewableGeneration",
    ("solar", "trend"): "renewableGeneration",
    ("wind", "latest"): "renewableGeneration",
    ("wind", "trend"): "renewableGeneration",
    ("co2", "latest"): "co2Emissions",
    ("co2", "trend"): "co2Emissions",
    ("co2", "analyze"): "co2Emissions",
    ("emissions", "latest"): "co2Emissions",
    ("emissions", "trend"): "co2Emissions",
    ("emissions", "analyze"): "co2Emissions",
    ("carbon", "latest"): "co2Emissions",
    ("carbon", "trend"): "co2Emissions",
    ("forecast", "latest"): "steo",
    ("forecast", "forecast"): "steo",
    ("steo", "latest"): "steo",
    ("outlook", "latest"): "steo",
    ("outlook", "forecast"): "steo",
}


# ─── Guardrails ────────────────────────────────────────────────────────────────

def apply_guardrails(user_input: str) -> Dict[str, Any]:
    """
    Validate and sanitize user input.
    
    Checks:
    1. Empty/too short input
    2. Input too long (>2000 chars)
    3. Prompt injection patterns (regex-based)
    4. Topic relevance (energy domain keywords)
    
    Returns: {passed: bool, reason: str, sanitized_input: str}
    """
    # Sanitize: strip and normalize whitespace
    sanitized = re.sub(r"\s+", " ", user_input.strip())

    # Empty check
    if not sanitized:
        return {"passed": False, "reason": "Input is empty.", "sanitized_input": sanitized}

    # Length check
    if len(sanitized) < 3:
        return {"passed": False, "reason": "Input too short. Please ask a more detailed question.", "sanitized_input": sanitized}

    if len(sanitized) > 2000:
        return {"passed": False, "reason": "Input exceeds 2000 character limit.", "sanitized_input": sanitized[:2000]}

    # Prompt injection detection
    lower_input = sanitized.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lower_input, re.IGNORECASE):
            return {
                "passed": False,
                "reason": "Input contains disallowed content (potential prompt injection attempt).",
                "sanitized_input": sanitized,
            }

    # Topic relevance check
    has_energy_topic = any(topic in lower_input for topic in ALLOWED_TOPICS)
    
    # Also allow general info/data requests that may not contain energy keywords
    is_general_request = any(w in lower_input for w in [
        "what", "how", "show", "tell", "give", "get", "find", "search",
        "list", "help", "can you", "could you", "please"
    ])

    if not has_energy_topic and not is_general_request:
        return {
            "passed": False,
            "reason": (
                "Your question doesn't appear to be related to energy topics. "
                "This assistant specializes in U.S. energy data from the EIA "
                "(electricity, natural gas, petroleum, renewables, emissions, etc.)."
            ),
            "sanitized_input": sanitized,
        }

    return {"passed": True, "reason": "Input passed all guardrail checks.", "sanitized_input": sanitized}


# ─── RAG ──────────────────────────────────────────────────────────────────────

def initialize_rag() -> None:
    """Seed the RAG knowledge base if it's empty."""
    if storage.count_rag_chunks() == 0:
        logger.info("Seeding RAG knowledge base with energy domain knowledge...")
        for chunk in ENERGY_KNOWLEDGE_BASE:
            storage.create_rag_chunk(
                source=chunk["source"],
                content=chunk["content"],
                metadata={"domain": "energy", "source_type": "knowledge_base"},
            )
        logger.info(f"Seeded {len(ENERGY_KNOWLEDGE_BASE)} RAG chunks.")


def retrieve_context(query: str) -> List[Dict[str, Any]]:
    """
    Retrieve relevant knowledge chunks using keyword-based search.
    Returns top 5 most relevant chunks.
    """
    return storage.search_rag_chunks(query, top_k=5)


# ─── Intent Detection ──────────────────────────────────────────────────────────

def detect_intent(query: str) -> Dict[str, Any]:
    """
    Detect user intent and extract energy entities from query.
    
    Intents: analyze, trend, compare, forecast, latest, explain, explore
    """
    lower = query.lower()
    
    # Score each intent
    intent_scores: Dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > 0:
            intent_scores[intent] = score

    # Default intent
    detected_intent = max(intent_scores, key=intent_scores.get) if intent_scores else "analyze"

    # Extract energy entities
    entities: Dict[str, Any] = {}
    
    # Data type detection
    data_type_map = {
        "electricity": ["electricity", "electric", "power", "kwh", "megawatt", "gigawatt", "grid"],
        "natural gas": ["natural gas", "gas price", "gas market", "henry hub", "lng"],
        "petroleum": ["petroleum", "gasoline", "diesel", "fuel oil", "jet fuel"],
        "oil": ["crude oil", "crude", "oil production", "barrel", "bbl", "wti", "brent"],
        "renewable": ["renewable", "solar", "wind", "hydro", "hydropower", "clean energy"],
        "solar": ["solar", "photovoltaic", "pv", "sun"],
        "wind": ["wind energy", "wind power", "turbine"],
        "co2": ["co2", "carbon dioxide", "carbon emissions", "greenhouse gas"],
        "emissions": ["emissions", "carbon footprint", "pollution"],
        "coal": ["coal", "lignite", "anthracite"],
        "nuclear": ["nuclear", "uranium", "reactor", "fission"],
        "forecast": ["forecast", "steo", "outlook", "prediction"],
        "gasoline": ["gasoline", "gas prices", "pump price"],
    }

    for dtype, keywords in data_type_map.items():
        if any(kw in lower for kw in keywords):
            entities["data_type"] = dtype
            break

    # Geography
    us_states = [
        "texas", "california", "florida", "new york", "pennsylvania", "ohio",
        "illinois", "georgia", "north carolina", "michigan"
    ]
    for state in us_states:
        if state in lower:
            entities["state"] = state
            break

    # Time periods
    if "this year" in lower or "2024" in lower or "2025" in lower:
        entities["period"] = "recent"
    elif "last year" in lower or "2023" in lower:
        entities["period"] = "2023"
    elif "5 year" in lower or "five year" in lower:
        entities["period"] = "5_year"
    elif "10 year" in lower or "ten year" in lower:
        entities["period"] = "10_year"

    return {
        "intent": detected_intent,
        "entities": entities,
        "confidence": max(intent_scores.values()) / 3.0 if intent_scores else 0.5,
    }


# ─── Preset Selection ──────────────────────────────────────────────────────────

def select_preset(data_type: str, intent: str) -> Optional[str]:
    """Select the best preset query based on detected data type and intent."""
    # Direct lookup
    key = (data_type.lower(), intent.lower())
    if key in PRESET_SELECTION_MAP:
        return PRESET_SELECTION_MAP[key]
    
    # Partial data_type match
    for (dt, it), preset in PRESET_SELECTION_MAP.items():
        if dt in data_type.lower() or data_type.lower() in dt:
            return preset
    
    return None


# ─── Data Formatting ──────────────────────────────────────────────────────────

def format_data_for_response(data: Any, route: str = "") -> Dict[str, Any]:
    """
    Format EIA API response data into a markdown table and chart-ready format.
    
    Returns: {summary, chart_data, chart_type}
    """
    if not data or "error" in str(data).lower():
        return {"summary": "No data available.", "chart_data": None, "chart_type": None}

    # Navigate to the response data array
    records = []
    if isinstance(data, dict):
        if "data" in data and isinstance(data, dict):
            # Direct preset response with "data" key containing API response
            api_resp = data.get("data", {})
            if isinstance(api_resp, dict):
                records = api_resp.get("response", {}).get("data", [])
        if not records:
            records = data.get("response", {}).get("data", [])

    if not records:
        return {"summary": "No data records found.", "chart_data": None, "chart_type": None}

    # Limit to 20 records for display
    display_records = records[:20]

    # Build markdown table
    if display_records:
        headers = list(display_records[0].keys())
        # Prioritize common columns
        priority = ["period", "stateid", "sectorid", "fueltypeid", "value", "price", "sales", "revenue", "generation"]
        ordered = [h for h in priority if h in headers] + [h for h in headers if h not in priority]
        ordered = ordered[:6]  # Max 6 columns

        table_lines = ["| " + " | ".join(ordered) + " |"]
        table_lines.append("| " + " | ".join(["---"] * len(ordered)) + " |")
        for row in display_records[:15]:
            values = []
            for col in ordered:
                val = row.get(col, "")
                if isinstance(val, float):
                    val = f"{val:.2f}"
                values.append(str(val) if val is not None else "")
            table_lines.append("| " + " | ".join(values) + " |")

        summary = f"**Latest {min(len(display_records), 15)} records:**\n\n" + "\n".join(table_lines)
        summary += f"\n\n*Total records available: {len(records)}*"
    else:
        summary = "No tabular data available."

    # Build chart data
    chart_data = None
    chart_type = "line"

    # Try to extract period and value columns for charting
    period_col = next((k for k in ["period", "date"] if k in (records[0] if records else {})), None)
    value_col = next((k for k in ["value", "price", "sales", "generation", "revenue"] if k in (records[0] if records else {})), None)

    if period_col and value_col and records:
        # Aggregate by period — many datasets return multiple rows per period
        # (e.g. one row per state/sector). Average the numeric values per period.
        from collections import defaultdict
        period_agg: dict = defaultdict(list)
        for r in records:
            period = str(r.get(period_col, ""))
            try:
                v = float(r.get(value_col, 0) or 0)
                if v != 0:  # skip zero/null entries
                    period_agg[period].append(v)
            except (ValueError, TypeError):
                pass

        # Sort periods chronologically and compute averages
        sorted_periods = sorted(period_agg.keys())
        labels = sorted_periods
        values = [round(sum(period_agg[p]) / len(period_agg[p]), 2) if period_agg[p] else 0 for p in sorted_periods]

        # Skip chart if only 1 data point (would just be a dot)
        if len(labels) < 2:
            labels = []
            values = []

        # Determine chart type
        if "co2" in route.lower() or "emission" in route.lower():
            chart_type = "bar"
        elif "renewable" in route.lower() or "fueltypeid" in str(records[0]):
            chart_type = "bar"
        else:
            chart_type = "line"

        if labels and values:
            chart_data = {
                "labels": labels,
                "datasets": [
                    {
                        "label": value_col.replace("_", " ").title(),
                        "data": values,
                        "borderColor": "#0F766E",
                        "backgroundColor": "rgba(15, 118, 110, 0.1)",
                        "tension": 0.4,
                        "fill": True,
                    }
                ],
                "route": route,
                "record_count": len(records),
            }

    return {"summary": summary, "chart_data": chart_data, "chart_type": chart_type}


# ─── LLM-as-Judge Evaluation ──────────────────────────────────────────────────

def evaluate_response(query: str, response: str, data_used: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Rule-based LLM-as-Judge evaluation with 3 dimensions (1-5 scale each):
    1. Factual Accuracy: Does the response use real EIA data correctly?
    2. Relevance: Does the response address the query?
    3. Completeness: Does the response thoroughly cover the topic?
    
    Overall score = average of three dimensions.
    """
    query_lower = query.lower()
    response_lower = response.lower()

    # ── Factual Accuracy (1-5) ─────────────────────────────────────────────────
    factual_accuracy = 3.0  # baseline

    # Reward: uses specific data, numbers, percentages
    if re.search(r"\d+\.?\d*\s*(cents?|dollars?|%|kwh|mmcf|bbl|mmt|bcf|mw|gw)", response_lower):
        factual_accuracy += 1.0
    # Reward: cites EIA as source
    if "eia" in response_lower or "energy information" in response_lower:
        factual_accuracy += 0.5
    # Reward: uses data from API response
    if data_used and not data_used.get("error"):
        factual_accuracy += 0.5
    # Penalty: vague hedging without data
    if response_lower.count("approximately") + response_lower.count("around") + response_lower.count("roughly") > 3:
        factual_accuracy -= 0.5
    # Penalty: explicitly no data
    if "no data available" in response_lower or "unable to retrieve" in response_lower:
        factual_accuracy -= 1.0

    factual_accuracy = max(1.0, min(5.0, factual_accuracy))

    # ── Relevance (1-5) ────────────────────────────────────────────────────────
    relevance = 3.0  # baseline

    # Extract key topics from query
    query_topics = [w for w in query_lower.split() if len(w) > 4]
    matched = sum(1 for t in query_topics if t in response_lower)
    relevance += min(1.5, matched * 0.3)

    # Reward: direct answer to question type
    if "what" in query_lower and ("is" in response_lower or "are" in response_lower):
        relevance += 0.3
    if "how" in query_lower and ("by" in response_lower or "through" in response_lower or "due to" in response_lower):
        relevance += 0.3
    if any(w in query_lower for w in ["trend", "over time", "history"]) and (
        "trend" in response_lower or "increased" in response_lower or "decreased" in response_lower
    ):
        relevance += 0.3

    relevance = max(1.0, min(5.0, relevance))

    # ── Completeness (1-5) ────────────────────────────────────────────────────
    completeness = 3.0  # baseline

    # Reward length and depth
    word_count = len(response.split())
    if word_count > 200:
        completeness += 0.5
    if word_count > 400:
        completeness += 0.5
    # Reward: has table
    if "|" in response and "---" in response:
        completeness += 0.5
    # Reward: has context/explanation
    if any(w in response_lower for w in ["because", "due to", "as a result", "this means"]):
        completeness += 0.3
    # Reward: covers data timeframe
    if re.search(r"20\d\d", response):
        completeness += 0.3
    # Penalty: very short response
    if word_count < 50:
        completeness -= 1.0
    elif word_count < 100:
        completeness -= 0.5

    completeness = max(1.0, min(5.0, completeness))

    # Overall score
    overall = round((factual_accuracy + relevance + completeness) / 3, 2)

    # Build reasoning
    score_label = (
        "Excellent" if overall >= 4.5 else
        "Good" if overall >= 3.5 else
        "Adequate" if overall >= 2.5 else
        "Poor"
    )

    reasoning = (
        f"{score_label} response (overall {overall}/5). "
        f"Factual accuracy: {factual_accuracy}/5 — "
        f"{'Good use of EIA data with specific figures' if factual_accuracy >= 4 else 'Could include more precise data'}. "
        f"Relevance: {relevance}/5 — "
        f"{'Response directly addresses the query' if relevance >= 4 else 'Response could be more focused on the query'}. "
        f"Completeness: {completeness}/5 — "
        f"{'Thorough coverage with context' if completeness >= 4 else 'Could provide more depth and context'}."
    )

    return {
        "score": overall,
        "reasoning": reasoning,
        "factual_accuracy": factual_accuracy,
        "relevance": relevance,
        "completeness": completeness,
    }


# ─── Response Synthesis ───────────────────────────────────────────────────────

def synthesize_response(
    query: str,
    intent: Dict[str, Any],
    context: List[Dict[str, Any]],
    data: Optional[Dict[str, Any]],
    formatted: Optional[Dict[str, Any]],
    steps: List[Dict[str, Any]],
) -> str:
    """
    Synthesize a comprehensive response from all gathered information.
    Uses RAG context + EIA data to produce a helpful, accurate answer.
    """
    # Build context text from RAG
    context_text = ""
    if context:
        context_text = "\n\n".join([
            f"**{c['source']}:** {c['content']}"
            for c in context[:3]
        ])

    intent_str = intent.get("intent", "analyze")
    entities = intent.get("entities", {})
    data_type = entities.get("data_type", "energy")

    # Build response sections
    response_parts = []

    # Introduction based on intent
    intro_map = {
        "trend": f"Here's the trend analysis for {data_type} data from the U.S. EIA:",
        "compare": f"Here's a comparison of {data_type} data from the U.S. EIA:",
        "forecast": "Here's the latest energy outlook and forecast data from EIA's Short-Term Energy Outlook (STEO):",
        "latest": f"Here are the latest {data_type} figures from the U.S. EIA:",
        "explain": f"Here's what you need to know about {data_type}:",
        "explore": "Here's an overview of available EIA energy datasets:",
        "analyze": f"Here's an analysis of {data_type} data from the U.S. EIA:",
    }
    response_parts.append(intro_map.get(intent_str, f"Here's information about {data_type} from the U.S. EIA:"))

    # Add data summary if available
    if formatted and formatted.get("summary") and "No data" not in formatted["summary"]:
        response_parts.append("\n\n" + formatted["summary"])
    elif data and isinstance(data, dict):
        # Try to extract key info from data
        if "label" in data:
            response_parts.append(f"\n\n**Dataset:** {data.get('label', '')} — {data.get('description', '')}")

    # Add contextual knowledge from RAG (use full content, no truncation)
    if context:
        response_parts.append("\n\n### Energy Context\n")
        for c in context[:2]:
            response_parts.append(f"\n**{c['source']}:** {c['content']}\n")

    # Add intent-specific insights
    if intent_str == "trend":
        response_parts.append(
            "\n\n### Trend Insights\nThe data above shows the historical trajectory. "
            "Look for seasonal patterns, year-over-year changes, and long-term trends. "
            "EIA data is updated regularly to reflect the latest energy market conditions."
        )
    elif intent_str == "forecast":
        response_parts.append(
            "\n\n### Forecast Notes\nEIA's Short-Term Energy Outlook (STEO) provides 24-month forward projections. "
            "These forecasts are updated monthly and reflect current market conditions, policy changes, "
            "and seasonal demand patterns. Forecast uncertainty increases further into the future."
        )
    elif intent_str == "compare":
        response_parts.append(
            "\n\n### Comparison Notes\nWhen comparing energy sources, consider: "
            "(1) Generation capacity vs. actual generation, "
            "(2) Levelized cost of energy (LCOE), "
            "(3) Capacity factors (e.g., solar ~25%, wind ~35%, nuclear ~92%), "
            "(4) Regional availability and grid integration costs."
        )

    # Add data source attribution
    response_parts.append(
        "\n\n---\n*Data source: U.S. Energy Information Administration (EIA) API v2 — [eia.gov](https://www.eia.gov)*"
    )

    return "".join(response_parts)


# ─── ReAct Agent Loop ─────────────────────────────────────────────────────────

def run_agent(query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Main ReAct (Reasoning + Acting) agent loop.
    
    Flow:
    1. THINK: Guardrails check
    2. THINK: Intent detection
    3. ACT: RAG retrieval
    4. THINK: Select MCP tool / preset
    5. ACT: Query EIA API via MCP
    6. OBSERVE: Format data
    7. THINK: Synthesize response
    8. EVAL: LLM-as-Judge evaluation
    9. LOG: Observability trace
    
    Returns: {answer, steps, evaluation, data_used, chart_data, chart_type}
    """
    start_time = time.time()
    steps: List[Dict[str, Any]] = []

    def add_step(thought: str = "", action: str = "", observation: str = "", step_type: str = "thought"):
        steps.append({
            "type": step_type,
            "thought": thought,
            "action": action,
            "observation": observation,
            "timestamp": time.time() - start_time,
        })

    # ── Step 1: THINK — Guardrails ─────────────────────────────────────────────
    t0 = time.time()
    guardrail_result = apply_guardrails(query)
    guard_ms = (time.time() - t0) * 1000

    storage.create_log(
        "guardrail",
        {
            "passed": guardrail_result["passed"],
            "reason": guardrail_result["reason"],
            "query_length": len(query),
        },
        conversation_id=conversation_id,
        duration_ms=guard_ms,
    )

    add_step(
        thought=f"Applying guardrails to user input ({len(query)} chars)",
        observation=f"Guardrail result: {'PASSED' if guardrail_result['passed'] else 'BLOCKED'} — {guardrail_result['reason']}",
        step_type="guardrail",
    )

    if not guardrail_result["passed"]:
        answer = f"I'm unable to process that request. {guardrail_result['reason']}\n\nI specialize in U.S. energy data from the EIA — please ask about electricity, natural gas, petroleum, renewables, CO2 emissions, or energy forecasts."
        return {
            "answer": answer,
            "steps": steps,
            "evaluation": {"score": 0, "reasoning": "Blocked by guardrails", "factual_accuracy": 0, "relevance": 0, "completeness": 0},
            "data_used": None,
            "chart_data": None,
            "chart_type": None,
        }

    sanitized_query = guardrail_result["sanitized_input"]

    # ── Step 2: THINK — Intent Detection ──────────────────────────────────────
    t0 = time.time()
    intent = detect_intent(sanitized_query)
    intent_ms = (time.time() - t0) * 1000

    storage.create_log(
        "intent_detection",
        {"intent": intent["intent"], "entities": intent["entities"], "confidence": intent["confidence"]},
        conversation_id=conversation_id,
        duration_ms=intent_ms,
    )

    add_step(
        thought=f"Detecting intent from query: '{sanitized_query[:100]}...' " if len(sanitized_query) > 100 else f"Detecting intent from query: '{sanitized_query}'",
        observation=f"Detected intent: {intent['intent']} | Entities: {json.dumps(intent['entities'])} | Confidence: {intent['confidence']:.2f}",
        step_type="intent",
    )

    # ── Step 3: ACT — RAG Retrieval ───────────────────────────────────────────
    t0 = time.time()
    context_chunks = retrieve_context(sanitized_query)
    rag_ms = (time.time() - t0) * 1000

    storage.create_log(
        "rag_retrieve",
        {"query": sanitized_query[:100], "chunks_retrieved": len(context_chunks), "sources": [c["source"] for c in context_chunks]},
        conversation_id=conversation_id,
        duration_ms=rag_ms,
    )

    add_step(
        thought=f"Retrieving relevant knowledge from RAG knowledge base",
        action="search_rag_chunks",
        observation=f"Retrieved {len(context_chunks)} relevant chunks: {[c['source'] for c in context_chunks]}",
        step_type="rag",
    )

    # ── Step 4: THINK — MCP Tool Selection ────────────────────────────────────
    data_type = intent["entities"].get("data_type", "")
    intent_str = intent["intent"]
    
    # Select preset or decide to explore
    preset_name = None
    use_explore = False

    if intent_str == "explore" or "explore" in sanitized_query.lower() or "available" in sanitized_query.lower():
        use_explore = True
    else:
        preset_name = select_preset(data_type, intent_str)
        # Fallback: try each data_type keyword
        if not preset_name:
            for kw in ["electricity", "natural gas", "petroleum", "oil", "renewable", "co2", "emissions", "forecast"]:
                if kw in sanitized_query.lower():
                    preset_name = select_preset(kw, intent_str)
                    if not preset_name:
                        preset_name = select_preset(kw, "latest")
                    if preset_name:
                        break

    tool_selected = "explore_eia_datasets" if use_explore else ("get_preset_data" if preset_name else "query_eia_data")

    add_step(
        thought=f"Selecting MCP tool based on intent='{intent_str}', data_type='{data_type}'",
        action=f"MCP Tool: {tool_selected}" + (f" | Preset: {preset_name}" if preset_name else ""),
        observation=f"Tool selected: {tool_selected}",
        step_type="tool_selection",
    )

    storage.create_log(
        "tool_selection",
        {"tool": tool_selected, "preset": preset_name, "intent": intent_str, "data_type": data_type},
        conversation_id=conversation_id,
    )

    # ── Step 5: ACT — Execute MCP Tool / Query EIA ────────────────────────────
    t0 = time.time()
    api_data = None
    formatted = None

    try:
        if use_explore:
            route = ""
            if "electricity" in sanitized_query.lower():
                route = "electricity"
            elif "petroleum" in sanitized_query.lower() or "oil" in sanitized_query.lower():
                route = "petroleum"
            elif "gas" in sanitized_query.lower():
                route = "natural-gas"

            add_step(
                thought=f"Exploring EIA API datasets at route: '{route or 'root'}'",
                action=f"execute_mcp_tool('explore_eia_datasets', {{route: '{route}'}})",
                step_type="action",
            )

            api_data = eia_api.execute_mcp_tool(
                "explore_eia_datasets",
                {"route": route},
                conversation_id=conversation_id,
            )

            # Format explore result
            routes_info = []
            if isinstance(api_data, dict):
                resp = api_data.get("response", {})
                routes_list = resp.get("routes", [])
                if routes_list:
                    routes_info = [f"- **{r.get('id', '')}**: {r.get('name', '')} — {r.get('description', '')[:100]}" for r in routes_list[:15]]

            if routes_info:
                formatted = {
                    "summary": "### Available EIA API Routes\n\n" + "\n".join(routes_info),
                    "chart_data": None,
                    "chart_type": None,
                }
            else:
                formatted = {"summary": "Explored EIA datasets. See context for available data.", "chart_data": None, "chart_type": None}

            obs = f"Explored EIA API route='{route}', found {len(routes_info)} routes"

        elif preset_name:
            add_step(
                thought=f"Fetching preset dataset: {preset_name}",
                action=f"execute_mcp_tool('get_preset_data', {{preset_name: '{preset_name}'}})",
                step_type="action",
            )

            api_data = eia_api.execute_mcp_tool(
                "get_preset_data",
                {"preset_name": preset_name},
                conversation_id=conversation_id,
            )

            route = eia_api.PRESET_QUERIES.get(preset_name, {}).get("route", "")
            formatted = format_data_for_response(api_data, route)
            records_count = 0
            if isinstance(api_data, dict):
                inner = api_data.get("data", {})
                if isinstance(inner, dict):
                    records_count = len(inner.get("response", {}).get("data", []))

            obs = f"Retrieved preset '{preset_name}': {records_count} records"

        else:
            # Generic query based on query text
            add_step(
                thought="No specific preset matched; using generic electricity retail sales query",
                action="execute_mcp_tool('query_eia_data', {route: 'electricity/retail-sales', ...})",
                step_type="action",
            )

            api_data = eia_api.execute_mcp_tool(
                "query_eia_data",
                {
                    "route": "electricity/retail-sales",
                    "data": ["price", "sales"],
                    "frequency": "monthly",
                    "length": 24,
                },
                conversation_id=conversation_id,
            )

            formatted = format_data_for_response(api_data, "electricity/retail-sales")
            obs = "Retrieved generic electricity retail sales data"

        api_ms = (time.time() - t0) * 1000
        steps[-1]["observation"] = obs
        steps[-1]["duration_ms"] = api_ms

        storage.create_log(
            "api_result",
            {"preset": preset_name, "has_data": bool(formatted and formatted.get("chart_data")), "explore": use_explore},
            conversation_id=conversation_id,
            duration_ms=api_ms,
        )

    except Exception as e:
        api_ms = (time.time() - t0) * 1000
        storage.create_log("error", {"stage": "api_call", "error": str(e)}, conversation_id=conversation_id, duration_ms=api_ms)
        add_step(
            thought="API call encountered an error",
            observation=f"Error: {str(e)[:200]}",
            step_type="error",
        )
        formatted = {"summary": f"Note: Could not retrieve live data. {str(e)[:100]}", "chart_data": None, "chart_type": None}

    # ── Step 6: THINK — Synthesize Response ───────────────────────────────────
    t0 = time.time()
    add_step(
        thought="Synthesizing comprehensive response from RAG context and EIA data",
        action="synthesize_response",
        step_type="synthesis",
    )

    answer = synthesize_response(
        query=sanitized_query,
        intent=intent,
        context=context_chunks,
        data=api_data,
        formatted=formatted,
        steps=steps,
    )

    synth_ms = (time.time() - t0) * 1000
    steps[-1]["observation"] = f"Response synthesized: {len(answer)} characters"

    storage.create_log(
        "synthesis",
        {"response_length": len(answer), "context_chunks": len(context_chunks)},
        conversation_id=conversation_id,
        duration_ms=synth_ms,
    )

    # ── Step 7: EVAL — LLM-as-Judge ───────────────────────────────────────────
    t0 = time.time()
    evaluation = evaluate_response(sanitized_query, answer, api_data)
    eval_ms = (time.time() - t0) * 1000

    storage.create_log(
        "eval",
        {
            "score": evaluation["score"],
            "factual_accuracy": evaluation["factual_accuracy"],
            "relevance": evaluation["relevance"],
            "completeness": evaluation["completeness"],
        },
        conversation_id=conversation_id,
        duration_ms=eval_ms,
    )

    add_step(
        thought="Running LLM-as-Judge evaluation on response quality",
        action="evaluate_response(factual_accuracy, relevance, completeness)",
        observation=f"Evaluation complete: Score={evaluation['score']}/5 | Accuracy={evaluation['factual_accuracy']} | Relevance={evaluation['relevance']} | Completeness={evaluation['completeness']}",
        step_type="eval",
    )

    # ── Final Log ──────────────────────────────────────────────────────────────
    total_ms = (time.time() - start_time) * 1000
    storage.create_log(
        "agent_complete",
        {
            "query": sanitized_query[:100],
            "intent": intent["intent"],
            "steps": len(steps),
            "eval_score": evaluation["score"],
            "has_chart": bool(formatted and formatted.get("chart_data")),
        },
        conversation_id=conversation_id,
        duration_ms=total_ms,
    )

    return {
        "answer": answer,
        "steps": steps,
        "evaluation": evaluation,
        "data_used": api_data,
        "chart_data": formatted.get("chart_data") if formatted else None,
        "chart_type": formatted.get("chart_type") if formatted else None,
    }
