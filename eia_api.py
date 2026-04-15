"""
EIA API v2 Integration with caching, retry, and rate limiting
Author: Srikaran Anand (fsrikar@okstate.edu), Oklahoma State University
Course: Agentic AI Systems - Capstone Project (Option 4: Research Assistant)

Features:
- Rate limiting: 200ms minimum between requests
- Retry: 3 attempts, exponential backoff [1s, 2s, 4s]
- Cache: SQLite-backed 1hr TTL cache
- MCP Tool definitions (Model Context Protocol)
"""

import os
import time
import json
import hashlib
import logging
import requests
from typing import Any, Dict, List, Optional
from datetime import datetime

import storage

logger = logging.getLogger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2"
EIA_API_KEY = os.environ.get("EIA_API_KEY", "DEMO_KEY")
CACHE_TTL = 3600          # 1 hour in seconds
RATE_LIMIT_MS = 200       # 200ms minimum between requests
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff in seconds
MAX_RETRIES = 3

# Track last request time for rate limiting
_last_request_time: float = 0.0


# ─── Preset Queries ───────────────────────────────────────────────────────────

PRESET_QUERIES: Dict[str, Dict[str, Any]] = {
    "electricityRetailSales": {
        "route": "electricity/retail-sales",
        "label": "Electricity Retail Sales & Prices",
        "description": "Monthly electricity retail sales, revenue, and prices by sector",
        "params": {
            "data[]": ["price", "revenue", "sales"],
            "frequency": "monthly",
            "length": 60,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
        },
    },
    "naturalGasPrices": {
        "route": "natural-gas/pri/sum",
        "label": "Natural Gas Price Summary",
        "description": "Monthly natural gas wellhead and citygate prices",
        "params": {
            "data[]": ["value"],
            "frequency": "monthly",
            "length": 60,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
        },
    },
    "petroleumPrices": {
        "route": "petroleum/pri/gnd",
        "label": "Petroleum & Gasoline Prices",
        "description": "Weekly U.S. gasoline and diesel retail prices",
        "params": {
            "data[]": ["value"],
            "frequency": "weekly",
            "facets[product][]": ["EPM0F"],
            "length": 52,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
        },
    },
    "renewableGeneration": {
        "route": "electricity/electric-power-operational-data",
        "label": "Renewable Energy Generation",
        "description": "Monthly solar, wind, and hydro electricity generation",
        "params": {
            "data[]": ["generation"],
            "facets[fueltypeid][]": ["SUN", "WND", "HYC"],
            "facets[sectorid][]": ["99"],
            "frequency": "monthly",
            "length": 60,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
        },
    },
    "co2Emissions": {
        "route": "co2-emissions/co2-emissions-aggregates",
        "label": "U.S. CO2 Emissions",
        "description": "Annual U.S. total CO2 emissions across all sectors",
        "params": {
            "data[]": ["value"],
            "frequency": "annual",
            "facets[stateId][]": ["US"],
            "facets[sectorId][]": ["TT"],
            "length": 30,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
        },
    },
    "crudeOilProduction": {
        "route": "petroleum/crd/crpdn",
        "label": "U.S. Crude Oil Production",
        "description": "Monthly U.S. crude oil production by area",
        "params": {
            "data[]": ["value"],
            "frequency": "monthly",
            "length": 60,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
        },
    },
    "steo": {
        "route": "steo",
        "label": "Short-Term Energy Outlook (STEO)",
        "description": "EIA monthly short-term energy forecasts for world petroleum",
        "params": {
            "data[]": ["value"],
            "frequency": "monthly",
            "facets[seriesId][]": ["PATC_WORLD"],
            "length": 24,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
        },
    },
}


# ─── MCP Tool Definitions (Model Context Protocol) ────────────────────────────

MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "query_eia_data",
        "description": (
            "Query any EIA API v2 endpoint for energy data. "
            "Use this tool to fetch specific energy datasets by route, facets, and date range."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "description": "EIA API v2 route path (e.g. 'electricity/retail-sales', 'petroleum/pri/gnd')"
                },
                "data": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Data series to retrieve (e.g. ['price', 'sales', 'value'])"
                },
                "frequency": {
                    "type": "string",
                    "enum": ["annual", "monthly", "weekly", "hourly"],
                    "description": "Data frequency"
                },
                "facets": {
                    "type": "object",
                    "description": "Filter facets as key-value pairs (e.g. {'product': ['EPM0F']})"
                },
                "length": {
                    "type": "integer",
                    "description": "Number of records to return (default: 24)"
                },
                "start": {
                    "type": "string",
                    "description": "Start date (YYYY-MM for monthly, YYYY for annual)"
                },
                "end": {
                    "type": "string",
                    "description": "End date (YYYY-MM for monthly, YYYY for annual)"
                }
            },
            "required": ["route"]
        }
    },
    {
        "name": "explore_eia_datasets",
        "description": (
            "Explore available EIA API v2 routes, categories, and dataset metadata. "
            "Use this to discover what data is available before querying."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "description": "Route to explore (e.g. 'electricity', 'petroleum', ''  for root)"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_preset_data",
        "description": (
            "Fetch a pre-configured EIA dataset using a preset name. "
            "Presets are optimized queries for common energy research use cases."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "preset_name": {
                    "type": "string",
                    "enum": list(PRESET_QUERIES.keys()),
                    "description": "Name of the preset query to execute"
                }
            },
            "required": ["preset_name"]
        }
    }
]


# ─── Rate Limiting ─────────────────────────────────────────────────────────────

def _enforce_rate_limit():
    """Enforce 200ms minimum between API requests."""
    global _last_request_time
    now = time.time()
    elapsed_ms = (now - _last_request_time) * 1000
    if elapsed_ms < RATE_LIMIT_MS:
        sleep_ms = RATE_LIMIT_MS - elapsed_ms
        time.sleep(sleep_ms / 1000)
    _last_request_time = time.time()


# ─── Cache Helpers ─────────────────────────────────────────────────────────────

def _make_cache_key(url: str, params: Dict[str, Any]) -> str:
    """Create a deterministic cache key from URL and params."""
    key_str = url + json.dumps(params, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


# ─── Core API Request ─────────────────────────────────────────────────────────

def _make_request(
    url: str,
    params: Dict[str, Any],
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Make an EIA API request with:
    - Cache check (1hr TTL)
    - Rate limiting (200ms)
    - Retry with exponential backoff (3 attempts: 1s, 2s, 4s)
    """
    cache_key = _make_cache_key(url, params)
    
    # Check cache
    cached = storage.get_cache_entry(cache_key)
    if cached:
        storage.create_log(
            "cache_hit",
            {"url": url, "cache_key": cache_key[:16] + "...", "route": url.replace(EIA_BASE_URL, "")},
            conversation_id=conversation_id,
        )
        logger.info(f"Cache hit for {url}")
        return cached["response_data"]

    # Rate limit enforcement
    _enforce_rate_limit()

    last_error = None
    for attempt, delay in enumerate(RETRY_DELAYS, start=1):
        start_ms = time.time() * 1000
        try:
            response = requests.get(url, params=params, timeout=30)
            duration_ms = time.time() * 1000 - start_ms

            if response.status_code == 200:
                data = response.json()
                # Store in cache
                storage.set_cache_entry(cache_key, data, CACHE_TTL)
                storage.create_log(
                    "api_call",
                    {
                        "url": url,
                        "status": 200,
                        "attempt": attempt,
                        "records": len(data.get("response", {}).get("data", [])),
                    },
                    conversation_id=conversation_id,
                    duration_ms=duration_ms,
                )
                return data

            elif response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}"
                logger.warning(f"Retryable error {response.status_code} on attempt {attempt}, retrying in {delay}s")
                storage.create_log(
                    "api_retry",
                    {"url": url, "status": response.status_code, "attempt": attempt, "delay_s": delay},
                    conversation_id=conversation_id,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                continue

            else:
                error_text = response.text[:500]
                storage.create_log(
                    "api_error",
                    {"url": url, "status": response.status_code, "error": error_text},
                    conversation_id=conversation_id,
                    duration_ms=time.time() * 1000 - start_ms,
                )
                return {"error": f"HTTP {response.status_code}", "detail": error_text}

        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            logger.warning(f"Timeout on attempt {attempt}")
            if attempt < MAX_RETRIES:
                time.sleep(delay)
        except requests.exceptions.ConnectionError as e:
            last_error = str(e)
            logger.warning(f"Connection error on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(delay)
        except Exception as e:
            last_error = str(e)
            logger.error(f"Unexpected error: {e}")
            break

    storage.create_log(
        "error",
        {"url": url, "error": last_error, "attempts": MAX_RETRIES},
        conversation_id=conversation_id,
    )
    return {"error": last_error or "Unknown error after retries"}


# ─── Public API Functions ──────────────────────────────────────────────────────

def query_eia(params: Dict[str, Any], conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Query an EIA API v2 endpoint.
    
    params must include 'route' key. Other keys become query parameters.
    """
    route = params.pop("route", "")
    url = f"{EIA_BASE_URL}/{route.lstrip('/')}/data"
    
    query_params: Dict[str, Any] = {"api_key": EIA_API_KEY}
    
    # Handle data[], facets[], sort[] — expand lists properly
    for key, val in params.items():
        if isinstance(val, list):
            for item in val:
                query_params.setdefault(f"{key}[]", [])
                if isinstance(query_params[f"{key}[]"], list):
                    query_params[f"{key}[]"].append(item)
                else:
                    query_params[f"{key}[]"] = [query_params[f"{key}[]"], item]
        else:
            query_params[key] = val

    return _make_request(url, query_params, conversation_id)


def explore_eia_route(route: str = "", conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Explore EIA API v2 route metadata to discover available datasets.
    """
    url = f"{EIA_BASE_URL}/{route.lstrip('/')}" if route else EIA_BASE_URL
    params = {"api_key": EIA_API_KEY}
    return _make_request(url, params, conversation_id)


def get_preset(preset_name: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute a pre-configured preset query.
    Returns both the API response and the preset metadata.
    """
    if preset_name not in PRESET_QUERIES:
        return {"error": f"Unknown preset '{preset_name}'. Available: {list(PRESET_QUERIES.keys())}"}

    preset = PRESET_QUERIES[preset_name]
    route = preset["route"]
    params = dict(preset["params"])  # Copy to avoid mutation
    params["route"] = route

    result = query_eia(params, conversation_id)
    return {
        "preset": preset_name,
        "label": preset["label"],
        "description": preset["description"],
        "data": result,
    }


def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any], conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute an MCP tool by name with given arguments.
    This is the standardized MCP interface.
    """
    start_ms = time.time() * 1000

    try:
        if tool_name == "query_eia_data":
            route = arguments.get("route", "")
            params: Dict[str, Any] = {"route": route}
            if "data" in arguments:
                params["data"] = arguments["data"]
            if "frequency" in arguments:
                params["frequency"] = arguments["frequency"]
            if "facets" in arguments:
                for k, v in arguments["facets"].items():
                    params[f"facets[{k}]"] = v if isinstance(v, list) else [v]
            if "length" in arguments:
                params["length"] = arguments["length"]
            if "start" in arguments:
                params["start"] = arguments["start"]
            if "end" in arguments:
                params["end"] = arguments["end"]
            result = query_eia(params, conversation_id)

        elif tool_name == "explore_eia_datasets":
            route = arguments.get("route", "")
            result = explore_eia_route(route, conversation_id)

        elif tool_name == "get_preset_data":
            preset_name = arguments.get("preset_name", "")
            result = get_preset(preset_name, conversation_id)

        else:
            return {"error": f"Unknown MCP tool: {tool_name}"}

        duration_ms = time.time() * 1000 - start_ms
        storage.create_log(
            "mcp_tool",
            {"tool": tool_name, "arguments": arguments},
            conversation_id=conversation_id,
            duration_ms=duration_ms,
        )
        return result

    except Exception as e:
        storage.create_log(
            "error",
            {"tool": tool_name, "error": str(e)},
            conversation_id=conversation_id,
        )
        return {"error": str(e)}
