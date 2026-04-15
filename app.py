"""
EIA Energy Research Assistant — Streamlit Application
Author: Srikaran Anand (fsrikar@okstate.edu), Oklahoma State University
Course: Agentic AI Systems — Week 5 Capstone Project (Option 4: Research Assistant)

Run:  streamlit run app.py
"""

import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import storage
import agent
import eia_api

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EIA Energy Research Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Teal energy theme */
    .stApp { background-color: #f7fffe; }
    .block-container { max-width: 1200px; }
    
    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e0f2f1;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #0f766e; }
    .kpi-label { font-size: 0.85rem; color: #666; margin-top: 4px; }
    
    /* Trace log badges */
    .trace-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        margin-right: 6px;
    }
    .badge-guardrail { background: #7c3aed; }
    .badge-intent { background: #2563eb; }
    .badge-rag { background: #059669; }
    .badge-action { background: #d97706; }
    .badge-tool_selection { background: #9333ea; }
    .badge-synthesis { background: #0891b2; }
    .badge-eval { background: #dc2626; }
    .badge-error { background: #ef4444; }
    
    /* Eval score pills */
    .eval-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .eval-high { background: #d1fae5; color: #065f46; }
    .eval-mid { background: #fef3c7; color: #92400e; }
    .eval-low { background: #fee2e2; color: #991b1b; }
    
    /* Sidebar nav */
    [data-testid="stSidebar"] { background-color: #0d4f4f; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label { color: white !important; }
    
    /* Chat input fix */
    .stChatInput { border-color: #0f766e !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Initialize database + RAG on first run
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_app():
    storage.init_db()
    agent.initialize_rag()
    return True

init_app()

# ──────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    conv = storage.create_conversation("Research Session")
    st.session_state.conversation_id = conv["id"]
if "last_chart" not in st.session_state:
    st.session_state.last_chart = None
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None
if "last_steps" not in st.session_state:
    st.session_state.last_steps = []

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ EIA Research Assistant")
    st.markdown("Agentic AI Capstone Project")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔬 Research Chat", "📊 Dashboard", "🔍 Observability", "🏗 Architecture", "🔧 MCP Tools"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Srikaran Anand**")
    st.markdown("Oklahoma State University")
    st.markdown("April 2026")
    st.markdown("---")
    st.markdown(
        "<small>Data: <a href='https://www.eia.gov/opendata/' style='color:#7dd3d8'>EIA API v2</a></small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helper: render a Plotly chart from chart_data dict
# ══════════════════════════════════════════════════════════════════════════════
def render_chart(chart_data, chart_type="line"):
    if not chart_data:
        return
    labels = chart_data.get("labels", [])
    datasets = chart_data.get("datasets", [])
    if not labels or not datasets:
        return
    
    df = pd.DataFrame({"Period": labels})
    for ds in datasets:
        df[ds["label"]] = ds["data"]
    
    if chart_type == "bar":
        fig = px.bar(df, x="Period", y=[ds["label"] for ds in datasets],
                     color_discrete_sequence=["#0f766e", "#d97706", "#7c3aed"])
    else:
        fig = px.line(df, x="Period", y=[ds["label"] for ds in datasets],
                      color_discrete_sequence=["#0f766e", "#d97706", "#7c3aed"])
        fig.update_traces(fill="tozeroy", fillcolor="rgba(15,118,110,0.08)")
    
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=30, b=40),
        legend=dict(orientation="h", y=1.08),
        xaxis_title="",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)


def eval_pill(label, score):
    cls = "eval-high" if score >= 4 else ("eval-mid" if score >= 3 else "eval-low")
    return f'<span class="eval-pill {cls}">{label}: {score}/5</span>'


def trace_badge(event_type):
    return f'<span class="trace-badge badge-{event_type}">{event_type}</span>'


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: Research Chat
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔬 Research Chat":
    st.title("🔬 Energy Research Chat")
    st.caption("Ask any question about U.S. energy data — powered by EIA API v2 with Agentic RAG")

    # Suggested queries
    suggestions = [
        "What are current electricity prices?",
        "Show natural gas price trends",
        "Compare renewable energy generation",
        "What are U.S. CO2 emission trends?",
        "Explore available EIA datasets",
        "Show energy forecast outlook",
    ]
    cols = st.columns(3)
    for i, q in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_query = q

    st.markdown("---")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle pending suggestion click
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask about electricity, natural gas, petroleum, renewables, emissions...")
    query = pending or user_input

    if query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = agent.run_agent(query, st.session_state.conversation_id)

            # Store messages in DB
            storage.create_message(st.session_state.conversation_id, "user", query)
            storage.create_message(
                st.session_state.conversation_id, "assistant", result["answer"],
                metadata={"evaluation": result["evaluation"], "steps": len(result["steps"])}
            )

            # Display answer
            st.markdown(result["answer"])

            # Chart
            if result.get("chart_data"):
                st.session_state.last_chart = (result["chart_data"], result.get("chart_type", "line"))
                render_chart(result["chart_data"], result.get("chart_type", "line"))

            # Evaluation badges
            ev = result.get("evaluation", {})
            if ev and ev.get("score", 0) > 0:
                st.session_state.last_eval = ev
                pills = (
                    eval_pill("Accuracy", ev.get("factual_accuracy", 0))
                    + eval_pill("Relevance", ev.get("relevance", 0))
                    + eval_pill("Completeness", ev.get("completeness", 0))
                    + f' &nbsp; <strong>Overall: {ev.get("score", 0)}/5</strong>'
                )
                st.markdown(pills, unsafe_allow_html=True)
                st.caption(ev.get("reasoning", ""))

            # Agent trace expander
            steps = result.get("steps", [])
            st.session_state.last_steps = steps
            if steps:
                with st.expander(f"Agent Trace ({len(steps)} steps)", expanded=False):
                    for i, step in enumerate(steps, 1):
                        badge = trace_badge(step.get("type", "thought"))
                        st.markdown(f"{badge} **Step {i}**", unsafe_allow_html=True)
                        if step.get("thought"):
                            st.markdown(f"  💭 {step['thought']}")
                        if step.get("action"):
                            st.markdown(f"  🔧 {step['action']}")
                        if step.get("observation"):
                            st.markdown(f"  👁 {step['observation']}")
                        st.markdown("---")

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Telemetry Dashboard")
    st.caption("Real-time metrics from the agentic pipeline")

    # KPI cards
    stats = storage.get_log_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{stats['total_events']}</div>
            <div class="kpi-label">Total Events</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{stats['avg_latency_ms']:.0f}ms</div>
            <div class="kpi-label">Avg Latency</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{stats['cache_hits']}</div>
            <div class="kpi-label">Cache Hits</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{stats['errors']}</div>
            <div class="kpi-label">Errors</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("###")

    # Event type distribution chart
    evt_counts = stats.get("event_type_counts", {})
    if evt_counts:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Events by Type")
            df_evt = pd.DataFrame(list(evt_counts.items()), columns=["Event Type", "Count"])
            fig = px.bar(df_evt, x="Event Type", y="Count",
                         color_discrete_sequence=["#0f766e"])
            fig.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Event Distribution")
            fig2 = px.pie(df_evt, names="Event Type", values="Count",
                          color_discrete_sequence=px.colors.sequential.Teal)
            fig2.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=20, b=40))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No telemetry data yet. Ask a question in the Research Chat to generate events.")

    # Live EIA data preview
    st.markdown("---")
    st.subheader("Live EIA Data Preview")
    preset_choice = st.selectbox(
        "Select dataset",
        list(eia_api.PRESET_QUERIES.keys()),
        format_func=lambda x: eia_api.PRESET_QUERIES[x]["label"],
    )
    if st.button("Fetch Data", type="primary"):
        with st.spinner("Querying EIA API..."):
            result = eia_api.get_preset(preset_choice)
            data_resp = result.get("data", {})
            records = []
            if isinstance(data_resp, dict):
                records = data_resp.get("response", {}).get("data", [])
            if records:
                df = pd.DataFrame(records[:30])
                st.dataframe(df, use_container_width=True)
                
                # Try to chart it
                formatted = agent.format_data_for_response(result, eia_api.PRESET_QUERIES[preset_choice]["route"])
                if formatted.get("chart_data"):
                    render_chart(formatted["chart_data"], formatted.get("chart_type", "line"))
            else:
                st.warning("No records returned. The EIA API may be temporarily rate-limited with DEMO_KEY.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: Observability
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Observability":
    st.title("🔍 Agent Observability")
    st.caption("Full trace stream of the agentic ReAct pipeline")

    # Filters
    col1, col2 = st.columns([2, 1])
    with col2:
        limit = st.selectbox("Show last N events", [25, 50, 100, 200], index=0)

    logs = storage.get_logs(limit=limit)

    if not logs:
        st.info("No agent events logged yet. Ask a question in the Research Chat to generate traces.")
    else:
        # Summary stats row
        event_types = set(l["event_type"] for l in logs)
        badges_html = " ".join(trace_badge(et) for et in sorted(event_types))
        st.markdown(f"**Active event types:** {badges_html}", unsafe_allow_html=True)
        st.markdown("---")

        for log in logs:
            badge = trace_badge(log["event_type"])
            ts = log["created_at"][:19].replace("T", " ")
            dur = f' — {log["duration_ms"]:.0f}ms' if log.get("duration_ms") else ""

            with st.expander(f"{log['event_type'].upper()} — {ts}{dur}", expanded=False):
                st.markdown(badge, unsafe_allow_html=True)
                if log.get("conversation_id"):
                    st.caption(f"Conversation: {log['conversation_id'][:8]}...")
                details = log.get("details", {})
                st.json(details)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: Architecture
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗 Architecture":
    st.title("🏗 System Architecture")
    st.caption("EIA Energy Research Assistant — Agentic AI Pipeline")

    # Architecture flow
    st.subheader("Agent Pipeline Flow")
    flow_steps = [
        ("1. User Query", "Natural language question about energy"),
        ("2. Guardrails", "Input validation, prompt injection detection, topic relevance"),
        ("3. Intent Detection", "ReAct pattern — classify intent (trend, compare, forecast, analyze, explore)"),
        ("4. RAG Retrieval", "Keyword search over 8 energy domain knowledge chunks"),
        ("5. MCP Tool Selection", "Choose from: query_eia_data, explore_eia_datasets, get_preset_data"),
        ("6. EIA API v2 Query", "Caching (1hr TTL), retry (3x exponential backoff), rate limiting (200ms)"),
        ("7. Response Synthesis", "Combine RAG context + API data into markdown with tables and charts"),
        ("8. LLM-as-Judge Eval", "3-dimension scoring: factual accuracy, relevance, completeness (1-5)"),
        ("9. Observability Log", "Full trace with event_type, duration_ms, conversation_id"),
        ("10. Response to User", "Formatted answer + charts + evaluation badges + trace viewer"),
    ]

    cols = st.columns(5)
    for i, (title, desc) in enumerate(flow_steps[:5]):
        with cols[i]:
            st.markdown(f"""<div style="background:#0f766e; color:white; border-radius:10px; padding:14px; text-align:center; min-height:120px;">
                <strong>{title}</strong><br><small>{desc}</small>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align:center; font-size:1.5rem; margin: 8px 0;'>↓</div>", unsafe_allow_html=True)
    
    cols = st.columns(5)
    for i, (title, desc) in enumerate(flow_steps[5:]):
        with cols[i]:
            st.markdown(f"""<div style="background:#134e4a; color:white; border-radius:10px; padding:14px; text-align:center; min-height:120px;">
                <strong>{title}</strong><br><small>{desc}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Use Cases
    st.subheader("Use Cases")
    uc_col1, uc_col2, uc_col3 = st.columns(3)
    use_cases = [
        ("Price Analysis", "Query electricity and gas prices by state, sector, period"),
        ("Trend Exploration", "Visualize historical energy trends with auto-generated charts"),
        ("Renewable Monitoring", "Track solar, wind, hydro generation growth over time"),
        ("Emissions Tracking", "Analyze CO2 trends by sector and fuel type"),
        ("Dataset Discovery", "Explore available EIA datasets and their structure"),
        ("Energy Forecasting", "Access STEO short-term energy outlook projections"),
    ]
    for i, (title, desc) in enumerate(use_cases):
        with [uc_col1, uc_col2, uc_col3][i % 3]:
            st.markdown(f"""<div style="background:white; border:1px solid #e0f2f1; border-radius:10px; padding:16px; margin-bottom:12px; border-top:3px solid #0f766e;">
                <strong>{title}</strong><br><small style="color:#666">{desc}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Tech Stack
    st.subheader("Technology Stack")
    ts_col1, ts_col2 = st.columns(2)
    with ts_col1:
        st.markdown("""
        | Layer | Technologies |
        |-------|-------------|
        | **Frontend** | Streamlit, Plotly, Pandas |
        | **Backend** | Python, SQLite |
        | **AI / Agent** | ReAct Pattern, RAG, MCP Protocol, LLM-as-Judge |
        | **Data Source** | EIA API v2 — electricity, natural gas, petroleum, renewables, emissions, STEO |
        """)
    with ts_col2:
        st.markdown("""
        **Phase 1 — Architecture:**  Use case definition, ReAct agentic pattern
        
        **Phase 2 — Development:**  Agentic RAG, MCP Protocol, Flow Engineering
        
        **Phase 3 — Reliability:**  Guardrails, LLM-as-Judge, Caching, Retry, Rate Limiting, Observability
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MCP Tools
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 MCP Tools":
    st.title("🔧 MCP Tools (Model Context Protocol)")
    st.caption("Standardized tool interface for EIA API interactions")

    # List MCP tools
    st.subheader("Available Tools")
    for tool in eia_api.MCP_TOOLS:
        with st.expander(f"🔧 {tool['name']}", expanded=False):
            st.markdown(f"**Description:** {tool['description']}")
            st.markdown("**Parameters:**")
            st.json(tool["parameters"])

    st.markdown("---")

    # Interactive tool execution
    st.subheader("Execute MCP Tool")
    tool_name = st.selectbox("Select tool", [t["name"] for t in eia_api.MCP_TOOLS])

    if tool_name == "get_preset_data":
        preset = st.selectbox("Preset", list(eia_api.PRESET_QUERIES.keys()),
                              format_func=lambda x: f"{x} — {eia_api.PRESET_QUERIES[x]['label']}")
        args = {"preset_name": preset}
    elif tool_name == "explore_eia_datasets":
        route = st.text_input("Route (leave blank for root)", value="")
        args = {"route": route}
    else:
        route = st.text_input("Route", value="electricity/retail-sales")
        data_fields = st.text_input("Data fields (comma-separated)", value="price,sales")
        freq = st.selectbox("Frequency", ["monthly", "annual", "weekly"])
        length = st.number_input("Records", value=24, min_value=1, max_value=200)
        args = {
            "route": route,
            "data": [d.strip() for d in data_fields.split(",")],
            "frequency": freq,
            "length": length,
        }

    if st.button("Execute", type="primary"):
        with st.spinner(f"Executing {tool_name}..."):
            result = eia_api.execute_mcp_tool(tool_name, args, st.session_state.conversation_id)
        
        if isinstance(result, dict) and "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            # Show raw result
            records = []
            if isinstance(result, dict):
                if "data" in result and isinstance(result["data"], dict):
                    records = result["data"].get("response", {}).get("data", [])
                elif "response" in result:
                    records = result.get("response", {}).get("data", [])

            if records:
                st.success(f"Retrieved {len(records)} records")
                df = pd.DataFrame(records[:30])
                st.dataframe(df, use_container_width=True)
            else:
                st.json(result)

    st.markdown("---")

    # Guardrails test
    st.subheader("Test Guardrails")
    test_input = st.text_input("Test input", value="What are current electricity prices?")
    if st.button("Test Guardrails"):
        result = agent.apply_guardrails(test_input)
        if result["passed"]:
            st.success(f"✅ PASSED — {result['reason']}")
        else:
            st.error(f"❌ BLOCKED — {result['reason']}")

    # Eval test
    st.subheader("Test LLM-as-Judge Evaluation")
    eval_query = st.text_input("Query", value="What are electricity prices?", key="eval_q")
    eval_response = st.text_area("Response to evaluate", value="U.S. electricity prices average 12-16 cents per kWh according to EIA data.", key="eval_r")
    if st.button("Run Evaluation"):
        ev = agent.evaluate_response(eval_query, eval_response)
        pills = (
            eval_pill("Accuracy", ev["factual_accuracy"])
            + eval_pill("Relevance", ev["relevance"])
            + eval_pill("Completeness", ev["completeness"])
        )
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown(f"**Overall: {ev['score']}/5**")
        st.caption(ev["reasoning"])
