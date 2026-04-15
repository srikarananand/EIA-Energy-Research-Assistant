"""
SQLite storage layer for EIA Energy Research Assistant
Author: Srikaran Anand (fsrikar@okstate.edu), Oklahoma State University
Course: Agentic AI Systems - Capstone Project (Option 4: Research Assistant)
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

DB_PATH = "eia_assistant.db"


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize all database tables."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS eia_cache (
                id TEXT PRIMARY KEY,
                cache_key TEXT UNIQUE NOT NULL,
                response_data TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_logs (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL,
                duration_ms REAL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_logs_conversation ON agent_logs(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_logs_event_type ON agent_logs(event_type);
            CREATE INDEX IF NOT EXISTS idx_cache_key ON eia_cache(cache_key);
        """)


# ─── Conversations ────────────────────────────────────────────────────────────

def create_conversation(title: str, conv_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a new conversation."""
    conv_id = conv_id or str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
            (conv_id, title, now)
        )
    return {"id": conv_id, "title": title, "created_at": now}


def get_conversations() -> List[Dict[str, Any]]:
    """Get all conversations ordered by creation date."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    """Get a single conversation by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    return dict(row) if row else None


# ─── Messages ─────────────────────────────────────────────────────────────────

def create_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    msg_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new message in a conversation."""
    msg_id = msg_id or str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    meta_str = json.dumps(metadata or {})
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, conversation_id, role, content, meta_str, now)
        )
    return {
        "id": msg_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "created_at": now
    }


def get_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """Get all messages for a conversation."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d["metadata"] or "{}")
        except Exception:
            d["metadata"] = {}
        result.append(d)
    return result


# ─── Cache ────────────────────────────────────────────────────────────────────

def get_cache_entry(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get a cache entry if it exists and is not expired."""
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM eia_cache WHERE cache_key = ? AND expires_at > ?",
            (cache_key, now)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["response_data"] = json.loads(d["response_data"])
    except Exception:
        pass
    return d


def set_cache_entry(cache_key: str, response_data: Any, ttl_seconds: int = 3600):
    """Store a cache entry with TTL."""
    from datetime import timedelta
    now = datetime.utcnow()
    expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
    cache_id = str(uuid.uuid4())
    data_str = json.dumps(response_data)
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO eia_cache (id, cache_key, response_data, expires_at, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (cache_id, cache_key, data_str, expires_at, now.isoformat())
        )


def clear_expired_cache():
    """Remove all expired cache entries."""
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute("DELETE FROM eia_cache WHERE expires_at <= ?", (now,))


# ─── Logs ─────────────────────────────────────────────────────────────────────

def create_log(
    event_type: str,
    details: Dict[str, Any],
    conversation_id: Optional[str] = None,
    duration_ms: Optional[float] = None
) -> Dict[str, Any]:
    """Create an observability log entry."""
    log_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    details_str = json.dumps(details)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO agent_logs (id, conversation_id, event_type, details, duration_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (log_id, conversation_id, event_type, details_str, duration_ms, now)
        )
    return {
        "id": log_id,
        "conversation_id": conversation_id,
        "event_type": event_type,
        "details": details,
        "duration_ms": duration_ms,
        "created_at": now
    }


def get_logs(conversation_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Get agent logs, optionally filtered by conversation."""
    with get_db() as conn:
        if conversation_id:
            rows = conn.execute(
                "SELECT * FROM agent_logs WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agent_logs ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["details"] = json.loads(d["details"] or "{}")
        except Exception:
            d["details"] = {}
        result.append(d)
    return result


def get_log_stats() -> Dict[str, Any]:
    """Get aggregated statistics from agent logs."""
    with get_db() as conn:
        # Total counts by event type
        type_counts = conn.execute(
            "SELECT event_type, COUNT(*) as count FROM agent_logs GROUP BY event_type"
        ).fetchall()
        
        # Average latency
        avg_latency = conn.execute(
            "SELECT AVG(duration_ms) as avg_ms FROM agent_logs WHERE duration_ms IS NOT NULL"
        ).fetchone()
        
        # Cache hit count
        cache_hits = conn.execute(
            "SELECT COUNT(*) as count FROM agent_logs WHERE event_type = 'cache_hit'"
        ).fetchone()
        
        # Error count
        errors = conn.execute(
            "SELECT COUNT(*) as count FROM agent_logs WHERE event_type = 'error'"
        ).fetchone()
        
        # Total events
        total = conn.execute(
            "SELECT COUNT(*) as count FROM agent_logs"
        ).fetchone()

    type_counts_dict = {r["event_type"]: r["count"] for r in type_counts}
    return {
        "total_events": total["count"] if total else 0,
        "avg_latency_ms": round(avg_latency["avg_ms"] or 0, 2) if avg_latency else 0,
        "cache_hits": cache_hits["count"] if cache_hits else 0,
        "errors": errors["count"] if errors else 0,
        "event_type_counts": type_counts_dict,
    }


# ─── RAG Chunks ───────────────────────────────────────────────────────────────

def create_rag_chunk(
    source: str,
    content: str,
    embedding: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a RAG knowledge chunk."""
    chunk_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    meta_str = json.dumps(metadata or {})
    with get_db() as conn:
        conn.execute(
            """INSERT INTO rag_chunks (id, source, content, embedding, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (chunk_id, source, content, embedding, meta_str, now)
        )
    return {
        "id": chunk_id,
        "source": source,
        "content": content,
        "embedding": embedding,
        "metadata": metadata or {},
        "created_at": now
    }


def get_all_rag_chunks() -> List[Dict[str, Any]]:
    """Get all RAG chunks from the knowledge base."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM rag_chunks ORDER BY created_at ASC").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d["metadata"] or "{}")
        except Exception:
            d["metadata"] = {}
        result.append(d)
    return result


def search_rag_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Keyword-based search over RAG chunks."""
    keywords = [w.lower().strip() for w in query.split() if len(w) > 2]
    if not keywords:
        return get_all_rag_chunks()[:top_k]

    with get_db() as conn:
        rows = conn.execute("SELECT * FROM rag_chunks").fetchall()

    scored = []
    for row in rows:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d["metadata"] or "{}")
        except Exception:
            d["metadata"] = {}
        text = (d["content"] + " " + d["source"]).lower()
        score = sum(text.count(kw) for kw in keywords)
        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def count_rag_chunks() -> int:
    """Return count of RAG chunks in the database."""
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as c FROM rag_chunks").fetchone()
    return row["c"] if row else 0
