# app.py (Chatbot-style minimal UI)
import os
import re
import textwrap
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Optional providers (lazy)
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ClickHouse client
try:
    import clickhouse_connect  # type: ignore
except Exception:
    clickhouse_connect = None


# =========================
# Safety & extraction utils
# =========================
_BANNED_PATTERNS = re.compile(
    r"""
    \b(
        insert|update|delete|merge|upsert|replace|
        create|alter|drop|truncate|attach|detach|
        grant|revoke|rename|
        optimize|kill|
        set\s+(\w+)|use\s+\w+|
        call|begin|commit|rollback|
        vacuum|analyze
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SELECT_HEAD = re.compile(
    r"""^\s*(?:with\b[\s\S]+?\)\s*)?select\b""",
    re.IGNORECASE,
)

_CODE_FENCE = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)
def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*[\s\S]*?\*/", " ", sql)
    sql = re.sub(r"--[^\n\r]*", " ", sql)
    return sql

def strip_explanations(text: str) -> str:
    text = (text or "").strip()
    m = _CODE_FENCE.search(text)
    if m:
        cand = m.group(1).strip()
    else:
        s = re.search(r"(with\b[\s\S]*?select\b|select\b)", text, re.IGNORECASE)
        if not s:
            return ""
        cand = text[s.start():].strip()
    semi = cand.find(";")
    if semi != -1:
        cand = cand[: semi + 1]
    return cand.strip()

def is_select_only(sql: str) -> bool:
    if not sql or not isinstance(sql, str):
        return False
    cleaned = _strip_sql_comments(sql).strip()
    if not cleaned:
        return False
    if cleaned.count(";") > 1:
        return False
    if not _SELECT_HEAD.match(cleaned):
        return False
    if _BANNED_PATTERNS.search(cleaned):
        return False
    return True


# =========================
# DDL & schema summary (from file path in env)
# =========================
DDL_DEFAULT_PATH = os.getenv("DDL_PATH", "orbit_japa.sql")

def parse_clickhouse_ddl(ddl_text: str) -> Dict[str, Dict[str, str]]:
    tables: Dict[str, Dict[str, str]] = {}
    txt = ddl_text
    for m in re.finditer(r"CREATE\s+TABLE\s+([.\w`]+)\s*\((.*?)\)\s*ENGINE", txt, re.IGNORECASE | re.DOTALL):
        raw_name = m.group(1).strip().strip('`')
        cols_block = m.group(2)
        cols_dict: Dict[str, str] = {}
        for line in cols_block.splitlines():
            line = line.strip().strip(",")
            if not line or line.startswith("--"):
                continue
            mm = re.match(r"`?([A-Za-z0-9_]+)`?\s+([A-Za-z0-9_]+(?:\([^)]*\))?)", line)
            if mm:
                col, typ = mm.group(1), mm.group(2)
                cols_dict[col] = typ
        tables[raw_name] = cols_dict
    return tables

def make_schema_card(tables: Dict[str, Dict[str, str]]) -> str:
    parts = []
    for t, cols in sorted(tables.items()):
        col_str = ", ".join(f"{c}:{typ}" for c, typ in cols.items())
        parts.append(f"{t} → {col_str}")
    return "\n".join(parts)


# =========================
# Prompting
# =========================
SYSTEM_RULES = """
You answer the user's question. If the question is about the Japa analytics data, you convert it into EXACTLY ONE SQL SELECT statement for ClickHouse and (optionally) execute it.
Rules for SQL:
- Output only SQL (no backticks) when producing SQL.
- Use table/column names from the provided schema.
- Prefer these tables when relevant:
  * user join date / active status → orbit_japa.j_user (created_at, is_deleted)
  * per-session facts → orbit_japa.j_user_chant (start_time, end_time, session_duration_sec, mala_count)
  * daily aggregates → orbit_japa.j_user_chant_category_metrics_daily
- Use ClickHouse date helpers (today(), now(), toDate(), toYYYYMM(), toYear(), toStartOfWeek()) correctly.
- Single SELECT only (CTEs allowed). No DDL/DML/transactions.
- If the question is casual chit-chat or cannot be answered with a single SELECT, DO NOT output SQL. Just answer naturally.
"""

def build_examples() -> str:
    # Minimal few-shots tuned to the three most-used tables
    return """
Q: How many users completed at least one session today?
SQL:
SELECT countDistinct(user_identifier) AS users_completed_today
FROM orbit_japa.j_user_chant
WHERE toDate(start_time) = today()
  AND end_time IS NOT NULL;

Q: List all active users who joined in the last 30 days
SQL:
SELECT user_identifier, full_name, created_at
FROM orbit_japa.j_user
WHERE is_deleted = 0
  AND created_at >= now() - INTERVAL 30 DAY
ORDER BY created_at DESC;

Q: Daily trend of total sessions for the last 14 days
SQL:
SELECT period_start, SUM(total_session_count) AS total_sessions
FROM orbit_japa.j_user_chant_category_metrics_daily
WHERE period_start >= today() - 14
GROUP BY period_start
ORDER BY period_start ASC;
""".strip()

def build_sql_prompt(schema_card: str, question: str) -> str:
    examples = build_examples()
    checklist = """
PLAN BEFORE YOU WRITE SQL:
1) Pick the correct tables for the question.
2) Identify required filters and dates (today, last N days).
3) Include fields required by the question (e.g., created_at for "joined").
4) Use precise conditions and aliases.
If the question is not about data analysis, DO NOT write SQL.
""".strip()
    return textwrap.dedent(f"""
    {SYSTEM_RULES}

    SCHEMA (table → columns:types):
    {schema_card}

    EXAMPLES:
    {examples}

    {checklist}

    QUESTION:
    {question}

    SQL:
    """).strip()


# =========================
# LLM helpers (SQL + chat)
# =========================
def complete_with_gemini(api_key: str, model: str, prompt: str) -> str:
    if genai is None or not api_key:
        return ""
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model or "gemini-1.5-pro")
    r = m.generate_content(prompt)
    return (getattr(r, "text", "") or "").strip()

def complete_with_openai(api_key: str, model: str, prompt: str) -> str:
    if OpenAI is None or not api_key:
        return ""
    client = OpenAI(api_key=api_key)
    try:
        r = client.responses.create(model=model or "gpt-4o-mini", input=prompt)
        return (getattr(r, "output_text", "") or "").strip()
    except Exception:
        try:
            chat = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role":"system","content":"You output only SQL."},
                          {"role":"user","content":prompt}],
                temperature=0.0,
            )
            return (chat.choices[0].message.content or "").strip()
        except Exception:
            return ""

def llm_generate_sql(provider: str, api_key: str, model: str, prompt: str) -> str:
    if provider == "Gemini":
        return complete_with_gemini(api_key, model, prompt)
    if provider == "OpenAI":
        return complete_with_openai(api_key, model, prompt)
    return "SELECT 1 WHERE 0;"

def llm_chat(provider: str, api_key: str, model: str, user_text: str) -> str:
    # A simple general-purpose chat response (no SQL constraint)
    system = "You are a friendly, concise assistant. Answer helpfully."
    if provider == "Gemini" and api_key and genai is not None:
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model or "gemini-1.5-pro")
        r = m.generate_content(f"{system}\n\nUser: {user_text}\nAssistant:")
        return (getattr(r, "text", "") or "").strip()
    if provider == "OpenAI" and api_key and OpenAI is not None:
        client = OpenAI(api_key=api_key)
        try:
            chat = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role":"system","content":system},
                          {"role":"user","content":user_text}],
                temperature=0.5,
            )
            return (chat.choices[0].message.content or "").strip()
        except Exception:
            return "Sorry, I couldn't reach the LLM right now."
    # Fallback
    return "Hi! 👋 How can I help?"


# =========================
# Validator for question ↔ SQL alignment
# =========================
def validate_sql_against_question(question: str, sql: str) -> Optional[str]:
    q = (question or "").lower()
    s = (sql or "").lower()
    # If "join" intent but missing j_user/created_at
    if any(k in q for k in ["join", "joined", "signup", "signed up", "created"]):
        if "orbit_japa.j_user" not in s or "created_at" not in s:
            return "Use orbit_japa.j_user with created_at (and is_deleted = 0 if active)."
    # If "active" not enforced
    if "active" in q and "is_deleted" not in s:
        return "Filter orbit_japa.j_user.is_deleted = 0 for active users."
    # If "completed session today"
    if ("today" in q or "today?" in q) and "session" in q:
        if "orbit_japa.j_user_chant" not in s or "end_time is not null" not in s:
            return "Use orbit_japa.j_user_chant with end_time IS NOT NULL and toDate(start_time) = today()."
    return None


# =========================
# ClickHouse execution
# =========================
def run_clickhouse_select(sql: str, host: str, port: int, username: str, password: str, database: str, secure: bool):
    if clickhouse_connect is None:
        raise RuntimeError("clickhouse-connect is not installed.")
    client = clickhouse_connect.get_client(
        host=host, port=port, username=username, password=password,
        database=database or None, secure=secure,
        connect_timeout=10, send_receive_timeout=120,
    )
    try:
        df = client.query_df(sql)
        cols = list(df.columns)
        return df, cols
    finally:
        client.close()


# =========================
# Env config
# =========================
PROVIDER = os.getenv("LLM_PROVIDER", "Gemini")  # Gemini | OpenAI | Mock
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini" if PROVIDER == "OpenAI" else "gemini-1.5-pro")

EXECUTE_SQL = os.getenv("EXECUTE_SQL", "false").lower() in {"1","true","yes"}
SUMMARIZE = os.getenv("SUMMARIZE", "true").lower() in {"1","true","yes"}

CH_HOST = os.getenv("CLICKHOUSE_HOST", "")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", "8443"))
CH_USER = os.getenv("CLICKHOUSE_USER", "")
CH_PASS = os.getenv("CLICKHOUSE_PASSWORD", "")
CH_DB = os.getenv("CLICKHOUSE_DATABASE", "")
CH_SECURE = os.getenv("CLICKHOUSE_SECURE", "true").lower() in {"1","true","yes"}


# =========================
# Minimal Chatbot UI
# =========================
st.set_page_config(page_title="Japa Text‑to‑SQL Chat", page_icon="🧘", layout="centered")
st.title("🧘Text‑to‑SQL Analytical Chatbot")

# Prepare DDL schema_card (no UI noise)
ddl_text = ""
if os.path.exists(DDL_DEFAULT_PATH):
    with open(DDL_DEFAULT_PATH, "r", encoding="utf-8") as f:
        ddl_text = f.read()
tables = parse_clickhouse_ddl(ddl_text) if ddl_text else {}
schema_card = make_schema_card(tables) if tables else "-- (no tables parsed)"

# Chat history
if "chat" not in st.session_state:
    st.session_state["chat"] = []

for turn in st.session_state["chat"]:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant" and turn.get("is_sql") and turn.get("sql"):
            st.markdown(turn["content"])
            st.code(turn["sql"], language="sql")
        else:
            st.markdown(turn["content"])

user_msg = st.chat_input("Type your message…")
if user_msg:
    st.session_state["chat"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Try to produce SQL for data-related queries
    api_key = GEMINI_API_KEY if PROVIDER == "Gemini" else OPENAI_API_KEY
    sql_prompt = build_sql_prompt(schema_card=schema_card, question=user_msg)
    raw = llm_generate_sql(provider=PROVIDER, api_key=api_key or "", model=LLM_MODEL or "", prompt=sql_prompt)
    sql = strip_explanations(raw).strip()

    # One correction pass
    hint = validate_sql_against_question(user_msg, sql)
    if hint:
        sql_prompt2 = sql_prompt + "\n\n# CORRECTION: " + hint + "\nSQL:"
        raw2 = llm_generate_sql(provider=PROVIDER, api_key=api_key or "", model=LLM_MODEL or "", prompt=sql_prompt2)
        sql2 = strip_explanations(raw2).strip()
        if is_select_only(sql2):
            sql = sql2

    # Decide: SQL answer or chit-chat
    sql_is_valid = is_select_only(sql) and ("orbit_japa." in sql or " from " in sql.lower())
    if sql_is_valid:
        # Build assistant message with SQL + optional execution + summary
        blocks = []
        blocks.append("Here you go:")
        exec_df = None
        if EXECUTE_SQL and CH_HOST and CH_USER:
            try:
                exec_df, cols = run_clickhouse_select(sql, CH_HOST, CH_PORT, CH_USER, CH_PASS, CH_DB, CH_SECURE)
            except Exception as e:
                blocks.append(f"_Execution error:_ `{e}`")

        # render
        with st.chat_message("assistant"):
            st.markdown("\n\n".join(blocks))
            st.code(sql, language="sql")
            if exec_df is not None:
                st.dataframe(exec_df, use_container_width=True)
                if SUMMARIZE:
                    # summarizer
                    # Reuse provider+key; concise summary
                    preview_rows = min(30, len(exec_df))
                    preview = exec_df.head(preview_rows).to_csv(index=False)
                    summ_prompt = f"""
You are a SQL result summarizer. Your job is to describe the result table truthfully, concisely, and without guessing.

Rules:
1. ONLY describe facts you can directly observe from the query results.
2. If a condition (e.g., active users, join dates, leaderboard status) cannot be confirmed because required columns are missing, explicitly state: "This data does not include <missing_field>, so <metric> cannot be determined."
3. Do not assume all rows meet a condition unless there is an explicit column verifying it.
4. Be brief:
   - If result is a count → state the count
   - If result is a list → say how many items and give up to 3 examples
   - If result is grouped → highlight top categories or values
5. Never speculate or infer missing dates/fields — be precise and literal.

Example:
- Result: `COUNT=45` → "There are 45 users in the dataset."
- Missing join_date → "Join dates not present; cannot confirm 'joined in last 30 days'."
- Grouped totals → "Top categories: A (120), B (95), C (60)."


QUESTION:
{user_msg}

CSV:
{preview}

ANSWER:
""".strip()
                    summary = ""
                    if PROVIDER == "Gemini":
                        summary = complete_with_gemini(api_key, LLM_MODEL, summ_prompt)
                    elif PROVIDER == "OpenAI":
                        summary = complete_with_openai(api_key, LLM_MODEL, summ_prompt)
                    if not summary:
                        summary = f"Returned {len(exec_df)} rows × {len(exec_df.columns)} cols."
                    st.markdown(summary)

        st.session_state["chat"].append({"role": "assistant", "content": "\n\n".join(blocks), "is_sql": True, "sql": sql})
    else:
        # Casual chat
        reply = llm_chat(PROVIDER, api_key, LLM_MODEL, user_msg)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state["chat"].append({"role": "assistant", "content": reply})

