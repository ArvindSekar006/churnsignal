"""
ChurnSignal — FastAPI Backend
AI Customer Health & Churn Prevention Copilot
Arvind Sekar | Capstone Project

v4:
  - RAG: playbooks are embedded and retrieved by a get_playbook tool
  - Guardrail: intent-based write detection (regex) instead of a short phrase list
  - Monitoring: structured JSON logs (account ids hashed), /api/metrics
  - Agent loop: forced final answer, argument validation, per-tool error handling
  - Memory: entity register (last explicitly named account) + conflict notice
  - Feedback: /api/feedback; corrections change later answers in the same session
  - Prompt variants A / B / C kept for evaluation
"""

import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.tools import tool
from pydantic import BaseModel

app = FastAPI(title="ChurnSignal API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LOGGING & METRICS ─────────────────────────────────────────────────────────
# Every request writes one JSON line to stdout (visible in Render logs).
# Account ids are hashed and message text is never logged, so logs stay PII-safe.

logger = logging.getLogger("churnsignal")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

METRICS = {
    "requests": 0,
    "blocked_keyword": 0,
    "errors": 0,
    "escalations": 0,
    "feedback_up": 0,
    "feedback_down": 0,
    "by_variant": {"A": 0, "B": 0, "C": 0},
}
LATENCIES = deque(maxlen=500)  # milliseconds, last 500 requests


def mask(value: Optional[str]) -> Optional[str]:
    """Stable, non-reversible tag for an account id (PII-safe logging)."""
    if not value:
        return None
    return hashlib.sha256(value.lower().encode()).hexdigest()[:8]


def log_event(event: str, **fields):
    logger.info(json.dumps({"ts": round(time.time(), 3), "event": event, **fields}))


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[idx], 1)


# ── SUPABASE CLIENT ────────────────────────────────────────────────────────────

def get_supabase():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    return create_client(url, key)


# ── KNOWN ACCOUNTS & ENTITY REGISTER ──────────────────────────────────────────

KNOWN_ACCOUNTS = {
    "globalbank": ["globalbank", "global bank"],
    "techstartup": ["techstartup", "tech startup"],
    "retailco": ["retailco", "retail co"],
}


def normalise_account(account_id: str) -> Optional[str]:
    """Return a valid account slug, or None if the id is unknown."""
    slug = re.sub(r"[^a-z0-9]", "", (account_id or "").lower())
    return slug if slug in KNOWN_ACCOUNTS else None


def accounts_in_text(text: str) -> List[str]:
    """Known accounts named in text, in order of appearance."""
    lowered = (text or "").lower()
    found = []
    for slug, aliases in KNOWN_ACCOUNTS.items():
        positions = [lowered.find(a) for a in aliases if a in lowered]
        if positions:
            found.append((min(positions), slug))
    return [slug for _, slug in sorted(found)]


def build_entity_register(message: str, history: List[dict], account_id: Optional[str]):
    """
    Entity register: the most recently, explicitly named account.
    Only USER turns count; assistant turns list every account (e.g. a risk
    ranking), which is what made 'them' resolve to the highest-risk account.
    """
    named_now = accounts_in_text(message)
    last_named = named_now[-1] if named_now else None
    if not last_named:
        for h in reversed(history or []):
            if h.get("role") == "user":
                mentioned = accounts_in_text(h.get("content", ""))
                if mentioned:
                    last_named = mentioned[-1]
                    break
    selected = normalise_account(account_id) if account_id else None

    conflict = None
    if selected and named_now and selected not in named_now:
        conflict = (selected, named_now[-1])
    return {"last_named": last_named, "selected": selected, "conflict": conflict}


def register_notice(reg: dict) -> str:
    lines = []
    if reg["conflict"]:
        sel, named = reg["conflict"]
        lines.append(
            f"[Context conflict: the CSM is viewing '{sel}' but their message names '{named}'. "
            f"The message wins: answer about '{named}' and say which account you used.]"
        )
    if reg["last_named"]:
        lines.append(
            f"[Entity register: last explicitly named account = '{reg['last_named']}'. "
            f"If the CSM says 'them', 'it' or 'that account' without naming one, "
            f"resolve it to '{reg['last_named']}'. Do not pick the highest-risk account by default.]"
        )
    elif reg["selected"]:
        lines.append(f"[Account context: {reg['selected']}]")
    return "\n".join(lines)


# ── RISK HELPER ───────────────────────────────────────────────────────────────

def risk_from_score(health_score: int) -> str:
    if health_score < 30:
        return "CRITICAL"
    if health_score < 50:
        return "HIGH"
    if health_score < 70:
        return "MEDIUM"
    return "LOW"


# ── PLAYBOOKS & RAG ───────────────────────────────────────────────────────────
# Each playbook is one retrievable document. Pipeline:
#   playbook text -> embedding (fastembed, BAAI/bge-small-en-v1.5) -> cosine
#   similarity against the query embedding -> top-k returned by get_playbook.
# If the embedding model cannot load (e.g. offline, low memory) retrieval falls
# back to keyword overlap so the agent never fails because of RAG.

PLAYBOOK_DOCS = [
    {"id": "critical_high_tickets", "title": "CRITICAL RISK PLAYBOOK — High Ticket Volume + Low Engagement",
     "text": "Schedule emergency QBR within 48 hours. Involve executive sponsor on both sides. "
             "Offer complimentary Success Workshop to re-onboard the team. Weekly cadence calls for "
             "minimum 4 weeks. Escalate to VP Customer Success if no improvement in 2 weeks."},
    {"id": "critical_usage_drop", "title": "CRITICAL RISK PLAYBOOK — Usage Drop + Low NPS",
     "text": "Schedule discovery call within 24 hours to understand pain points. Review product adoption "
             "metrics: feature usage, login frequency, workflow completion. Propose 30-day success "
             "milestone plan with personalized onboarding. Assign dedicated CSM for 60 days."},
    {"id": "high_risk", "title": "HIGH RISK PLAYBOOK — Elevated Ticket Volume + Moderate Disengagement",
     "text": "Schedule QBR within 2 weeks. Conduct ticket clustering analysis to identify recurring issue "
             "categories. Share product roadmap updates relevant to their pain points. Set up health "
             "score alerts for weekly monitoring."},
    {"id": "qbr_guide", "title": "QBR PREPARATION GUIDE",
     "text": "Review success metrics since last QBR. Open ticket audit: categorise by age, severity, owner. "
             "Usage and adoption trends: feature heatmap, login frequency. Product roadmap items relevant "
             "to their stated pain points. Competitive threats: check if competitors mentioned in tickets. "
             "Joint success plan with 3 specific commitments on both sides."},
    {"id": "escalation_criteria", "title": "ESCALATION CRITERIA",
     "text": "CRITICAL risk and contract value >= $50K: VP Customer Success within 24 hours. Competitor "
             "mentioned in ticket or call: immediate Retention Team involvement. Renewal within 60 days "
             "and HIGH or CRITICAL risk: involve Account Executive. No login for more than 14 days at any "
             "risk level: CSM outreach required same day."},
    {"id": "medium_risk", "title": "MEDIUM RISK PLAYBOOK",
     "text": "Proactive check-in within 3 business days. Share relevant new features and product updates. "
             "Offer optional training session or webinar. Monthly health review."},
    {"id": "low_risk", "title": "LOW RISK PLAYBOOK",
     "text": "Quarterly health check-in. Share product newsletter and roadmap updates. Consider for "
             "reference customer or case study program."},
]

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class PlaybookRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.model = None
        self.matrix = None
        self.mode = "uninitialised"

    def _init(self):
        if self.mode != "uninitialised":
            return
        try:
            import numpy as np
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name=EMBED_MODEL_NAME)
            texts = [f"{d['title']}. {d['text']}" for d in self.docs]
            vectors = np.array(list(self.model.embed(texts)), dtype="float32")
            self.matrix = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            self.mode = "embeddings"
        except Exception as e:  # keep the agent alive; report degraded mode
            log_event("rag_fallback", reason=str(e)[:120])
            self.mode = "keyword_fallback"

    @staticmethod
    def _tokens(text):
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def search(self, query: str, k: int = 2):
        self._init()
        if self.mode == "embeddings":
            import numpy as np
            q = np.array(list(self.model.embed([query])), dtype="float32")[0]
            q = q / np.linalg.norm(q)
            scores = self.matrix @ q
            order = np.argsort(-scores)[:k]
            return [{**self.docs[i], "score": round(float(scores[i]), 3)} for i in order]
        q_tokens = self._tokens(query)
        scored = []
        for d in self.docs:
            d_tokens = self._tokens(d["title"] + " " + d["text"])
            scored.append((len(q_tokens & d_tokens) / (len(q_tokens) or 1), d))
        scored.sort(key=lambda x: -x[0])
        return [{**d, "score": round(s, 3)} for s, d in scored[:k]]


RETRIEVER = PlaybookRetriever(PLAYBOOK_DOCS)
PLAYBOOK_TITLES = "\n".join(f"- {d['title']}" for d in PLAYBOOK_DOCS)


# ── LANGCHAIN TOOLS ────────────────────────────────────────────────────────────

UNKNOWN_ACCOUNT_MSG = "Unknown account '{}'. Valid accounts: globalbank, techstartup, retailco"


def _query_by_account(table: str, account_id: str, label: str, *, single=False, order=None, only_open=False):
    slug = normalise_account(account_id)
    if not slug:
        return UNKNOWN_ACCOUNT_MSG.format(account_id)
    try:
        q = get_supabase().table(table).select("*").eq("account_id", slug)
        if only_open:
            q = q.eq("status", "open")
        if order:
            q = q.order(order)
        res = q.execute()
        if not res.data:
            return f"No {label} found for '{slug}'"
        return json.dumps(res.data[0] if single else res.data)
    except Exception as e:
        return f"Error fetching {label}: {e}"


@tool
def get_account_info(account_id: str) -> str:
    """Get core CRM data for an account: health score, MRR, ARR, last login, open tickets, CSM owner, industry, tier, contract end date."""
    slug = normalise_account(account_id)
    if not slug:
        return UNKNOWN_ACCOUNT_MSG.format(account_id)
    try:
        res = get_supabase().table("accounts").select("*").eq("id", slug).execute()
        if not res.data:
            return f"No account found with id '{slug}'"
        return json.dumps(res.data[0])
    except Exception as e:
        return f"Error fetching account info: {e}"


@tool
def get_open_tickets(account_id: str) -> str:
    """Get all open support tickets for an account including title, priority and how many days ago they were created."""
    return _query_by_account("tickets", account_id, "open tickets", only_open=True)


@tool
def get_interaction_history(account_id: str) -> str:
    """Get recent interaction history for an account: calls, emails, QBRs — including summaries and next touchpoint timing."""
    return _query_by_account("interactions", account_id, "interactions", order="days_ago")


@tool
def get_health_trend(account_id: str) -> str:
    """Get the 8-week health score trend for an account to identify if health is improving, declining or stable."""
    return _query_by_account("health_signals", account_id, "health trend data", order="week_number")


@tool
def get_feature_usage(account_id: str) -> str:
    """Get feature adoption data for an account: which features are used daily/weekly/never and adoption scores."""
    return _query_by_account("feature_usage", account_id, "feature usage data")


@tool
def get_renewal_info(account_id: str) -> str:
    """Get renewal and commercial data for an account: renewal date, renewal probability, discount applied, upsell opportunity and value."""
    return _query_by_account("renewals", account_id, "renewal data", single=True)


@tool
def list_all_accounts() -> str:
    """List all accounts in the portfolio with their core health metrics for portfolio-level overview queries."""
    try:
        res = get_supabase().table("accounts").select("*").execute()
        if not res.data:
            return "No accounts found"
        return json.dumps(res.data)
    except Exception as e:
        return f"Error listing accounts: {e}"


@tool
def get_playbook(query: str) -> str:
    """Semantic search over the churn-prevention playbooks. Describe the situation (e.g. 'usage dropped 60% and NPS is low', 'prepare a QBR', 'when to escalate') and get the 2 most relevant playbooks with similarity scores. Always call this before recommending actions."""
    try:
        hits = RETRIEVER.search(query, k=2)
        return json.dumps([
            {"playbook": h["title"], "relevance": h["score"], "steps": h["text"]} for h in hits
        ])
    except Exception as e:
        return f"Error retrieving playbook: {e}"


TOOLS = [
    get_account_info,
    get_open_tickets,
    get_interaction_history,
    get_health_trend,
    get_feature_usage,
    get_renewal_info,
    list_all_accounts,
    get_playbook,
]

TOOL_MAP = {t.name: t for t in TOOLS}


# ── PROMPT VARIANTS ────────────────────────────────────────────────────────────

PLAYBOOK_INSTRUCTION = f"""PLAYBOOKS: the playbook library is retrieved on demand. Call get_playbook with a short
description of the account's situation BEFORE recommending actions, and base your actions on the
steps it returns. Cite the playbook name it returns. Playbooks available:
{PLAYBOOK_TITLES}"""

# Variant A — Minimal baseline prompt
PROMPT_A = """You are a customer success assistant. Answer questions about customer accounts."""

# Variant B — Structured role prompt (submitted version)
PROMPT_B = f"""You are ChurnSignal, an AI Customer Health & Churn Prevention Copilot for Customer Success Managers (CSMs).

You are a READ-ONLY decision support agent. You NEVER send emails, update CRMs, approve discounts, or contact customers autonomously. If asked to perform any action, refuse clearly and offer to prepare the content instead. Drafting an email for the CSM to review and send is allowed.

You have access to live Supabase tools to fetch real account data. Use them to answer questions accurately. Always fetch data before making recommendations — do not rely on memory. When drafting emails, fetch the account's data first and never use placeholder names.

Available accounts: globalbank, techstartup, retailco

{PLAYBOOK_INSTRUCTION}

RESPONSE FORMAT for health checks and recommendations:
**RISK LEVEL:** [CRITICAL / HIGH / MEDIUM / LOW]

**ASSESSMENT:** 2-3 clear, data-driven sentences about account health.

**KEY RISK SIGNALS:**
- Signal 1 (with specific number/data point)
- Signal 2
- Signal 3

**PLAYBOOK RECOMMENDATION:** Name the specific playbook

**RECOMMENDED ACTIONS FOR CSM:**
1. First action (specific and time-bound)
2. Second action
3. Third action

**ESCALATION FLAG:** YES / NO — with brief reason

For portfolio queries, summarise all accounts ranked by urgency.
For QBR prep, provide a detailed agenda and talking points.
For general questions, answer concisely and helpfully.
Be direct, professional, and specific — avoid vague advice."""

# Variant C — Chain-of-thought prompt with entity resolution
PROMPT_C = f"""You are ChurnSignal, an AI Customer Success assistant for a B2B SaaS company.
Your users are Senior Customer Success Managers (CSMs) managing a portfolio of enterprise accounts.

Before responding to any query, follow these steps internally:

STEP 1 — Identify the account(s) being discussed.
  If a pronoun is used ("them", "they", "it", "that account"), resolve it using the
  [Entity register] note in the message (last explicitly named account). If there is no note and
  it is still ambiguous, ask the user to clarify before proceeding — do not guess.

STEP 2 — Classify the intent:
  [risk_triage | health_report | email_draft | data_lookup | advisory_action | write_request | other]
  advisory_action = the CSM asks what THEY should do ("what should I do about them?", "what do you
  recommend?"). This is a read-only advisory request, NOT a write request.
  write_request = the CSM asks YOU to change, mark, update, delete or send something in a system.

STEP 3 — If intent = write_request, refuse immediately and clearly. Do not proceed further.
  Otherwise, fetch live data using your tools and respond.

Response guidelines by intent:
  - risk_triage: rank ALL accounts by urgency. Cite health score, last login days, and open
    ticket count per account. Never omit an account.
  - health_report: include health score trend (current vs prior), escalation flag (YES/NO),
    and exactly 3 specific playbook actions with timeframes.
  - email_draft: personalise every sentence with real account metrics. Never use generic
    templates. Reference specific data points (login gap, health score, ticket count).
  - data_lookup: respond in one concise sentence. Do not add unsolicited risk commentary.
  - advisory_action: fetch the account's data, then give the playbook-based next steps.
  - write_request: refuse clearly. Explain you are a Decision Support agent only.

{PLAYBOOK_INSTRUCTION}

HARD CONSTRAINTS:
  - Decision Support ONLY. You cannot write to, update, or modify any system, ever.
  - Never fabricate data. Every claim must come from retrieved account records.
  - If uncertain about which account is being referenced, ask for clarification.
  - Always fetch fresh data — never rely on memory or prior responses for account metrics."""

PROMPT_MAP = {"A": PROMPT_A, "B": PROMPT_B, "C": PROMPT_C}


# ── SAFETY GUARDRAIL (layer 1, pre-LLM) ───────────────────────────────────────
# Intent-based: a write verb near a data target, plus explicit action phrases.
# Drafting an email is allowed; sending one is not.

WRITE_VERBS = (
    r"update|delete|remove|modify|edit|overwrite|change|insert|drop|truncate|"
    r"rename|reassign|cancel|terminate|mark|flag|log"
)
WRITE_TARGETS = (
    r"record|crm|renewal|date|score|status|database|system|account|ticket|tier|"
    r"owner|arr|mrr|contract|subscription|discount|renewed|churned|closed|health"
)

WRITE_PATTERNS = [
    re.compile(rf"\b({WRITE_VERBS})\b[^.?!]{{0,40}}\b({WRITE_TARGETS})\b"),
    re.compile(r"\bmark\b[^.?!]{0,60}\bas\b"),
    re.compile(r"\bset\b[^.?!]{0,40}\bto\b"),
    re.compile(r"\b(send|email|contact)\b\s+(it|this|that|the (email|customer|client)|them now)\b"),
    re.compile(r"\bsend (an? )?(email|message|mail)\b"),
    re.compile(r"\b(approve|give|offer|apply)\b[^.?!]{0,20}\bdiscount\b"),
    re.compile(r"\b(schedule|book)\b[^.?!]{0,20}\b(call|meeting|qbr)\b[^.?!]{0,20}\b(for me|in the calendar)\b"),
    re.compile(r"\b(contact|call|email) (globalbank|techstartup|retailco|the customer)\b"),
]

# Advisory phrasing the CSM uses about their OWN actions is never a write request.
ADVISORY_PATTERN = re.compile(r"\b(what should i|what do you recommend|how should i|should i)\b")


def is_write_request(message: str) -> bool:
    text = message.lower()
    if ADVISORY_PATTERN.search(text) and not re.search(r"\b(mark|update|delete|set)\b", text):
        return False
    return any(p.search(text) for p in WRITE_PATTERNS)


REFUSAL_TEXT = (
    "❌ **Action not permitted** — I am a read-only decision support agent.\n\n"
    "I cannot send emails, update CRM records, approve discounts, or contact customers directly.\n\n"
    "What I *can* do: draft an email for you to review and send, prepare talking points, "
    "draft a QBR agenda, summarise account health, or recommend the right playbook for you to act on.\n\n"
    "Would you like me to prepare any of these instead?"
)


# ── FEEDBACK STORE ────────────────────────────────────────────────────────────
# Explicit signal: thumbs up/down (+ optional correction text) from the CSM.
# Implicit signal: a blocked request or a clarification re-ask is logged as an event.
# Corrections are applied to later answers in the same session.

FEEDBACK_LOG = deque(maxlen=1000)
SESSION_CORRECTIONS: dict = {}  # session_id -> list[str]


def corrections_notice(session_id: Optional[str]) -> str:
    notes = SESSION_CORRECTIONS.get(session_id or "", [])[-3:]
    if not notes:
        return ""
    joined = "\n".join(f"- {n}" for n in notes)
    return (
        "\n\nCSM FEEDBACK EARLIER IN THIS SESSION (apply it to your answer, "
        "do not repeat the mistakes described):\n" + joined
    )


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def warm_up_rag():
    """Load the embedding model in the background so the first chat is not slow."""
    import threading
    threading.Thread(target=RETRIEVER._init, daemon=True).start()


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ChurnSignal API", "version": "4.0.0"}


@app.get("/api/metrics")
def metrics():
    lat = list(LATENCIES)
    total = METRICS["requests"] or 1
    return {
        **METRICS,
        "block_rate": round(METRICS["blocked_keyword"] / total, 3),
        "error_rate": round(METRICS["errors"] / total, 3),
        "latency_ms": {
            "samples": len(lat),
            "avg": round(sum(lat) / len(lat), 1) if lat else None,
            "p50": percentile(lat, 50),
            "p95": percentile(lat, 95),
            "max": round(max(lat), 1) if lat else None,
        },
        "rag_mode": RETRIEVER.mode,
    }


@app.get("/api/retrieve")
def retrieve(q: str, k: int = 2):
    """Debug endpoint for the evaluation report: shows what RAG retrieves for a query."""
    hits = RETRIEVER.search(q, k=max(1, min(k, 5)))
    return {
        "query": q,
        "mode": RETRIEVER.mode,
        "results": [{"playbook": h["title"], "score": h["score"]} for h in hits],
    }


@app.get("/api/accounts")
def list_accounts():
    try:
        res = get_supabase().table("accounts").select("*").execute()
        return [
            {
                **acc,
                "risk": risk_from_score(acc["health_score"]),
                "arr_formatted": f"${acc['arr']:,}",
                "logo_initial": acc["name"][:2].upper(),
            }
            for acc in res.data
        ]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/accounts/{account_id}")
def get_account(account_id: str):
    slug = normalise_account(account_id)
    if not slug:
        return {"error": "Account not found"}
    try:
        res = get_supabase().table("accounts").select("*").eq("id", slug).execute()
        if not res.data:
            return {"error": "Account not found"}
        acc = res.data[0]
        return {**acc, "risk": risk_from_score(acc["health_score"]), "arr_formatted": f"${acc['arr']:,}"}
    except Exception as e:
        return {"error": str(e)}


class FeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    rating: str                      # "up" | "down"
    variant: Optional[str] = "B"
    account_id: Optional[str] = None
    correction: Optional[str] = None  # e.g. "I meant RetailCo, not GlobalBank"


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    rating = "down" if req.rating.lower() in ("down", "thumbs_down", "-1") else "up"
    METRICS[f"feedback_{rating}"] += 1
    correction = (req.correction or "").strip()[:300]
    FEEDBACK_LOG.append({"ts": time.time(), "rating": rating, "variant": req.variant,
                         "has_correction": bool(correction)})
    if rating == "down" and correction and req.session_id:
        SESSION_CORRECTIONS.setdefault(req.session_id, []).append(correction)
    log_event("feedback", rating=rating, variant=req.variant,
              account=mask(req.account_id), has_correction=bool(correction))
    return {"ok": True, "applied_to_session": bool(correction and rating == "down" and req.session_id)}


# ── CHAT WITH TOOL CALLING ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    account_id: Optional[str] = None
    history: Optional[List[dict]] = []
    variant: Optional[str] = "B"      # "A", "B", or "C"
    session_id: Optional[str] = None  # lets feedback corrections reach later turns


MAX_ITERATIONS = 5
ESCALATION_PATTERN = re.compile(r"ESCALATION FLAG[\s\*:]*YES|ESCALATION REQUIRED")


def run_agent(req: ChatRequest, variant: str, trace: dict) -> str:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.3,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    system_text = PROMPT_MAP.get(variant, PROMPT_B) + corrections_notice(req.session_id)
    messages = [SystemMessage(content=system_text)]

    history = req.history or []
    for h in history[-6:]:
        role, content = h.get("role"), h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Variant A is the untouched baseline: no entity register.
    user_message = req.message
    if variant != "A":
        reg = build_entity_register(req.message, history, req.account_id)
        trace["entity"] = reg["last_named"]
        trace["conflict"] = bool(reg["conflict"])
        notice = register_notice(reg)
        if notice:
            user_message = f"{notice}\n{req.message}"
    elif req.account_id:
        user_message = f"[Account context: {req.account_id}]\n{req.message}"
    messages.append(HumanMessage(content=user_message))

    response = None
    for i in range(MAX_ITERATIONS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        trace["iterations"] = i + 1
        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            name = tc["name"]
            trace["tools"].append(name)
            tool_fn = TOOL_MAP.get(name)
            if not tool_fn:
                result = f"Unknown tool: {name}"
            else:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:  # one bad tool call must not kill the turn
                    result = f"Tool {name} failed: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # If the loop ended on tool calls (or empty text), force a final answer with no tools.
    if response is None or response.tool_calls or not (response.content or "").strip():
        trace["forced_final"] = True
        messages.append(HumanMessage(content="Using the data retrieved so far, give your final answer now."))
        response = llm.invoke(messages)

    return response.content or "I could not produce an answer. Please rephrase your question."


@app.post("/api/chat")
def chat(req: ChatRequest):
    started = time.time()
    variant = (req.variant or "B").upper()
    if variant not in PROMPT_MAP:
        variant = "B"

    METRICS["requests"] += 1
    METRICS["by_variant"][variant] += 1
    trace = {"tools": [], "iterations": 0}
    base_log = {"variant": variant, "account": mask(req.account_id), "session": mask(req.session_id)}

    # Layer 1 — guardrail fires before the LLM (A is the unguarded baseline by design)
    if variant != "A" and is_write_request(req.message):
        METRICS["blocked_keyword"] += 1
        latency = (time.time() - started) * 1000
        LATENCIES.append(latency)
        log_event("chat", **base_log, outcome="blocked", layer="keyword", latency_ms=round(latency, 1))
        return {
            "response": REFUSAL_TEXT,
            "escalation": False,
            "blocked": True,
            "variant": variant,
            "meta": {"layer": "keyword", "latency_ms": round(latency, 1), "tools": []},
        }

    try:
        final_text = run_agent(req, variant, trace)
        escalation = bool(ESCALATION_PATTERN.search(final_text.upper()))
        if escalation:
            METRICS["escalations"] += 1
        latency = (time.time() - started) * 1000
        LATENCIES.append(latency)
        log_event("chat", **base_log, outcome="ok", escalation=escalation,
                  latency_ms=round(latency, 1), iterations=trace["iterations"], tools=trace["tools"],
                  entity=mask(trace.get("entity")), conflict=trace.get("conflict", False),
                  forced_final=trace.get("forced_final", False),
                  corrections_applied=len(SESSION_CORRECTIONS.get(req.session_id or "", [])))
        return {
            "response": final_text,
            "escalation": escalation,
            "blocked": False,
            "variant": variant,
            "meta": {"latency_ms": round(latency, 1), "tools": trace["tools"],
                     "iterations": trace["iterations"]},
        }

    except Exception as e:
        METRICS["errors"] += 1
        latency = (time.time() - started) * 1000
        LATENCIES.append(latency)
        log_event("chat", **base_log, outcome="error", error=type(e).__name__,
                  latency_ms=round(latency, 1), tools=trace["tools"])
        return {
            "response": f"⚠️ Agent error: {e}\n\nPlease check that GROQ_API_KEY, SUPABASE_URL and SUPABASE_SERVICE_KEY are set in Render environment variables.",
            "escalation": False,
            "blocked": False,
            "variant": variant,
            "meta": {"latency_ms": round(latency, 1), "tools": trace["tools"]},
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
