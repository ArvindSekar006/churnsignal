"""
ChurnSignal — FastAPI Backend
AI Customer Health & Churn Prevention Copilot
Arvind Sekar | Capstone Project

v3: Prompt variant support (A / B / C) for evaluation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json

app = FastAPI(title="ChurnSignal API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SUPABASE CLIENT ────────────────────────────────────────────────────────────

def get_supabase():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    return create_client(url, key)


# ── LANGCHAIN TOOLS ────────────────────────────────────────────────────────────

from langchain_core.tools import tool

@tool
def get_account_info(account_id: str) -> str:
    """Get core CRM data for an account: health score, MRR, ARR, last login, open tickets, CSM owner, industry, tier, contract end date."""
    try:
        sb = get_supabase()
        res = sb.table("accounts").select("*").eq("id", account_id.lower()).execute()
        if not res.data:
            return f"No account found with id '{account_id}'. Available: globalbank, techstartup, retailco"
        return json.dumps(res.data[0])
    except Exception as e:
        return f"Error fetching account info: {e}"


@tool
def get_open_tickets(account_id: str) -> str:
    """Get all open support tickets for an account including title, priority and how many days ago they were created."""
    try:
        sb = get_supabase()
        res = sb.table("tickets").select("*").eq("account_id", account_id.lower()).eq("status", "open").execute()
        if not res.data:
            return f"No open tickets found for '{account_id}'"
        return json.dumps(res.data)
    except Exception as e:
        return f"Error fetching tickets: {e}"


@tool
def get_interaction_history(account_id: str) -> str:
    """Get recent interaction history for an account: calls, emails, QBRs — including summaries and next touchpoint timing."""
    try:
        sb = get_supabase()
        res = sb.table("interactions").select("*").eq("account_id", account_id.lower()).order("days_ago").execute()
        if not res.data:
            return f"No interactions found for '{account_id}'"
        return json.dumps(res.data)
    except Exception as e:
        return f"Error fetching interactions: {e}"


@tool
def get_health_trend(account_id: str) -> str:
    """Get the 8-week health score trend for an account to identify if health is improving, declining or stable."""
    try:
        sb = get_supabase()
        res = sb.table("health_signals").select("*").eq("account_id", account_id.lower()).order("week_number").execute()
        if not res.data:
            return f"No health trend data found for '{account_id}'"
        return json.dumps(res.data)
    except Exception as e:
        return f"Error fetching health trend: {e}"


@tool
def get_feature_usage(account_id: str) -> str:
    """Get feature adoption data for an account: which features are used daily/weekly/never and adoption scores."""
    try:
        sb = get_supabase()
        res = sb.table("feature_usage").select("*").eq("account_id", account_id.lower()).execute()
        if not res.data:
            return f"No feature usage data found for '{account_id}'"
        return json.dumps(res.data)
    except Exception as e:
        return f"Error fetching feature usage: {e}"


@tool
def get_renewal_info(account_id: str) -> str:
    """Get renewal and commercial data for an account: renewal date, renewal probability, discount applied, upsell opportunity and value."""
    try:
        sb = get_supabase()
        res = sb.table("renewals").select("*").eq("account_id", account_id.lower()).execute()
        if not res.data:
            return f"No renewal data found for '{account_id}'"
        return json.dumps(res.data[0])
    except Exception as e:
        return f"Error fetching renewal info: {e}"


@tool
def list_all_accounts() -> str:
    """List all accounts in the portfolio with their core health metrics for portfolio-level overview queries."""
    try:
        sb = get_supabase()
        res = sb.table("accounts").select("*").execute()
        if not res.data:
            return "No accounts found"
        return json.dumps(res.data)
    except Exception as e:
        return f"Error listing accounts: {e}"


TOOLS = [
    get_account_info,
    get_open_tickets,
    get_interaction_history,
    get_health_trend,
    get_feature_usage,
    get_renewal_info,
    list_all_accounts,
]

TOOL_MAP = {t.name: t for t in TOOLS}


# ── PLAYBOOKS ─────────────────────────────────────────────────────────────────

PLAYBOOKS = """
CRITICAL RISK PLAYBOOK — High Ticket Volume + Low Engagement:
- Schedule emergency QBR within 48 hours
- Involve executive sponsor on both sides
- Offer complimentary Success Workshop to re-onboard the team
- Weekly cadence calls for minimum 4 weeks
- Escalate to VP Customer Success if no improvement in 2 weeks

CRITICAL RISK PLAYBOOK — Usage Drop + Low NPS:
- Schedule discovery call within 24 hours to understand pain points
- Review product adoption metrics — feature usage, login frequency, workflow completion
- Propose 30-day success milestone plan with personalized onboarding
- Assign dedicated CSM for 60 days

HIGH RISK PLAYBOOK — Elevated Ticket Volume + Moderate Disengagement:
- Schedule QBR within 2 weeks
- Conduct ticket clustering analysis to identify recurring issue categories
- Share product roadmap updates relevant to their pain points
- Set up health score alerts for weekly monitoring

QBR PREPARATION GUIDE:
- Review success metrics since last QBR
- Open ticket audit — categorise by age, severity, owner
- Usage and adoption trends — feature heatmap, login frequency
- Product roadmap items relevant to their stated pain points
- Competitive threats — check if competitors mentioned in tickets
- Joint success plan with 3 specific commitments on both sides

ESCALATION CRITERIA:
- CRITICAL risk + contract value >= $50K → VP Customer Success within 24 hours
- Competitor mentioned in ticket or call → Immediate Retention Team involvement
- Renewal within 60 days + HIGH or CRITICAL risk → Involve Account Executive
- No login for >14 days + any risk level → CSM outreach required same day

MEDIUM RISK PLAYBOOK:
- Proactive check-in within 3 business days
- Share relevant new features and product updates
- Offer optional training session or webinar
- Monthly health review

LOW RISK PLAYBOOK:
- Quarterly health check-in
- Share product newsletter and roadmap updates
- Consider for reference customer or case study program
"""

# ── PROMPT VARIANTS ────────────────────────────────────────────────────────────

# Variant A — Minimal baseline prompt
PROMPT_A = """You are a customer success assistant. Answer questions about customer accounts."""

# Variant B — Structured role prompt (original / submitted version)
PROMPT_B = f"""You are ChurnSignal, an AI Customer Health & Churn Prevention Copilot for Customer Success Managers (CSMs).

You are a READ-ONLY decision support agent. You NEVER send emails, update CRMs, approve discounts, or contact customers autonomously. If asked to perform any action, refuse clearly and offer to prepare the content instead.

You have access to live Supabase tools to fetch real account data. Use them to answer questions accurately. Always fetch data before making recommendations — do not rely on memory.

Available accounts: globalbank, techstartup, retailco

CHURN PREVENTION PLAYBOOKS:
{PLAYBOOKS}

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
  If a pronoun is used ("them", "they", "it", "that account"), resolve it to the most recently
  explicitly named account in the conversation history. If still ambiguous, ask the user to
  clarify before proceeding — do not guess.

STEP 2 — Classify the intent:
  [risk_triage | health_report | email_draft | data_lookup | write_request | other]

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
  - write_request: refuse clearly. Explain you are a Decision Support agent only.

CHURN PREVENTION PLAYBOOKS:
{PLAYBOOKS}

HARD CONSTRAINTS:
  - Decision Support ONLY. You cannot write to, update, or modify any system, ever.
  - Never fabricate data. Every claim must come from retrieved account records.
  - If uncertain about which account is being referenced, ask for clarification.
  - Always fetch fresh data — never rely on memory or prior responses for account metrics."""

PROMPT_MAP = {
    "A": PROMPT_A,
    "B": PROMPT_B,
    "C": PROMPT_C,
}

# Default is B (the submitted version)
SYSTEM_PROMPT = PROMPT_B


# ── SAFETY GUARDRAIL ─────────────────────────────────────────────────────────

SAFETY_PHRASES = [
    "send email", "send an email", "draft email",
    "update crm", "update the crm", "log in crm",
    "approve discount", "give discount", "offer discount",
    "schedule a call", "book a call", "book meeting",
    "contact the customer", "contact globalbank", "contact techstartup", "contact retailco",
    "modify", "delete account", "remove account",
    "cancel subscription", "terminate contract",
]


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ChurnSignal API", "version": "3.0.0"}


@app.get("/api/accounts")
def list_accounts():
    try:
        sb = get_supabase()
        res = sb.table("accounts").select("*").execute()
        result = []
        for acc in res.data:
            hs = acc["health_score"]
            if hs < 30:
                risk = "CRITICAL"
            elif hs < 50:
                risk = "HIGH"
            elif hs < 70:
                risk = "MEDIUM"
            else:
                risk = "LOW"
            result.append({
                **acc,
                "risk": risk,
                "arr_formatted": f"${acc['arr']:,}",
                "logo_initial": acc["name"][:2].upper(),
            })
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/accounts/{account_id}")
def get_account(account_id: str):
    try:
        sb = get_supabase()
        res = sb.table("accounts").select("*").eq("id", account_id.lower()).execute()
        if not res.data:
            return {"error": "Account not found"}
        acc = res.data[0]
        hs = acc["health_score"]
        if hs < 30:
            risk = "CRITICAL"
        elif hs < 50:
            risk = "HIGH"
        elif hs < 70:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        return {**acc, "risk": risk, "arr_formatted": f"${acc['arr']:,}"}
    except Exception as e:
        return {"error": str(e)}


# ── CHAT WITH TOOL CALLING ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    account_id: Optional[str] = None
    history: Optional[List[dict]] = []
    variant: Optional[str] = "B"   # "A", "B", or "C"


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.3,
        )

        # Select system prompt based on variant (default B)
        variant = (req.variant or "B").upper()
        selected_prompt = PROMPT_MAP.get(variant, PROMPT_B)

        # Safety guardrail — fires before LLM (only enforce for B and C; A has no guardrail by design)
        msg_lower = req.message.lower()
        if variant != "A" and any(phrase in msg_lower for phrase in SAFETY_PHRASES):
            return {
                "response": (
                    "❌ **Action not permitted** — I am a read-only decision support agent.\n\n"
                    "I cannot send emails, update CRM records, approve discounts, or contact customers directly.\n\n"
                    "What I *can* do: prepare talking points, draft a QBR agenda, summarise account health, "
                    "or recommend the right playbook for you to act on.\n\n"
                    "Would you like me to prepare any of these instead?"
                ),
                "escalation": False,
                "blocked": True,
                "variant": variant,
            }

        # Build message list
        llm_with_tools = llm.bind_tools(TOOLS)

        messages = [SystemMessage(content=selected_prompt)]

        # Add conversation history
        for h in (req.history or [])[-6:]:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                messages.append(AIMessage(content=h["content"]))

        # If account_id provided, hint to the LLM
        user_message = req.message
        if req.account_id:
            user_message = f"[Account context: {req.account_id}]\n{req.message}"

        messages.append(HumanMessage(content=user_message))

        # Agentic loop — LLM calls tools until it has enough data
        max_iterations = 5
        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # If no tool calls, we have the final answer
            if not response.tool_calls:
                break

            # Execute each tool call
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_fn = TOOL_MAP.get(tool_name)
                if tool_fn:
                    tool_result = tool_fn.invoke(tool_args)
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"]
                ))

        final_text = response.content

        escalation = any(kw in final_text.upper() for kw in [
            "ESCALATION FLAG: YES",
            "ESCALATION REQUIRED",
            "VP CUSTOMER SUCCESS",
            "CRITICAL RISK",
        ])

        return {
            "response": final_text,
            "escalation": escalation,
            "blocked": False,
            "variant": variant,
        }

    except Exception as e:
        return {
            "response": f"⚠️ Agent error: {str(e)}\n\nPlease check that GROQ_API_KEY, SUPABASE_URL and SUPABASE_SERVICE_KEY are set in Render environment variables.",
            "escalation": False,
            "blocked": False,
            "variant": req.variant or "B",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
