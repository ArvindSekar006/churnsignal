"""
ChurnSignal — FastAPI Backend
AI Customer Health & Churn Prevention Copilot
Arvind Sekar | Capstone Project
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(title="ChurnSignal API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ACCOUNTS DATA ─────────────────────────────────────────────────────────────
ACCOUNTS = {
    "globalbank": {
        "id": "globalbank",
        "name": "GlobalBank",
        "domain": "globalbank.com",
        "industry": "Financial Services",
        "plan": "Starter",
        "arr": 50000,
        "users": 2000,
        "open_tickets": 156,
        "usage_drop_pct": 60,
        "nps_score": 4,
        "days_since_login": 12,
        "csm": "Arvind Sekar",
        "renewal_date": "2027-03-31",
        "logo_initial": "GB",
        "health_trend": "declining",
    },
    "techflow": {
        "id": "techflow",
        "name": "TechFlow",
        "domain": "techflow.io",
        "industry": "SaaS / Tech",
        "plan": "Pro",
        "arr": 9000,
        "users": 200,
        "open_tickets": 12,
        "usage_drop_pct": 5,
        "nps_score": 9,
        "days_since_login": 1,
        "csm": "Arvind Sekar",
        "renewal_date": "2027-06-30",
        "logo_initial": "TF",
        "health_trend": "stable",
    },
    "acmecorp": {
        "id": "acmecorp",
        "name": "Acme Corp",
        "domain": "acmecorp.com",
        "industry": "Banking",
        "plan": "Starter",
        "arr": 12500,
        "users": 500,
        "open_tickets": 47,
        "usage_drop_pct": 25,
        "nps_score": 6,
        "days_since_login": 7,
        "csm": "Arvind Sekar",
        "renewal_date": "2026-12-31",
        "logo_initial": "AC",
        "health_trend": "at risk",
    },
}

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
- CRITICAL risk + contract value ≥ $50K → VP Customer Success within 24 hours
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


def compute_risk(acc: dict) -> str:
    if acc["usage_drop_pct"] > 50 or acc["open_tickets"] > 100 or acc["nps_score"] < 5:
        return "CRITICAL"
    elif acc["usage_drop_pct"] > 25 or acc["open_tickets"] > 30 or acc["nps_score"] < 7:
        return "HIGH"
    elif acc["usage_drop_pct"] > 10 or acc["open_tickets"] > 10 or acc["nps_score"] < 8:
        return "MEDIUM"
    return "LOW"


# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ChurnSignal API"}


@app.get("/api/accounts")
def list_accounts():
    result = []
    for acc in ACCOUNTS.values():
        result.append({
            **acc,
            "risk": compute_risk(acc),
            "arr_formatted": f"${acc['arr']:,}",
        })
    return result


@app.get("/api/accounts/{account_id}")
def get_account(account_id: str):
    acc = ACCOUNTS.get(account_id.lower())
    if not acc:
        return {"error": "Account not found"}
    return {**acc, "risk": compute_risk(acc), "arr_formatted": f"${acc['arr']:,}"}


# ── CHAT ───────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    account_id: Optional[str] = None
    history: Optional[List[dict]] = []


SAFETY_PHRASES = [
    "send email", "send an email", "draft email",
    "update crm", "update the crm", "log in crm",
    "approve discount", "give discount", "offer discount",
    "schedule a call", "book a call", "book meeting",
    "contact the customer", "contact globalbank", "contact acme", "contact techflow",
    "modify", "delete account", "remove account",
]


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.3,
        )

        msg_lower = req.message.lower()
        if any(phrase in msg_lower for phrase in SAFETY_PHRASES):
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
            }

        # Build account context
        account_context = ""
        if req.account_id:
            acc = ACCOUNTS.get(req.account_id.lower())
            if acc:
                risk = compute_risk(acc)
                account_context = f"""
SELECTED ACCOUNT:
- Name: {acc['name']} ({acc['domain']})
- Industry: {acc['industry']} | Plan: {acc['plan']} | Users: {acc['users']:,}
- ARR: ${acc['arr']:,} | Renewal: {acc['renewal_date']}
- Risk Level: {risk}
- Open Tickets: {acc['open_tickets']} | Usage Drop: {acc['usage_drop_pct']}%
- NPS Score: {acc['nps_score']} | Days Since Login: {acc['days_since_login']}
- Health Trend: {acc['health_trend'].title()}
- CSM: {acc['csm']}
"""

        # Also include all accounts for portfolio queries
        portfolio_context = "PORTFOLIO OVERVIEW:\n"
        for a in ACCOUNTS.values():
            r = compute_risk(a)
            portfolio_context += f"- {a['name']}: {r} risk | ARR ${a['arr']:,} | Tickets {a['open_tickets']} | Usage drop {a['usage_drop_pct']}% | NPS {a['nps_score']}\n"

        system_prompt = f"""You are ChurnSignal, an AI Customer Health & Churn Prevention Copilot for Customer Success Managers (CSMs).

You are a READ-ONLY decision support agent. You NEVER send emails, update CRMs, approve discounts, or contact customers autonomously. If asked to perform any action, refuse clearly and offer to prepare the content instead.

{account_context if account_context else ""}

{portfolio_context}

CHURN PREVENTION PLAYBOOKS:
{PLAYBOOKS}

RESPONSE FORMAT (use this structure for health checks and recommendations):
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

        messages = [{"role": "system", "content": system_prompt}]

        # Include recent history (last 6 messages = 3 turns)
        for h in (req.history or [])[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": req.message})

        response = llm.invoke(messages)
        text = response.content

        escalation = any(kw in text.upper() for kw in [
            "ESCALATION FLAG: YES",
            "ESCALATION REQUIRED",
            "VP CUSTOMER SUCCESS",
            "CRITICAL RISK",
        ])

        return {
            "response": text,
            "escalation": escalation,
            "blocked": False,
        }

    except Exception as e:
        return {
            "response": f"⚠️ Agent error: {str(e)}\n\nPlease ensure GROQ_API_KEY is set and the API is reachable.",
            "escalation": False,
            "blocked": False,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
