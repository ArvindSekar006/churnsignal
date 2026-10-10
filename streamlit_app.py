import streamlit as st
import json
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prevention Copilot",
    page_icon="🤖",
    layout="wide"
)

# ── ACCOUNTS DATA ─────────────────────────────────────────────
ACCOUNTS = {
    "GlobalBank": {
        "industry": "Finance", "users": 2000, "plan": "Starter",
        "open_tickets": 156, "usage_drop_pct": 60, "nps_score": 4,
        "days_since_login": 12, "contract_value": 50000
    },
    "TechFlow": {
        "industry": "SaaS", "users": 200, "plan": "Pro",
        "open_tickets": 12, "usage_drop_pct": 5, "nps_score": 9,
        "days_since_login": 1, "contract_value": 9000
    },
    "Acme Corp": {
        "industry": "Banking", "users": 500, "plan": "Starter",
        "open_tickets": 47, "usage_drop_pct": 25, "nps_score": 6,
        "days_since_login": 7, "contract_value": 12500
    }
}

# ── RISK COLOURS ──────────────────────────────────────────────
RISK_COLORS = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "🟢"
}

def get_risk_level(account_name):
    acc = ACCOUNTS[account_name]
    if acc["usage_drop_pct"] > 50 or acc["open_tickets"] > 100 or acc["nps_score"] < 5:
        return "CRITICAL"
    elif acc["usage_drop_pct"] > 25 or acc["open_tickets"] > 30 or acc["nps_score"] < 7:
        return "HIGH"
    elif acc["usage_drop_pct"] > 10 or acc["open_tickets"] > 10:
        return "MEDIUM"
    return "LOW"

# ── INITIALISE LLM ────────────────────────────────────────────
@st.cache_resource
def init_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.environ.get("GROQ_API_KEY")
    )

# ── INITIALISE RAG ────────────────────────────────────────────
@st.cache_resource
def init_rag():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    playbooks = [
        Document(page_content="""CRITICAL RISK PLAYBOOK — High Ticket Volume + Low Engagement
Trigger: Open tickets > 100 AND usage drop > 50% AND NPS < 5
Actions: Schedule emergency QBR within 48 hours. Involve executive sponsor on both sides.
Offer complimentary Success Workshop. Create joint action plan. Weekly check-ins for 4 weeks.
Escalate to VP Customer Success if no improvement after 2 weeks.""",
        metadata={"source": "critical_high_tickets"}),

        Document(page_content="""CRITICAL RISK PLAYBOOK — Usage Drop + Low NPS
Trigger: Usage drop > 50% AND NPS < 6 AND days since login > 10
Actions: Discovery call within 24 hours. Review product adoption metrics.
Offer personalised onboarding. Share relevant case studies.
Propose success milestones within 30 days.""",
        metadata={"source": "critical_usage_drop"}),

        Document(page_content="""HIGH RISK PLAYBOOK — Elevated Ticket Volume
Trigger: Open tickets 30-100 OR usage drop 25-50% OR NPS 5-7
Actions: Schedule QBR within 2 weeks. Conduct ticket clustering analysis.
Share product roadmap. Offer training session. Set up health score alerts.""",
        metadata={"source": "high_risk"}),

        Document(page_content="""QBR PREPARATION GUIDE — High Risk Accounts
Agenda: Review success metrics. Open ticket review with owners and dates.
Usage and adoption review. Product roadmap relevant to their use case.
Joint action plan with specific items. Never go into QBR with bad news and no plan.""",
        metadata={"source": "qbr_guide"}),

        Document(page_content="""ESCALATION CRITERIA
Escalate to VP CS: CRITICAL risk + no improvement after 2 weeks.
Contract value > $50K AND risk HIGH or CRITICAL.
Customer mentions competitor or switching intent.
Escalate to AE: Pricing concerns, discount requests, renewal within 60 days.""",
        metadata={"source": "escalation_criteria"}),

        Document(page_content="""MEDIUM RISK PLAYBOOK
Trigger: Usage drop 10-25% OR tickets 10-30 OR NPS 7-8
Actions: Proactive check-in email within 3 days. Share new feature updates.
Offer optional training. Monitor weekly for degradation.""",
        metadata={"source": "medium_risk"}),
    ]
    vectorstore = FAISS.from_documents(playbooks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# ── AGENT FUNCTION ────────────────────────────────────────────
def run_agent(user_input: str, history: list, llm, retriever) -> str:
    """Run the agent and return response"""

    # Safety check
    unsafe_patterns = [
        "send email", "update crm", "approve discount",
        "schedule a call", "contact the customer",
        "modify", "delete", "give discount"
    ]
    for pattern in unsafe_patterns:
        if pattern in user_input.lower():
            return (
                "❌ I cannot perform that action — I am a read-only decision support agent.\n\n"
                "I can help you prepare content or analysis for your review, "
                "but the action itself must be taken by you.\n\n"
                "Would you like me to prepare an account health summary instead?"
            )

    # Get account data
    account_data = ""
    for account in ACCOUNTS:
        if account.lower() in user_input.lower():
            acc = ACCOUNTS[account]
            risk = get_risk_level(account)
            account_data = f"""
Account: {account}
Risk Level: {risk}
Industry: {acc['industry']} | Plan: {acc['plan']} | Users: {acc['users']}
Open Tickets: {acc['open_tickets']} | Usage Drop: {acc['usage_drop_pct']}%
NPS Score: {acc['nps_score']} | Days Since Login: {acc['days_since_login']}
Contract Value: ${acc['contract_value']:,}
"""
            break

    if not account_data:
        # Portfolio overview
        account_data = "Portfolio:\n"
        for account in ACCOUNTS:
            risk = get_risk_level(account)
            acc = ACCOUNTS[account]
            account_data += f"- {account}: {risk} | Tickets: {acc['open_tickets']} | NPS: {acc['nps_score']}\n"

    # Retrieve playbooks
    docs = retriever.invoke(user_input)
    playbook_context = "\n\n".join([doc.page_content for doc in docs])

    # Build prompt
    system_prompt = f"""You are an AI Customer Health & Churn Prevention Copilot.
Decision-support only. Never send emails, modify CRM, approve discounts.

Account Data:
{account_data}

Relevant Playbooks:
{playbook_context}

RESPONSE STRUCTURE:
1. RISK LEVEL: [CRITICAL/HIGH/MEDIUM/LOW]
2. ASSESSMENT: 2-3 sentences
3. KEY RISK SIGNALS: Top 3 with data points
4. PLAYBOOK RECOMMENDATION: Cite specific playbook
5. RECOMMENDED ACTIONS: 3 specific next steps for CSM
6. ESCALATION FLAG: YES/NO with reason"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    messages.append({"role": "user", "content": user_input})

    response = llm.invoke(messages)
    return response.content

# ── UI LAYOUT ─────────────────────────────────────────────────
st.title("🤖 Customer Health & Churn Prevention Copilot")
st.caption("AI-powered decision support for Customer Success Managers — read-only, safety-first")

# Sidebar — Account Health Dashboard
with st.sidebar:
    st.header("📊 Portfolio Health")
    st.divider()

    for account_name in ACCOUNTS:
        acc = ACCOUNTS[account_name]
        risk = get_risk_level(account_name)
        emoji = RISK_COLORS[risk]

        with st.expander(f"{emoji} {account_name} — {risk}"):
            st.metric("Open Tickets", acc["open_tickets"])
            st.metric("Usage Drop", f"{acc['usage_drop_pct']}%")
            st.metric("NPS Score", acc["nps_score"])
            st.metric("Days Since Login", acc["days_since_login"])
            st.metric("Contract Value", f"${acc['contract_value']:,}")

    st.divider()
    st.caption("💡 Ask me about any account or your full portfolio")

# Initialise
llm = init_llm()
retriever = init_rag()

# Session state for conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Suggested questions
if not st.session_state.messages:
    st.info("👋 Welcome! Try asking:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Check GlobalBank"):
            st.session_state.suggested = "Check GlobalBank for me"
    with col2:
        if st.button("Portfolio overview"):
            st.session_state.suggested = "Give me a risk overview of my portfolio"
    with col3:
        if st.button("QBR prep for Acme Corp"):
            st.session_state.suggested = "Help me prepare for my QBR with Acme Corp"

# Chat input
prompt = st.chat_input("Ask about account health, risk, playbooks, or QBR preparation...")

# Handle suggested question buttons
if "suggested" in st.session_state:
    prompt = st.session_state.suggested
    del st.session_state.suggested

if prompt:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analysing account data and retrieving playbooks..."):
            response = run_agent(prompt, st.session_state.history, llm, retriever)

        st.markdown(response)

        # Show escalation warning
        if "ESCALATION REQUIRED" in response or "ESCALATION FLAG: YES" in response:
            st.warning("⚠️ This account requires immediate escalation to VP Customer Success")

    # Update history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history.append(HumanMessage(content=prompt))
    st.session_state.history.append(AIMessage(content=response))