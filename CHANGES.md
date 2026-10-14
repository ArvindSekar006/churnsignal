# ChurnSignal v4 - changes

| Area | Change |
|---|---|
| RAG | Playbooks embedded (fastembed, bge-small) and retrieved via `get_playbook` tool; `/api/retrieve?q=` shows hits + scores; keyword fallback if the model can't load (`/api/metrics` -> `rag_mode`) |
| Guardrail | Intent-based regex (write verb near data target, "mark ... as", "set ... to", send/approve/contact). Email *drafting* allowed. Tested on 20 queries |
| Monitoring | One JSON log line per request (account ids hashed, no message text); `/api/metrics` (counts, block/error rate, avg/p50/p95 latency) |
| Agent loop | Forced final answer if loop ends on tool calls; account-id validation in every tool; per-tool exception handling |
| Memory | Entity register (last user-named account) injected for B/C; context-conflict notice when selected account != named account |
| Feedback | `POST /api/feedback` (thumbs + correction); corrections feed later answers in the same session; UI buttons added |
| Deploy | render.yaml has all 3 secrets + health check; requirements adds supabase, fastembed; debug console.logs removed |

Variant prompts: B/C text unchanged except playbook block -> retrieval instruction; C adds `advisory_action` intent
(fixes the Q5 false-positive). Re-run the 5 test queries and record before/after in the report.

## v4.1 (after the B/C evaluation round)
- Email drafts now use a fixed layout: NOTE TO CSM (internal: account mismatch, stated-vs-record differences, blank fields) + DRAFT EMAIL (customer-facing, facts from records only, no invented availability/offers). Fixes the feedback-demo finding where internal notes leaked into the email.
- `risk_level` computed by the platform is attached to account data and the prompts require using it (fixes MEDIUM-vs-HIGH mismatch seen in Q5).
- CSM corrections are injected as a user-turn instruction instead of a system-prompt suffix (stronger effect).
- Escalation banner text no longer claims VP-level criteria.
- Account scope: when the CSM is on one account's page and asks about another, the model (not a hard-coded rule) is told via a context-conflict note + ACCOUNT SCOPE prompt rule to stop, say which account the page is on vs. asked about, and ask the CSM to confirm.
