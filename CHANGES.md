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
