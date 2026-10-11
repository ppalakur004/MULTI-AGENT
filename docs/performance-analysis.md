# Final cost and latency analysis

These measurements were captured on October 10, 2026 from the final
`synthetic-health-reviewed-v3` implementation using local Ollama `qwen3:8b`. All inputs are
fictional. Source: `.runs/submission-live-evaluation.json` and the three referenced JSONL traces.

## Three required full-pipeline runs

| Application | Case ID | Outcome | Wall time | Input tokens | Output tokens | Total tokens | Provider charge |
|---|---|---|---:|---:|---:|---:|---:|
| A-1001 | APP-5B76161FE07C | Approve | 21.853 s | 1,180 | 486 | 1,666 | $0 |
| A-1004 | APP-A2F6F201E2B3 | Approve | 20.752 s | 1,181 | 486 | 1,667 | $0 |
| A-1006 | APP-0C12E2C6DC50 | Human review (fraud flag) | 22.681 s | 1,190 | 508 | 1,698 | $0 |

All three matched their expected outcome and executed intake, enrichment, risk, and
recommendation. Each trace records stage input/output, duration, token usage, model, and prompt
hash. A-1006 demonstrates that operational review flags remain separate from the health-risk
score while still forcing a human decision.

Mean wall latency = `(21.853 + 20.752 + 22.681) / 3` = **21.762 seconds**.

Total tokens = `1,666 + 1,667 + 1,698` = **5,031 tokens**; mean = **1,677 tokens/case**.

Ollama has no direct per-token provider bill, so the measured provider charge is **$0**. This is
not a claim that operating cost is zero: hardware, electricity, hosting, database, engineering,
and human-review costs were not measured.

## Stage latency

| Application | Intake | Enrichment | Risk | Recommendation |
|---|---:|---:|---:|---:|
| A-1001 | 0.871 s | 2.631 s | 15.699 s | 2.394 s |
| A-1004 | 0.794 s | 2.446 s | 14.892 s | 2.453 s |
| A-1006 | 0.793 s | 2.542 s | 16.398 s | 2.775 s |

Risk review dominates latency because Qwen must return six structured rule decisions and Python
then independently derives and compares the reviewer score. Parent stage timings already include
nested model/tool work and must not be added to generation spans again.

## Budget and suitability

For this demonstration, use a five-second synchronous response target and an overnight target of
1,000 sequential cases within eight hours. These are explicit analysis assumptions, not insurer
SLAs.

- **Real time:** none of the three cases met five seconds. The current local model is unsuitable
  for an interactive five-second experience without optimization or asynchronous status updates.
- **Overnight batch:** at the measured mean, 1,000 sequential cases would take approximately
  `21,762 seconds`, or **6.05 hours**. That fits the assumed eight-hour window on an otherwise idle
  machine.

The batch conclusion is an extrapolation from three cases, not a load test. Sustained contention,
model warm-up, retries, and time spent in human review are excluded. Before production capacity
claims, measure a larger sample, percentiles, concurrency, and infrastructure cost.
