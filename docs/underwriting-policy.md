# Synthetic health-insurance policy (synthetic-health-reviewed-v3)

This is a narrow teaching policy over fictional evidence. It demonstrates pre-underwriting
orchestration; it is not a medical diagnosis, actuarial model, premium quote, eligibility rule,
or binding coverage decision. Real health-insurance rules are jurisdiction-specific and require
legal, actuarial, clinical, privacy, and fairness review.

The health-risk index starts at 5 points:

- Declared smoker: add 20 points.
- One declared chronic condition: add 10 points; two or more: add 25 points.
- One or more hospitalizations in the last two years: add 20 points.
- Three or more claims in the last two years: add 15 points.
- One or more high-cost claims in the last two years: add 15 points.
- Cap the total at 100 points and divide by 100 for the displayed score.

Age, name, occupation, and coverage tier are collected but do not affect this illustrative score.

LOW: fewer than 30 points → recommend approve.
MEDIUM: 30 to fewer than 65 points → human review.
HIGH: 65 points or more → recommend deny.

Identity verification and fraud are operational gates, not health-risk weights. An unverified
identity or fraud flag requires human review without changing the health-risk score. Missing or
invalid external evidence stops automation before risk review; no placeholder score is generated.

Python calculates the authoritative score. Qwen receives normalized validated evidence and all six
rule definitions, then reports whether each rule triggered and the supporting evidence path. Python
derives a reviewer score from those decisions. The first mismatch receives one targeted correction
call. Continued score, band, factor, or data-quality disagreement preserves both assessments and
requires human review. The LLM cannot override the Python policy.

Evidence references must be published factors or operational review flags. Risk-review confidence
is derived from complete evidence, complete rule coverage, and exact agreement; it is not a
probability. Approval or denial also requires recommendation confidence at or above the configured
minimum (default 0.75). Recommendation confidence is model-reported and uncalibrated.

Eligibility verification and claims are fictional JSON records loaded by applicant ID. Unknown IDs
require human review rather than generated evidence. Failure simulations are injected in tests and
are separate from ordinary applicant data.
