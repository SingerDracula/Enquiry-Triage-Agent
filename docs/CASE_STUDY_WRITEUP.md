# AI Enquiry Triage Agent Write Up

## Architecture and design decisions

The MVP accepts a plain-text inbound email and produces a schema-validated proposal containing a case type, priority, short summary, reply draft, confidence score, and safety status. The email is explicitly framed as untrusted content. A provider receives the fixed instruction and the enclosed email, returns structured JSON, and Pydantic validates it. Invalid output becomes a visible `SCHEMA_VALIDATION_FAILED` result, rather than a silently repaired free-text answer.

Every valid proposal is placed in a private local SQLite review queue. A reviewer can accept, edit-and-accept, or discard it while seeing the original message, subject, proposal, and safety state. The queue file is owner-read/write and is not an evaluation log; production needs encryption, role-based access, retention limits, and an authenticated source-message reference. It has no outbound mail or policy-administration integration, so a successful review decision cannot have an external side effect.

The implementation uses an offline rule-based provider for repeatable demonstrations and a generic structured-output HTTP adapter for configured real models. A full web UI, email connector, policy lookup, retrieval system, and autonomous tool use were rejected for the MVP: they add material safety and privacy surface without helping answer the case study's central questions about output quality, measurement, and human control.

## Metrics and benchmark design

The repository includes 48 artificial, manually labelled enquiries. Thirty form a frozen golden set; the remainder are a development set for prompt and rule iteration. Each golden record has expected case type, priority, safety state, and reply review notes. The golden file has a SHA-256 integrity file so accidental benchmark edits are detectable.

The harness reports classification accuracy, per-class precision/recall/F1, macro-F1, priority accuracy, urgent recall, safety-action accuracy, schema success, groundedness, reply-quality heuristic, confidence buckets, expected calibration error, Brier score, p50/p95 latency, token counts, and cost. Groundedness initially flags reply numbers not found in the source email. It should later be complemented by labelled atomic-fact checks and blinded human review.

The composite score weights task quality at 55%, calibration at 20%, latency at 15%, and cost at 10%. Safety and schema-validity gates are reported separately because a low cost cannot compensate for a privacy or malformed-output failure. For customer-facing drafting, response quality and calibration deserve more weight than cost, while latency remains important for an agent waiting at a service desk.

## Model comparison and recommendation process

The same prompt hash and same golden IDs are passed to at least two distinct configured models. The demo providers validate only the mechanics; they are not valid evidence for the final comparison. A candidate model should be selected only after it passes schema success, refusal handling, groundedness, and urgent-recall gates, then has the best composite score among the remaining candidates.

If model-generated reply quality is judged by an LLM, the judge must not be the candidate model. Candidate identities and order should be masked, a fixed rubric should score factual grounding, professionalism, usefulness, and safe escalation, and all judge inputs/outputs should be retained. A small human double-review sample should test whether the judge agrees with people.

## Safety and governance

Prompt-injection phrases, requests for internal data, third-party policy details, verification bypasses, and assistance with claim-date falsification are blocked before the model is called and receive a deterministic bounded refusal. Post-validation grounding rejects replies that introduce unsupported numeric facts. The agent never confirms coverage, amounts, claim outcomes, completed changes, or customer details absent from the enquiry. Production logging should use request IDs, model versions, timing, and token counts, while encrypting or redacting PII and applying short retention limits. Before production, add role-based access control, secrets management, audit trails, real privacy review, retention/deletion workflows, representative labelled data, and monitoring for drift and disproportionately poor handling of customer segments.

## Next two weeks

First, obtain a privacy-approved, anonymized sample of real enquiries and have two subject-matter reviewers label taxonomy, urgency, factual anchors, and reply requirements. Second, compare two real models with fixed prices, repeat runs, blinded quality review, and confidence calibration. Third, strengthen factual grounding through claim/coverage policy constraints and an explicit verifier. Finally, run the agent in read-only shadow mode with CS agents, capture accept/edit/discard reasons, and use those feedback labels to improve prompts and thresholds before any wider trial.
