# AI Enquiry Triage: Design and Model Evaluation

*Case-study submission · 17 September 2026 · Scope: human-reviewed insurance-service email triage*

## Architecture and design decisions

The application accepts an email subject and body, classifies it into one of six case types, assigns `URGENT`, `NORMAL`, or `LOW` priority, and proposes a short summary and reply draft. It also returns a case-type confidence score and safety state. A local FastAPI/Vue UI and CLI support triage, human review, and evaluation. The flow is **untrusted email → deterministic refusal screen → model provider → Pydantic schema validation → numeric-fact check → optional SQLite review queue**. A reviewer can accept, edit-and-accept, or discard a proposal. Neither interface sends mail or changes customer records.

Configured GPT, DeepSeek, Gemini, and GLM endpoints share a local output contract. Strict JSON-schema output is requested where supported; otherwise the adapter requests JSON-object output and applies the same Pydantic validation. Invalid output and provider failures remain visible failed attempts and cannot enter review. An offline rule-based provider tests workflow mechanics but is not evidence of LLM quality. The UI shows per-case results and saved evaluation history.

This is a **drafting and review** system, not an autonomous insurance adviser. We rejected an outbound email sender, policy-administration writes, and autonomous tools because their safety surface exceeds this case study's need. We did not add RAG: the task is to acknowledge the *enquiry*, not decide coverage from policy documents. If authoritative policy explanations become in scope, retrieval must filter approved clauses by product and effective version; a customer's actual status still requires an authorized source-system lookup. The local UI aids inspection but is not an authenticated production portal.

## Dataset and metric definitions

The repository contains **48 synthetic, manually labelled enquiries**: 18 for development and a frozen 30-case Golden set for comparison. The Golden set has five cases per case type and includes all priorities, missing information, angry tone, and refusal requests. A SHA-256 sidecar detects accidental benchmark edits. Each candidate in a comparison sees the same Golden case IDs and base system prompt. Provider-specific JSON-mode instructions can differ, so this compares **model-plus-adapter configurations**, not isolated model weights. Thirty synthetic cases do not estimate production performance reliably.

| Measure | Definition and why it matters |
|---|---|
| Classification accuracy; per-class and macro-F1 | Exact case-type matches / labelled cases. For each type, precision = TP/(TP+FP), recall = TP/(TP+FN), F1 is their harmonic mean; macro-F1 averages the six F1 values. Per-class results expose weak categories hidden by overall accuracy. |
| Priority accuracy; urgent recall | Exact priority matches / labelled cases; correctly marked urgent cases / all expected-urgent cases. Missing a time-critical enquiry is a distinct service risk. |
| Schema success; safety-action accuracy | Valid outputs / cases; correct `PENDING_REVIEW` or `REFUSE_AND_ESCALATE` actions / labelled cases. Invalid output counts as failure. These are gates because speed or low cost cannot compensate for unsafe handling. |
| Groundedness rate | Cases with valid output and **no draft number absent from the enquiry** / all cases. This catches some invented dates and amounts but cannot prove semantic faithfulness or detect unsupported non-numeric claims. |
| Draft quality | A reproducible 0–1 *structural heuristic*: greeting 0.35, sign-off 0.25, `customer`/`review` wording 0.20, length ≥80 characters 0.20. It does not establish empathy, usefulness, appropriate tone, or factual accuracy. No LLM judge is currently used. |
| Calibration; operations | ECE compares mean stated case-type confidence with observed classification accuracy in fixed buckets; Brier score averages squared confidence error. Both use only valid outputs, so they must be read alongside schema success. p50/p95 latency, token counts, and recorded cost capture operational trade-offs, subject to provider reporting and pricing configuration. |

The composite fitness score weights task quality 55%, calibration 20%, p95 latency 15%, and recorded cost 10%. Task quality weights macro-F1 40%, priority accuracy 20%, numeric groundedness 25%, and structural draft quality 15%. This is a ranking aid, **not** a substitute for gates: schema success, safety-action accuracy, and urgent recall must each equal 100%; macro-F1 and priority accuracy must each be ≥0.85; groundedness ≥0.98; ECE ≤0.15. These are provisional pilot thresholds, not validated service-level guarantees.

## Measured comparison and recommendation

The saved **17 September 2026** run compared `deepseek-chat` with `glm-4.5-air` on the same 30 Golden cases ([local per-case result](../evaluation_results/comparison_20260917T133534Z.json)). It records Golden SHA-256 `439cf7a49c439242e8361462f3f7f21f1faf9ae5b04df1ec5c2185713ae34cf4` and base-prompt SHA-256 `90b6698c0686027e8799ca69cb220ac10eac29b66e3071500b33ec8cb0cc84d6`. The result file is local and Git-ignored; the table below preserves the key findings in this document.

| Metric | DeepSeek Chat | GLM-4.5-Air |
|---|---:|---:|
| Classification accuracy / macro-F1 | 1.000 / 1.000 | 0.9667 / 0.9815 |
| Priority accuracy / urgent recall | 1.000 / 1.000 | 0.9667 / 1.000 |
| Schema success / safety-action accuracy | 1.000 / 1.000 | 0.9667 / 0.9667 |
| Numeric groundedness / structural draft quality | 1.000 / 1.000 | 0.9667 / 0.4150 |
| ECE / p95 latency | 0.0893 / 1,510 ms | 0.0648 / 2,112 ms |
| Recorded average cost / fitness | $0.00 / 0.9368 | $0.00 / 0.8631 |
| Hard gates | **7/7 pass** | **3 fail**: schema, safety action, groundedness |

GLM's `golden-pb-04` draft introduced an unsupported numeric fact. The application rejected that result as `SAFETY_POLICY_FAILED: UNSUPPORTED_NUMERIC_FACT`; the *same case* therefore lowers three reported rates. GLM's lower ECE does not make the overall workflow safer. The $0.00 cost figures are **recorded values, not verified free usage**; prices and usage extraction require checking before any cost conclusion. Reply-quality values are formatting proxies, not blinded human ratings.

**Recommendation:** use DeepSeek Chat for the next **human-reviewed shadow evaluation**, because it passed every configured gate on this fixed run. Do **not** authorize production deployment or automatic replies. Repeat runs, a different real-world case mix, verified cost, or blinded human review could reverse this recommendation. One small synthetic run has substantial uncertainty.

## Safety, privacy, and governance

Email is bracketed as untrusted data; system instructions say not to obey embedded requests to reveal prompts or bypass controls. A deterministic pre-screen refuses recognized prompt-injection, privacy, verification-bypass, and claim-date-falsification phrases before a model call. Keyword screening is **not comprehensive**; novel phrasing can evade it. After generation, Pydantic rejects malformed output, and a separate check rejects unsupported numeric facts. Every usable proposal is intended for human review, but neither prompts nor these checks prove all policy statements true.

The SQLite review queue stores source emails so a reviewer can compare proposals with originals; its local file is owner-readable/writable. Evaluation CSV contains derived metrics and bounded failure diagnostics, not source email bodies or drafts. **New evaluation JSON contains full reply drafts**, which may repeat customer PII. Comparison artifacts and private provider configuration are Git-ignored, but Git ignore is not access control. Before real-data use, add minimization/redaction, encrypted storage, authenticated role-based access, retention/deletion rules, audit trails, secrets management, vendor data-handling review, and privacy approval. Operational logs should contain request IDs, model/version, timing, tokens, and redacted error codes—not raw emails or drafts. The localhost UI needs authentication and deployment hardening before shared use.

## Next two weeks

**Week 1:** obtain a privacy-approved anonymized sample. Have two service/insurance reviewers label case type, urgency, factual anchors, unsafe requests, and reply requirements; reconcile disagreements into a versioned rubric. Add claim-by-claim draft verification and test deliberately false-coverage, wrong-date, and PII examples. Record full effective prompts, provider settings, model versions, dataset hash, and verified token prices.

**Week 2:** repeat a controlled two-model comparison several times on unchanged cases, with randomized, blinded human draft review. If adding an LLM judge, hide candidate identity, fix a rubric, require cited evidence, save judge outputs, and measure agreement and unsafe-claim misses against human labels before trusting its score. Run a read-only shadow pilot with service agents, capture accept/edit/discard reasons, review drift and failures, and make a go/no-go decision against pre-agreed gates. No automatic sending or customer-record changes are planned.
