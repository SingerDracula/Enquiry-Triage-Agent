# AI Enquiry Triage

A production-minded MVP for classifying inbound insurance customer emails, drafting a grounded reply, and placing every proposal into human review. It never sends email or changes customer data.

## What is implemented

- A strict Pydantic JSON contract for classification, priority, summary, reply draft, confidence, and safety status.
- Safe, visible handling of malformed model output: invalid JSON is not silently repaired and cannot enter review.
- A local SQLite review queue with `ACCEPTED`, `EDITED_AND_ACCEPTED`, and `DISCARDED` decisions.
- 48 synthetic labelled enquiries: 30 frozen golden records and 18 development records. They cover all six case types, urgent/normal/low priorities, ambiguous messages, angry tone, missing information, prompt injection, and unsafe requests.
- A re-runnable evaluation harness for macro-F1, priority accuracy, urgent recall, groundedness, reply-quality heuristic, calibration, latency, cost, quality gates, and a weighted model comparison.
- An offline deterministic demo provider so the full workflow and test suite run without credentials. It is a workflow baseline, not a substitute for the case study's two real LLMs.
- A generic OpenAI-compatible structured-output adapter for real model comparisons.

## Quick start

Requires Python 3.11 or newer. Use an interpreter such as `python3.11` or `python3.12`; do not assume an unversioned `python3` is new enough.

```bash
cd /Users/zhongbo/IdeaProjects/ai-enquiry-triage
python3.11 -m venv .venv
.venv/bin/pip install -e .
```

Run a local demo triage. It writes a pending review record but does not send anything:

```bash
.venv/bin/triage-agent triage data/example_email.txt --subject "Payment page failed"
.venv/bin/triage-agent review list
.venv/bin/triage-agent review show 1
.venv/bin/triage-agent review decide 1 ACCEPTED
```

Run the offline comparison and create auditable JSON/CSV artifacts under `results/`:

```bash
.venv/bin/triage-agent evaluate
```

Run the `unittest` suite after installing the project dependency:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## Real LLM comparison

Copy `.env.example` to `.env`, set values for an HTTPS endpoint that supports OpenAI-compatible chat completions with JSON Schema output, then export its variables in your shell. Plain HTTP is accepted only for a localhost development endpoint. Prices are configuration values, not hard-coded assumptions; record the values used for each evaluation.

```bash
set -a
source .env
set +a
.venv/bin/triage-agent evaluate \
  --models compatible:model-a,compatible:model-b
```

The two model identifiers must be genuinely distinct models, tiers, or providers. The default `demo-fast,demo-conservative` command validates the evaluation pipeline only; it must not be presented as a two-LLM case-study comparison.

## Output contract

Every successful provider call must validate against `TriageResult`. The permitted categories are:

```text
POLICY_QUERY | PREMIUM_BILLING | ADDRESS_CHANGE | CLAIM | COMPLAINT | OTHER
URGENT | NORMAL | LOW
PENDING_REVIEW | REFUSE_AND_ESCALATE
```

Invalid output becomes a `TriageAttempt` with `SCHEMA_VALIDATION_FAILED` and no `result`. It cannot be stored in `ReviewStore`.

## Safety boundaries

- Email text is wrapped as untrusted data in the model prompt; embedded instructions cannot authorize new actions.
- Requests for internal prompts, credentials, third-party policy details, identity-verification bypasses, or claim-date manipulation receive `REFUSE_AND_ESCALATE`.
- The private local review SQLite database retains the source email only so an authorized reviewer can compare it with the proposal. Its file mode is set to owner-read/write; it is never included in evaluation CSV/JSON artifacts. Production requires encryption, role-based access, retention limits, and an authenticated source-message reference.
- The deterministic groundedness check flags reply numbers that do not occur in the customer email. It is deliberately a limited safety net, not a semantic proof of truth.
- The only review operations are accept, edit-and-accept, and discard. There is no mail sender, customer-system connector, or automatic side effect.

## Evaluation design

`triage-agent evaluate` runs the same frozen golden set for each model and writes raw per-case records plus a summary. The composite fitness score uses task quality (55%), calibration (20%), p95 latency (15%), and cost (10%). Safety and schema success are reported as hard gates, not tradeable score components.

The reply-quality score is an intentionally transparent structural heuristic. For a real submission, add blinded human review or an independent judge model with a fixed rubric, stored judge outputs, and hidden candidate-model identity.

Suggested human-assisted pilot gates are in the generated `gates` section: 100% schema success, 100% safety action accuracy, 100% urgent recall on the golden set, macro-F1 and priority accuracy at least 0.85, groundedness at least 0.98, and ECE at most 0.15. A synthetic golden set is not production evidence.

## Repository layout

```text
src/enquiry_triage/  Agent, models, providers, review store, CLI, evaluation
data/                Synthetic development set, frozen golden set, integrity hash
tests/               Offline schema, safety, data, review, and evaluation tests
results/             Optional project-local comparison artifacts (ignored by Git)
docs/                Case-study write-up and implementation notes
```

By default, installed CLI state and generated results use a private user application-data directory. Set `AI_TRIAGE_DATA_DIR` to choose another private location, or pass `--database` / `--output-dir` explicitly.

See [docs/CASE_STUDY_WRITEUP.md](docs/CASE_STUDY_WRITEUP.md) for the concise architecture, trade-offs, evaluation design, and next steps.
