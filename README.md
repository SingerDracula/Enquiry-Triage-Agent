# AI Enquiry Triage

A production-minded MVP for classifying inbound insurance customer emails, drafting a grounded reply, and placing every proposal into human review. It never sends email or changes customer data.

## What is implemented

- A strict Pydantic JSON contract for classification, priority, summary, reply draft, confidence, and safety status.
- Safe, visible handling of malformed model output: invalid JSON is not silently repaired and cannot enter review.
- A local SQLite review queue with `ACCEPTED`, `EDITED_AND_ACCEPTED`, and `DISCARDED` decisions.
- 48 synthetic labelled enquiries: 30 frozen golden records and 18 development records. They cover all six case types, urgent/normal/low priorities, ambiguous messages, angry tone, missing information, prompt injection, and unsafe requests.
- A re-runnable evaluation harness for macro-F1, priority accuracy, urgent recall, groundedness, reply-quality heuristic, calibration, latency, cost, quality gates, and a weighted model comparison.
- An offline deterministic demo provider so the full workflow and test suite run without credentials. It is a workflow baseline, not a substitute for the case study's two real LLMs.
- First-class GPT, DeepSeek, Gemini, and Zhipu GLM adapters, plus an OpenRouter Jev classification overlay, with local Pydantic enforcement for every final result.

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

Run the offline comparison and create auditable JSON/CSV artifacts under the project's `evaluation_results/` folder:

```bash
.venv/bin/triage-agent evaluate
```

Run the `unittest` suite after installing the project dependency:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## Local web UI

Start the local FastAPI and Vue UI after installing the updated dependencies:

```bash
.venv/bin/pip install -e .
.venv/bin/triage-web
```

Open `http://127.0.0.1:8000`. The page accepts an email subject/body for triage, lets a reviewer accept, edit-and-accept, or discard queued drafts, and runs provider evaluations. After evaluation, it lists every case for each provider with pass/fail status, expected and actual labels, validation/grounding results, confidence, reply-quality heuristic, latency, and failure reason; the list can be filtered to passed or failed cases, and each case can expand to show its saved reply draft. Rejected model replies also appear with their validation reason, but cannot enter the review queue. Previous evaluations can be loaded without rerunning models. Evaluation JSON/CSV files are saved in the project's `evaluation_results/` folder. The server binds only to localhost, reads model settings from private `config.toml`, and never sends email.

Web evaluations return a job ID immediately. The page polls `/api/evaluation-jobs/{job_id}` and loads the saved comparison, including the recommended model, when the job completes. Job status is stored in the private application-data directory, so a refreshed page can resume polling. The work still runs inside the web process; a process restart interrupts an unfinished evaluation and the page reports that interruption. Use the CLI evaluation command when the hosting platform may restart long-running web processes.

## Real LLM comparison

Provider URLs must use HTTPS (plain HTTP is accepted only for localhost development). Prices are configuration values, not hard-coded assumptions; record the values used for each evaluation.

| Provider | CLI spec | Configuration section | Structured-output strategy |
| --- | --- | --- | --- |
| OpenAI GPT | `gpt` | `[providers.gpt]` | Chat Completions strict JSON Schema |
| DeepSeek | `deepseek` | `[providers.deepseek]` | JSON Object mode, explicit prompt contract, then local validation |
| Google Gemini | `gemini` | `[providers.gemini]` | GenerateContent JSON Schema, then local validation |
| Zhipu GLM-4.7-Flash | `glm` | `[providers.glm]` | Chat Completions JSON Object mode, then local validation |
| OpenRouter Jev + base provider | `jev` | `[providers.jev]` | Decisions API `/api/alpha/decisions` Choice answers for `case_type` and `priority` |

Copy the versioned template to the private configuration file, then set the model ID and API key there. `config.toml` is ignored by Git, so it will not be committed.

For cost estimates, also set each provider's current `input_usd_per_million` and `output_usd_per_million` in `config.toml`. The values are USD per million tokens and must come from the provider's pricing page for the selected model. Zero template values mean the price is unknown; evaluation displays cost as unavailable and gives its cost component a neutral score instead of treating it as free. If a provider is genuinely free, set `pricing_is_free = true` with both rates at zero. Cost estimates use returned token counts; calls that fail before token usage is available may still be billed. Saved evaluations retain their original cost data, so rerun an evaluation after correcting prices.

```bash
cp config.example.toml config.toml
```

Run a real-provider comparison after setting the model IDs and API keys you use in `config.toml`:

```bash
.venv/bin/triage-agent evaluate \
  --models gpt,deepseek,gemini,glm
```

For DeepSeek Chat Completions, choose `deepseek`. This sends `response_format={"type":"json_object"}` and embeds the required JSON Schema and example in the system prompt. The response still must pass local Pydantic validation and safety checks before it can be reviewed.

`compatible` is for endpoints that accept `response_format={"type":"json_schema", ...}`. DeepSeek rejects that mode with HTTP 400 ("This response_format type is unavailable now"), so pointing `compatible` at a DeepSeek base URL always fails. Use `deepseek` instead; the failure output names the HTTP status and the provider's error message so a mismatch is diagnosable.

```bash
.venv/bin/triage-agent triage data/example_email.txt \
  --provider deepseek

.venv/bin/triage-agent triage data/example_email.txt \
  --provider gpt

.venv/bin/triage-agent triage data/example_email.txt \
  --provider gemini

.venv/bin/triage-agent triage data/example_email.txt \
  --provider glm

.venv/bin/triage-agent triage data/example_email.txt \
  --provider jev
```

`glm` uses the official `https://open.bigmodel.cn/api/paas/v4` base URL by default and the `glm-4.7-flash` model ID in the configuration example. It sends JSON Object mode with thinking disabled and validates the response locally; it does not claim server-side strict JSON Schema enforcement. Add the `[providers.glm]` section from `config.example.toml` to an existing private `config.toml` if you are upgrading an existing installation.

### OpenRouter Jev classification overlay

`jev` is intentionally a composite provider because Jev returns typed decisions rather than reply text. The adapter first sends the untrusted email to OpenRouter's `/api/alpha/decisions` endpoint with two independent Choice questions. It then gives those authoritative labels to the configured `base_provider`, which generates `summary`, `draft_reply`, and `safety_status` through the existing pipeline. The adapter finally enforces the Jev labels so a generator cannot replace them. Since the output contract defines `confidence.score` as the probability that `case_type` is correct, that score and its methodology are updated to describe Jev's case-type decision. Token usage, latency, and cost include both sequential calls.

Create an API key in OpenRouter and add this private configuration; do not commit the real key:

```toml
[providers.jev]
model = "typesafe/jev-1.13"
api_key = "your_openrouter_api_key"
base_provider = "deepseek"
base_url = "https://openrouter.ai/api/alpha"
timeout_seconds = 30
input_usd_per_million = 0.042
output_usd_per_million = 0
```

If Jev fails, returns an unknown label, or omits probabilities, the whole attempt fails visibly with `PROVIDER_FAILURE`; it never silently falls back to the base provider's classification. Locally detected refusal cases still stop before either remote provider is called. Check OpenRouter's current Jev model page before recording pricing, because rates can change.

To compare the original base model against the same draft generator with Jev classification:

```bash
.venv/bin/triage-agent evaluate --models deepseek,jev
```

`compatible` remains available for another OpenAI-compatible endpoint that supports strict JSON Schema. Do not route a provider through the wrong adapter: DeepSeek and GLM use JSON Object mode, while Gemini uses its GenerateContent endpoint. Use `--config /path/to/config.toml` if the private file is not in the current directory.

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
evaluation_results/  Project-local comparison JSON/CSV artifacts (Git-ignored; JSON includes valid and rejected drafts)
results/             Legacy optional output directory (ignored by Git)
docs/                Case-study write-up and implementation notes
```

By default, the review database uses a private user application-data directory, while evaluation artifacts are written to `evaluation_results/` in the source project (or the current directory when installed outside the source tree). Set `AI_TRIAGE_DATA_DIR` to choose another review-database location, or pass `--database` / `--output-dir` explicitly. Comparison artifacts in the project folder are Git-ignored because new JSON results contain valid drafts and rejected model output; protect and review them before sharing. The metrics CSV omits all reply text.

See [docs/CASE_STUDY_WRITEUP.md](docs/CASE_STUDY_WRITEUP.md) for the concise architecture, trade-offs, evaluation design, and next steps.
