# Public Syllabus Research

This command-line Python application discovers, retrieves, verifies, compares, and exports
public course syllabi and detailed outlines. It is designed for a VS Code terminal and produces
an Excel workbook, not a web interface.

The default profile researches a conceptual, application-focused LLM course for master's
students in business analytics. Discovery uses the OpenAI Responses API hosted `web_search`
tool. The application then retrieves the cited HTML/PDF materials itself and analyzes that
local evidence; it does not treat a search summary as verification. The implementation follows
the official OpenAI documentation for [Responses](https://developers.openai.com/api/reference/responses/overview),
[web search](https://developers.openai.com/api/docs/guides/tools-web-search), and
[reasoning models](https://developers.openai.com/api/docs/guides/reasoning).

## Set up in VS Code

From the repository root:

```bash
python3 -m venv syllabus_research/.venv
source syllabus_research/.venv/bin/activate
python -m pip install -e 'syllabus_research[test]'
```

Keep the working directory at the repository root so the application can reuse `shared/`.
Put the API key in the repository-root `.env` file:

```text
OPENAI_API_KEY=...
```

The app reads a basic `KEY=VALUE` file itself. It does not overwrite variables already exported
by VS Code or the shell. `.env` and generated run directories are ignored by Git. See
[.env.example](.env.example) for optional model/retry settings.

## Run

Activate the environment, then run a normal research job:

```bash
syllabus-research
```

For a low-cost live smoke run:

```bash
syllabus-research --max-candidates 2
```

Each run creates `runs/<UTC timestamp>/` containing:

- `syllabus_comparison.xlsx` — the final filterable workbook;
- `state.json` — resumable structured state, failures, and usage totals;
- `artifacts/raw/` and `artifacts/text/` — retrieved originals and extracted text;
- `provenance/` — the exact profile snapshot, run settings, and raw API responses.

Normal discovery is split into configurable batches and uses strict Responses API JSON Schema.
Each raw batch response is saved before parsing, so an incomplete or malformed batch is recorded
and the remaining batches can continue instead of losing the whole discovery run.

Resume an interrupted run without repeating completed stages:

```bash
syllabus-research --resume syllabus_research/runs/20261015T180000Z
```

Retrieval and analysis failures are recorded per source/course and the run continues. API calls
and retrieval retries are bounded. The workbook reports an honest shortfall when fewer than the
configured minimum qualify; the selector does not silently relax its rules.

## Edit sources and research profiles

Edit [the default profile](config/profiles/llm_business_analytics.json), or copy it and pass the
copy with `--profile path/to/profile.json`. The JSON controls:

- required and covered-elsewhere topics;
- seed sources, explicit inclusions, and explicit exclusions;
- audience, language, allowed provider types, and minimum substantive coverage;
- the rolling recency window and configurable selection target (currently 15–30);
- discovery/analysis models, reasoning effort, call limits, output limits, and retrieval limits;
- an optional estimated-dollar ceiling, conservative per-call reserve, and user-supplied current
  token/web-search rates (left `null` by default so stale prices are never silently assumed).

An explicit inclusion bypasses normal selection criteria but is visibly flagged when it falls
outside them. Undated candidates stay unresolved. Course dates must be supported by a course
term or explicit publication/update statement; copyright and retrieval dates do not qualify.
To create a broader business-school AI profile, copy the file and replace its `topics`,
`audience`, `seed_sources`, and selection criteria.

## Interpret the workbook

- **Selected courses** contains one row per selected course, source links, explicit date evidence,
  selection rationale, and the full ordered outline.
- **Topic coverage** contains one row per course/topic. `explicit` means direct evidence;
  `inferred` means a bounded inference with cited support; `not_evidenced` means unknown from the
  available outline, not that the course omits the topic.
- **Potential curriculum gaps** separates course-material evidence from audience-oriented
  recommendations. Prevalence describes the selected collection and is not an importance score.
- **Excluded and unresolved** records duplicates, stale/undated candidates, retrieval failures,
  uncertainties, and other reasons.
- **Run information** records criteria, models, limitations, API usage, and any selection shortfall.
Dollar cost is not guessed from hard-coded prices because model/tool pricing can change. Populate
  the profile's rate fields to log an estimate and enforce a conservative spending ceiling; use
  the OpenAI usage dashboard and current pricing for authoritative cost.

Responsible-use coverage remains visible in the matrix but is marked as covered in another
class. Retrieved content is treated as untrusted data, not as instructions. The retriever uses
ordinary public HTTP access and does not bypass authentication, paywalls, robots challenges, or
other access controls.

## Shared OpenAI client compatibility

`shared.llms.openai_client.gpt_complete()` remains a Chat Completions wrapper and retains its
return type and parameter shape. The new additive `gpt_response()` helper enables Responses API
models, reasoning effort, built-in tools, and output-token limits. Credentials are resolved from
the existing `shared.config.API_KEY` setting or `OPENAI_API_KEY`; no repository-wide caller
migration is required. The client is now lazy when no key is present, so importing it for tests
or local-only tasks does not require credentials.

## Test

```bash
source syllabus_research/.venv/bin/activate
pytest syllabus_research/tests
```

Tests use local fixtures and fake API clients. They do not consume API credits. Live results are
always written under ignored `runs/` directories and are distinct from fixtures under `tests/`.

## License

This software is licensed under the MIT License used by the rest of the repository. See
[LICENSE](LICENSE) and the repository-level [SOFTWARE_LICENSE](../SOFTWARE_LICENSE).
