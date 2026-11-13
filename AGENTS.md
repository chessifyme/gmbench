# Rules for anyone (human or agent) working in this repository

## Naming

File, folder, run and task-id names use lowercase letters, digits and underscores only, never hyphens or spaces (`ap_0042`, `claude_openrouter`, `runs/claude_openrouter/20261017_120000`). `bench.naming.assert_underscore_name` enforces it; `make validate` checks the data tree.

## Code

- Python 3.12 or newer, managed by `uv`; every command goes through the `Makefile`.
- No mocks, placeholders or fallbacks in product code. Test doubles live under `tests/`.
- No silent exception swallowing. Fail fast with a clear error that names what was wrong.
- Strict typing: every function annotated, parameters passed by name (functions take keyword-only arguments).
- Few comments; names and structure carry the meaning. DRY and KISS.
- Pydantic models use `extra="forbid"`.
- Write a failing test first, then the code that makes it pass, then run the whole suite.

## Models

- Benchmark runs and their grading go through OpenRouter's API (transport `openrouter_api`, key `OPENROUTER_API_KEY`), every model on a pinned provider. Results from the command-line transports (`claude_cli`, `codex_cli`, `cursor_cli`, and `agy_cli` for Google Antigravity) are for local development, are never mixed with API results and are never published.
- `BENCH_PROFILE=claude_only` is a development profile whose results are never published; published results use the grading models in `config/models.yaml`, the same for every entrant: one extractor and one judge (`jury_size` and `claim_jury_size` 1 in `config/bench.yaml`).

## Data and secrets

- Secrets live only in `.env` (documented in `.env.example`) and are read only by the settings classes in `bench.config` (`EngineEnv`, `DataEnv`, `HttpEntrantEnv`, `OpenRouterEnv` and the others there). Never print, log or commit a key or password.
- Tasks and sheets are one YAML file each under `data/public/`; compiled files are build outputs.
- No usernames, account ids, emails or session ids in any committed file.
- Disclosure: the grading reference (Stockfish 19 at fixed node budgets) and every entrant's tool use are disclosed on the dashboard, in the methodology and on agent cards. Stock and third-party entrants name the engine when they use one.
  Never let the reference become a feature of any entrant.
