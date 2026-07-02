# HANDOFF: Remediate the `pip-audit` CI failure (23 vulnerable transitive packages)

**Status:** Open. The `pip-audit` job in `.github/workflows/ci.yml` is the one remaining red check on `main`. It was deliberately left failing (owner decision) pending a real security-posture fix — this doc is that follow-up.
**Prepared:** 2026-07-02, after fixing the sibling `quality layer checks` failure (PR #3, merged) and shipping the office-verify + typst-backend features (PRs #1/#2, merged).
**Repo:** `remedy-server` (branch `main`). Run everything via `uv`.

---

## 0. Bottom line

`main`'s CI `pip-audit` job fails because the dependency tree carries **23 packages with known vulnerabilities**, and it *cannot* be made green by upgrading alone — **several vulnerabilities have no published fix version**. Fully resolving this is a security-vs-stability effort with real breakage risk (the tree includes a heavy ML/RAG/crawl stack whose runtime paths the unit tests do **not** exercise), plus some CVEs that can only be handled by a documented `--ignore-vuln` (or by dropping/replacing the dependency). This is a judgment task — treat it as one.

**Do not** silently mask the whole job (e.g. blanket `continue-on-error: true`) — the owner explicitly rejected that. **Do** reduce real exposure by upgrading what's cleanly upgradable, and handle the genuinely-unfixable CVEs with narrowly-scoped, *justified* `--ignore-vuln` entries (each with a one-line reason), so the gate stays meaningful.

---

## 1. Exactly what CI runs (the gate you must satisfy)

`.github/workflows/ci.yml`, job `security` / `pip-audit`:

```yaml
- name: Sync deps
  run: uv sync                     # BASE deps (no --extra dev)
- name: Run pip-audit
  run: |
    uv pip install --upgrade pip pip-audit
    uv run pip-audit --skip-editable --ignore-vuln CVE-2026-3219
```

Reproduce locally from a clean worktree branched off `origin/main`:

```bash
uv sync
uv run pip-audit --skip-editable --ignore-vuln CVE-2026-3219
```

`CVE-2026-3219` (pip itself) is already ignored — "no fixed pip release published yet." Note `uv sync` installs **base** `[project.dependencies]` only; the huge ML stack below is pulled **transitively** by base deps, so it *is* in scope for this job.

---

## 2. The authoritative vulnerable set (base deps, `origin/main` lockfile, 2026-07-02)

All 23 are **transitive** — none are named in `[project.dependencies]`. Fix versions are from pip-audit; **blank = NO FIX EXISTS** (must be `--ignore-vuln`'d with justification or the dependency dropped/replaced).

| Package | Locked | Fix available | Notes |
|---|---|---|---|
| aiohttp | 3.13.3 | 3.14.1 | many CVEs; clean minor bump likely |
| chromadb | 1.5.8 | **none** (PYSEC-2026-311) | RAG vector store; unfixable → ignore or drop |
| crawl4ai | 0.8.0 | 0.9.0 | crawler; major-ish, verify |
| cryptography | 46.0.5 | 48.0.1 | ✅ verified upgradable (see §4) |
| jwcrypto | 1.5.6 | 1.5.7 | patch |
| litellm | 1.82.0 | 1.84.0 | LLM router; verify |
| lxml | 5.4.0 | 6.1.0 | major; used widely (OOXML/HTML) — verify |
| msgpack | 1.1.2 | 1.2.1 | minor |
| nltk | 3.9.3 | 3.9.4 for some; **none** for GHSA-rf74-v2fm-23pw & PYSEC-2026-597 | partially unfixable → ignore residual |
| pillow | 12.1.1 | 12.2.0 | ✅ likely clean |
| pip | 25.2 | 26.1.2 | CVE-2026-3219 already ignored; others fixable by bumping pip in the audit step |
| pyasn1 | 0.6.2 | 0.6.3 | patch |
| pydantic-settings | 2.14.1 | 2.14.2 | patch |
| pyjwt | 2.11.0 | 2.13.0 | several PYSEC; verify |
| pyopenssl | 25.3.0 | 26.0.0 | major |
| pypdf | 6.7.5 | 6.13.3 | ✅ verified upgradable (see §4) |
| python-multipart | 0.0.22 | 0.0.31 | ✅ verified upgradable (see §4); direct-ish via fastapi extra |
| requests | 2.32.5 | 2.33.0 | minor |
| starlette | 0.52.1 | 1.3.1 | **major (0→1)**; ✅ verified works with fastapi 0.136 (see §4) |
| torch | 2.10.0 | **none** (CVE-2025-3000, PYSEC-2026-139) | unfixable → ignore or accept |
| transformers | 4.57.6 | 5.3.0 for some; **none** for PYSEC-2025-217 | major (4→5) + partially unfixable |
| urllib3 | 2.6.3 | 2.7.0 | minor |
| yt-dlp | 2026.3.17 | 2026.6.9 | was capped in targeted upgrades — check the parent constraint |

Regenerate the current list before starting (the DB changes daily):
```bash
uv run pip-audit --skip-editable --ignore-vuln CVE-2026-3219 2>/dev/null | sort -u
```

---

## 3. Constraints (do not violate)

- **Zero functional regression.** Unit suite (`uv run pytest tests/unit -q`, currently **193 passing** on merged main) must stay green after any change.
- **The unit tests do NOT cover the ML/RAG/crawl/PDF-render runtime paths.** A green suite is necessary, not sufficient. Before trusting a major bump (lxml 6, transformers 5, starlette 1, pyopenssl 26, crawl4ai 0.9), verify the affected runtime path actually works (import + a representative call), not just that pytest passes. At minimum: app import + a `starlette.testclient.TestClient` smoke on `backend.app.main` (hit `/healthz`, `/openapi.json`).
- **Transitive-only.** Bump the **direct parents** in `[project.dependencies]` where that pulls a fixed transitive version, or add `[tool.uv]` constraint/override entries (`constraint-dependencies` / `override-dependencies`) to force fixed transitive versions. Prefer bumping the real parent over a blunt override where feasible.
- **`--ignore-vuln` only for genuinely unfixable CVEs**, each with a one-line justification comment in `ci.yml` (mirroring the existing `CVE-2026-3219` comment). Do not ignore a CVE that has a fix.
- **Match the CI audit exactly** (`uv sync` base + the exact `pip-audit` flags) — don't accidentally "pass" by auditing a different env.
- Follow the established worktree + PR workflow: work in a fresh worktree off `origin/main`, never in the main checkout (another session uses it); push via the `projectremedyai` gh account (the default `johnnyrobot` lacks push access), restore the account after; commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## 4. Known-good facts already established (don't re-derive)

Verified on 2026-07-02 in a throwaway worktree:
- **The core web stack upgrades cleanly and the app works:** bumping `cryptography>=48.0.1`, `pypdf>=6.13.3`, `python-multipart>=0.0.31` (direct floors) + re-locking pulled `cryptography 49.0.0`, `pypdf 6.14.2`, `python-multipart 0.0.32`, `idna 3.18`, and **`starlette 0.52.1 → 1.3.1`** — all with the *current* `fastapi 0.136.0` (its constraint is `starlette>=0.46.0`, uncapped). **193/193 unit tests passed**, and `TestClient` on `backend.app.main` returned 200 for `/healthz`, `/docs`, `/openapi.json`. So the starlette 0→1 major bump is safe with the pinned fastapi.
- One caveat surfaced: starlette 1.x's `TestClient` emits `StarletteDeprecationWarning: install httpx2 instead` — harmless warning, but if the (local-only) test suites use `TestClient` heavily, consider it.
- A blanket `uv lock --upgrade` also swept in reportlab 5, rich 15, structlog 26, fastapi 0.139, uvicorn 0.49 — **out of scope / higher risk**; prefer targeted upgrades over a full refresh unless you verify those majors too.

---

## 5. Suggested approach (frugal-fable routing)

Keep judgment with Fable; delegate the mechanical/verifiable legs. Rough shape:

1. **Fable:** regenerate the current vuln list; classify each package into {clean-bump, major-bump-needs-verification, no-fix→ignore}. Decide the parent-bump vs `[tool.uv]` override strategy per package. (Judgment — keep it.)
2. **Sonnet (bounded, verifiable):** apply the clean bumps (§2 rows with a fix), re-lock targeted, run `pip-audit` + full unit suite, report the surviving findings to a file. Cheap, mechanical, gated on the audit + suite.
3. **Sonnet/Opus (higher stakes):** the major bumps (lxml 6, transformers 5 if pursued, pyopenssl 26, crawl4ai 0.9) — each with a runtime smoke of the path that uses it; Opus for any that touch parsing/security-critical code. Fable reviews these diffs.
4. **Fable:** decide the final `--ignore-vuln` set for the no-fix CVEs (chromadb PYSEC-2026-311, torch CVE-2025-3000 / PYSEC-2026-139, nltk GHSA-rf74-v2fm-23pw / PYSEC-2026-597, transformers PYSEC-2025-217, and any residual), each justified inline in `ci.yml`. Consider whether any no-fix dep (e.g. chromadb, if only a dev/optional RAG path) can be dropped from base deps instead of ignored.
5. **Verify against the real gate**, open a PR to `main`, confirm the `pip-audit` job goes green (or green-modulo-documented-ignores) on the PR before merging.

**Definition of done:** the CI `pip-audit` job passes on a PR to `main`; every surviving `--ignore-vuln` has a one-line justification; `uv run pytest tests/unit -q` is green (193+); app import + TestClient smoke green; no new runtime deps beyond version bumps; `pyproject.toml` diff reviewed.

---

## 6. Pointers

- CI: `.github/workflows/ci.yml` (job `security`).
- Direct deps: `pyproject.toml` `[project.dependencies]`; dev extras in `[project.optional-dependencies].dev`.
- Lockfile: `uv.lock` (re-lock with `uv lock` / `uv lock --upgrade-package <name>`; force transitive versions via `[tool.uv] constraint-dependencies`).
- The already-fixed sibling failure (`quality layer checks`) is PR #3 (merged) — pattern reference for a scoped CI fix.
