# PRD: Typst as an Accessible-PDF-Generation Backend

**Status:** Draft for review
**Owner:** remedy-server engineering
**Prepared:** 2026-07-01
**Repo:** `remedy-server` (`/Users/laccd/Desktop/lamc_district_forms/remedy-server`)
**Related:** `RESEARCH_remedy_server_ADA_refactor.md` (§5 Remediation-Level taxonomy, §7.1 component specs, §9/9.1 OSS-alternatives research)

---

## 0. Bottom line up front

A hands-on pilot this session (§9.1, item 4 of `RESEARCH_remedy_server_ADA_refactor.md`) found that **Typst v0.15.0's `--pdf-standard ua-1` compiler flag is a hard, build-blocking accessibility gate**, and that its output passed veraPDF's PDF/UA-1 profile at **106/106 rules, 2891/2891 checks, zero failures**, with struct-tree shapes (`H1-H3`, nested `Table/THead/TBody/TH/TD`, `Figure` with authored `/Alt`, `L/LI/Lbl/LBody`) that map closely onto this repo's existing `rebuild/ast.py` block model. That is a materially stronger authoring-time guarantee than the current QuestPDF sidecar provides on paper — but the "QuestPDF has no equivalent gate" comparison in §1.2 is an inference about `sidecar/QuestPdfRenderer`'s C# (`sidecar/QuestPdfRenderer/Renderer.cs`, `Ast.cs`), not a verified read of that code; only the Python-side `rebuild/sidecar.py` wrapper was read (see §12 for the scope of what was actually inspected). QuestPDF's accessibility is checked only *after* the fact by the existing `pdf_acceptance.py` veraPDF gate, which is verified.

**Recommendation: (a) parallel backend, selectable per job, not a replacement and not merely an experimental toggle.** Typst should ship as a second implementation of the same `RebuildRequest → bytes` contract that `QuestPdfSidecar` already defines, selected via a `rebuild.backend` config value (mirroring the existing `rebuild.markdown_parser` string-config pattern in `config.py`), defaulting initially to `questpdf` (current behavior unchanged) with `typst` available as an explicit opt-in per the phased rollout in §11. It graduates to default only after the open items in §9 are closed with evidence of the same rigor as the pilot. It does not replace QuestPDF outright because: (1) AcroForm/fillable-form tagging — directly relevant since many LAMC district forms are fillable — is completely untested; (2) the pilot is single-document, not corpus-representative; (3) the "manually-styled fake heading" caveat (§1.3, Caveat 2) proves that veraPDF-passing is necessary but not sufficient, so no backend should be trusted as a silent default swap without a generator-level semantic guarantee, which is new work either way.

**A load-bearing caveat on this PRD's own grounding, stated up front rather than buried:** §5–§7's technical design is grounded in the AST/schema/sidecar-wrapper layer (`rebuild/ast.py`, `rebuild/sidecar.py`, `markdown_parser.py`, `ast_builder.py`, `vision_enricher.py`, `config.py`) but the **actual orchestration code that would need to change**, `backend/app/engine_service.py::_rebuild_from_semantics()`, is now read and reflected below (§1.1a, §5.1a). Treat the FR/NFR sections as accurate to that orchestration reality, not as a design that "mirrors" a clean strategy-pattern call site that doesn't exist.

---

## 1. Problem statement

### 1.1 Current state (grounded in repo read, not speculation)

The existing "faithful rebuild" / rebuild-tier path in `remedy-server` is:

1. `rebuild/markdown_parser.py` parses the extractor's `ocr_markdown` into a list of AST `Block`s (`HeadingBlock`, `ParagraphBlock`, `ListBlock`, `SimpleTableBlock`, transient `ImagePlaceholder`) using `markdown-it-py`.
2. `rebuild/vision_enricher.py` classifies each extracted image as decorative or informative and drafts alt text via a vision-model call, returning `ImageSemantics(alt, decorative, confidence)`, with per-image failure absorbed (falls back to `decorative=True, alt=""`) and only a *total* failure (`VisionEnrichmentError`) raised.
3. `rebuild/ast_builder.py` fans in the parsed blocks + vision semantics into a validated `RebuildRequest` (Pydantic model in `rebuild/ast.py`), replacing `ImagePlaceholder` with `FigureBlock` (if not decorative and alt is non-empty) or `ArtifactBlock` (if decorative).
4. `rebuild/sidecar.py`'s `QuestPdfSidecar.render()` serializes the `RebuildRequest` to JSON, pipes it to a native-AOT .NET subprocess over **stdin**, and reads a PDF back over **stdout** (`%PDF` magic-byte check is the only structural validation on the way out).
5. Separately, `faithful_rebuild/pipeline.py` implements a **different**, page-content-preserving rebuild mode (MCID/BDC-EMC content-stream rewriting via pikepdf, not AST-driven) — this is a different tier from the AST/QuestPDF path and is out of scope for a Typst backend, which targets the same `RebuildRequest` contract as `QuestPdfSidecar`.
6. Whatever PDF the sidecar returns is then run through the existing `pdf_acceptance.py` veraPDF gate (`evaluate_pdf_acceptance` / `validate_with_verapdf`) as an **after-the-fact check**, not a build-time constraint. QuestPDF itself enforces nothing about alt text or heading semantics at render time; `FigureBlock.alt`'s Pydantic `min_length=1` constraint is the *only* upstream gate today, and it only guarantees a non-empty string, not that the string reached the renderer's tag structure correctly.

### 1.1a The actual production wiring (the orchestration layer, not just the schema/sidecar layer)

Steps 1–4 above are not invoked through any clean, swappable call site. The **sole caller** of `ast_builder.build`, `markdown_parser.parse`, `vision_enricher.enrich`, and `QuestPdfSidecar` anywhere in the codebase (verified via repo-wide grep — no other caller exists) is `backend/app/engine_service.py::_rebuild_from_semantics()`, a private, monolithic `async def` that:

- Is only reachable when the caller passes `allow_semantic_rebuild: bool = Form(False)` (`backend/app/routes.py:192`) **and** `cfg.rebuild.enabled` is true, and only runs as a **fallback after a primary fix-and-verify attempt** (`backend/app/engine_service.py:207,225`) — it is not a standalone, directly-invokable rebuild entry point today.
- Hardcodes `metadata_ast`, `page_settings`, and `conformance` inline (`title=input_path.stem`, `language="en-US"`, `pdfua="PDFUA_1"`, `backend/app/engine_service.py:571-576`) rather than threading them from any per-job configuration. Any `backend` selector this PRD introduces has to be plumbed through this same function, alongside these existing hardcoded values, not around them.
- Writes every failure into `store.update(job.id, status="failed", error=<literal string>)` using an established naming convention: `rebuild_extractor_failed`, `rebuild_extractor_empty`, `rebuild_vision_provider_unavailable`, `rebuild_vision_total_failure`, `rebuild_ast_invariant_violation`, `rebuild_sidecar_not_available`, `rebuild_sidecar_timeout` (all confirmed at their respective lines in `engine_service.py`). A Typst backend's failure modes (§5.2, §5.4) must slot into this exact convention — e.g. `rebuild_typst_compile_failed`, `rebuild_typst_unsupported_construct`, `rebuild_struct_assert_failed` — not invent a parallel error-reporting shape.

This changes how FR-1/FR-2 (§5.1) must be read: there is no existing strategy-pattern seam to "mirror" for backend selection at the AST/rebuild-tier entry point. Introducing `backend` selection means adding a branch inside (or a thin dispatch wrapping) `_rebuild_from_semantics`, which is a modification to a real, already-shipping, monolithic function — a materially different (and larger) change than swapping an argument at a clean call site.

### 1.2 The evidence that motivates a second backend

A hands-on pilot this session (not re-derived here — cited as given, see §12 for full detail) established:

- **Typst CLI v0.15.0** (installed via Homebrew), `typst compile --pdf-standard ua-1`, **hard-fails the build (exit 1, no PDF emitted)** when an `image()`/equation element is missing alt text. Verified both directions: failed without alt text, succeeded with it.
- The resulting PDF passed **veraPDF's PDF/UA-1 validation profile at 106/106 rules, 2891/2891 checks, zero failures** — a full pass, not partial.
- `pikepdf` struct-tree inspection confirmed **correct `<H1>/<H2>/<H3>`** (from real `=`/`==`/`===` heading syntax, not styled text), a **fully-nested `<Table>/<THead>/<TBody>/<TH>/<TD>`** with real header-row semantics, a **`<Figure>`** carrying the exact authored `/Alt` string, and correct **`<L>/<LI>/<Lbl>/<LBody>`** list structure.
- Root-level accessibility scaffolding — `MarkInfo/Marked`, `/Lang`, `ViewerPreferences/DisplayDocTitle`, XMP title/language — was set **automatically and correctly** with no extra authoring effort.

This is a categorically different guarantee than what QuestPDF gives today on the verification side: QuestPDF's sidecar's *only confirmed* verification is the downstream veraPDF pass/fail that already exists for all rebuild output. The stronger claim — "every tag is only as correct as `sidecar/QuestPdfRenderer`'s C# code makes it, with no compiler-level gate at all" — is a reasonable and likely-true inference given the stdin/stdout JSON contract observed in `rebuild/sidecar.py`, but it has **not** been confirmed by reading `sidecar/QuestPdfRenderer/Renderer.cs` or `Ast.cs` (both exist in the repo and were not opened for this PRD). Typst moves part of the verification burden into the authoring toolchain itself, for the one category (image/equation alt text) it covers; the exact absence of a comparable mechanism on the QuestPDF side is asserted, not read.

### 1.3 The two caveats that bound the opportunity (must not be glossed over)

- **Caveat 1 — narrow enforcement scope.** Typst's UA-1 enforcement is scoped **only** to `image()`/equation elements. It will **not** catch missing alt text on other figure-like content, e.g. vector/CeTZ-drawn diagrams. `FigureBlock.alt`'s existing `min_length=1` Pydantic constraint in `rebuild/ast.py` must remain the authoritative gate regardless of what Typst's compiler does or doesn't catch — the compiler-level check is a **bonus**, not a **replacement** for the AST-level contract.
- **Caveat 2 — veraPDF conformance is not semantic correctness.** A document using manually-styled fake headings (bold+sized text, literal `"1."`/`"2."` prefixes) instead of real heading/list syntax **still passed veraPDF's PDF/UA-1 profile**, but every "heading" was tagged plain `<P>` and the numeric prefixes were auto-absorbed into spurious `<L>/<LI>` structures. veraPDF validates **tag well-formedness and clause conformance**, not **semantic correctness of authoring**. This is the single most important design constraint for the code generator (§5.3): a generator that "looks right" visually but emits manually-styled text instead of `heading()`/`table()`/`list()` constructs would produce a PDF that is simultaneously veraPDF-conformant and accessibility-garbage — and nothing downstream would catch it without a dedicated post-generation check.

### 1.4 What is explicitly not yet known (do not overclaim)

Not tested in the pilot, and must be flagged as open/future work rather than assumed to work:
- Multi-column reading order (see §1.5 for the current, more nuanced state of this repo's own reading-order problem — it is not simply "unaddressed").
- AcroForm / fillable form-field tagging — relevant because many LAMC district forms **are** fillable forms.
- Behavior on a real, multi-page, corpus-representative document (the pilot was a single synthetic/test document, not a District form).

### 1.5 Reading-order risk: current state, not a stale snapshot

An earlier finding on this repo (a personal working note, not a committed repo artifact) characterized the "Major Sheets" flyer class as genuinely reading-order-scrambled. That finding is now **partially superseded by shipped work**: commit `702d6a2` ("Add reading-order reorder engine + Adobe/BT-ET repair + corpus hardening"), merged to `main`, added `vision_struct_reorder.py` (cross-parent struct `/K` reorder via a vision function, integrity-gated on struct-leaf-count preservation) and `content_stream_reorder.py` (re-sequences movable marked-content blocks to match logical structure order, render-pixel-diff gated), both wired into the engine's `fix_all` path. This is a **large, partially-solved problem**, not an untouched one.

This matters for the Typst backend's design, not just its framing: **these reorder passes are QuestPDF-tier, post-hoc, struct-tree-level engine passes that operate on already-rendered/tagged output.** A Typst rebuild, by construction, produces its struct tree directly from the AST at compile time — there is no existing integration point where `vision_struct_reorder.py`/`content_stream_reorder.py` could apply to Typst-compiled output without modification, since those passes were built against the QuestPDF-tier's PDF shape. This PRD does not resolve this design question; it surfaces it as Open Question 8 (§9) and as a named risk in §10, because a Typst rebuild path that silently forgoes this machinery would regress reading-order quality relative to the QuestPDF-tier's current (improved) baseline on exactly the document class most likely to need it.

---

## 2. Goals

1. Add Typst as a **second implementation of the `RebuildRequest → PDF bytes` contract**, selectable per job alongside the existing QuestPDF sidecar, without changing the AST schema (`rebuild/ast.py`) or the upstream parsing/vision/composition stages (`markdown_parser.py`, `vision_enricher.py`, `ast_builder.py`).
2. Use Typst's compiler-level `--pdf-standard ua-1` enforcement as an **additional, earlier gate** in the pipeline — catching missing image/equation alt text before the existing veraPDF acceptance gate runs, not instead of it.
3. Design the AST-to-Typst code generator so that it is structurally incapable of emitting the "manually-styled fake heading" failure mode identified in the pilot (§1.3, Caveat 2) — i.e., it must always emit `heading()`/`table()`/`list()`/`image()` constructs from typed AST nodes, never string-concatenated styled text.
4. Add a post-generation struct-tree assertion pass that independently verifies semantic tag correctness (not just veraPDF pass/fail) for any document built via the Typst backend, closing the gap that Caveat 2 proved veraPDF alone does not close.
5. Produce a decision framework for how a Typst-rebuilt document maps onto the L0–L5 / L-REPLACE Remediation-Level taxonomy (§5 of the ADA-refactor research), including explicit fallback conditions to QuestPDF or to in-place remediation.
6. Close the four open pilot/design gaps (multi-column reading order including the reorder-machinery question in §1.5, AcroForm, corpus-representative testing, and the orchestration-layer integration named in §1.1a) with the same evidentiary rigor as the initial pilot before Typst is trusted as anything beyond an explicit opt-in.

## 3. Non-goals

1. **Not** replacing the QuestPDF sidecar or removing `sidecar.py` / the .NET sidecar project. QuestPDF remains the default and the fallback for anything Typst cannot yet handle (AcroForm-bearing sources, until §7.4 is resolved).
2. **Not** replacing or modifying the separate MCID/BDC-EMC-based `faithful_rebuild/pipeline.py` "preserving" mode — that pipeline does not consume `RebuildRequest`/AST at all and is unaffected by this PRD.
3. **Not** attempting to make Typst render pixel-identical output to the source document. Both QuestPDF and Typst are semantic-rebuild backends (they reconstruct content from the AST, not from the original page geometry) — visual fidelity to the source is explicitly out of scope for either, consistent with how the existing rebuild tier is scoped today (visual-diff checking belongs to the separate `faithful_rebuild` "preserving" pipeline, not this AST-driven tier).
4. **Not** claiming WCAG 2.1 AA "Supports"-level conformance from Typst output alone. Per the ADA-refactor research's L0–L5 taxonomy, machine-verified PDF/UA-1 (whether from QuestPDF+veraPDF or Typst's own compiler gate+veraPDF) tops out at L3/L4 — human validation is still required for L5, regardless of backend.
5. **Not** in scope for this PRD: building the AcroForm/fillable-form-field extension of `rebuild/ast.py` itself. This PRD identifies the requirement and blocks Typst-backend eligibility on it (§7.4), but the AST schema extension is a separate, prerequisite workstream.
6. **Not** in scope for this PRD: rewriting or refactoring `backend/app/engine_service.py::_rebuild_from_semantics()` into a clean strategy-pattern dispatcher as a prerequisite. This PRD's FR-1/FR-2 (§5.1) describe the minimal branching needed inside that existing function; a broader refactor of the orchestration layer, while arguably good hygiene, is explicitly out of scope here and should be tracked separately if desired.

---

## 4. Fit with the Remediation-Level taxonomy and VPAT/ACR model

Per `RESEARCH_remedy_server_ADA_refactor.md` §5, every document is scored into L0–L5 or the L-REPLACE terminal state. This PRD's scope only touches the **rebuild tier** of the engine, which is one route into that ladder (the alternative routes are in-place remediation of an existing PDF, or Office-source remediation).

**Where a Typst-rebuilt document lands, and under what conditions:**

| Condition | Level reached | Rationale |
|---|---|---|
| Typst compile succeeds with `--pdf-standard ua-1`, output passes veraPDF PDF/UA-1 | **L3** (PDF/UA-1 machine-verified) | Matches the existing `fix_and_verify` → veraPDF-gate definition of L3 in §5 of the research doc. Typst's compiler gate is an *additional* pre-check, not a substitute for the veraPDF acceptance gate that already defines L3 in this repo — `pdf_acceptance.py`'s `evaluate_pdf_acceptance` must still run on Typst output exactly as it does on QuestPDF output. |
| L3, plus contrast/alt-text/heading-list-table structural checks pass (engine's deeper checks, same bar as QuestPDF path) | **L4** (WCAG 2.1 AA — engine-complete) | No different from the QuestPDF path's L4 definition; Typst does not change what "engine-complete" means, only how the PDF is produced. |
| L4, plus human sign-off on the ~45 non-machine-checkable Matterhorn conditions (meaningful alt text, correct reading order, table semantics) | **L5** | Identical human-review-queue requirement regardless of backend. Critically: **Caveat 2 (§1.3) means Typst cannot claim L5, or even a trustworthy L4, purely from its own compiler gate + veraPDF** — a document with fake manually-styled headings would still reach a veraPDF-passing L3/L4 by table definition, which is exactly why the struct-tree assertion pass (§6.4) is required as an additional non-negotiable gate specific to the Typst backend's generator, not a nice-to-have. |
| Source document is an AcroForm-bearing fillable form | **Falls back to QuestPDF, or to in-place remediation** | Neither backend has verified AcroForm support in this repo's rebuild tier; QuestPDF is the "existing, in-production" default so it is the safer fallback until either backend is verified (see §7.4). If the *source* PDF already has usable AcroForm fields worth preserving (as opposed to needing full semantic rebuild), in-place remediation via the existing `pdf_fixer.py`/`office_remediator.py` paths — which already have functions/handling touching AcroForm (`pdf_checker.py`, `pdf_fixer.py`, `faithful_rebuild/font_analysis.py`, `faithful_rebuild/simple_font_replacer.py`) — is likely a better fit than any semantic-rebuild backend, since neither rebuild backend currently regenerates form fields from the AST at all. |
| Source document has multi-column layout | **Route to QuestPDF (which now benefits from `vision_struct_reorder.py`/`content_stream_reorder.py`, per §1.5), or flag for human reading-order review before trusting a Typst-produced L3/L4** | Per §1.4/§1.5, Typst's multi-column reading-order behavior is untested, and this repo's post-hoc reorder machinery (shipped in commit `702d6a2`) has no known integration point for Typst-compiled output. Do not let an untested backend silently certify reading order on the exact document class this repo has already invested dedicated engineering effort to partially fix on the QuestPDF-tier path. |
| Document matches an L-REPLACE condition (confirmed accessible District original exists) | **L-REPLACE** | Unaffected by backend choice — this decision precedes rebuild entirely per §7.1 item D of the research doc. |

**VPAT/ACR implication:** exactly as with the QuestPDF path, a Typst-produced PDF at L3/L4 can only support an ACR conformance value of **"Partially Supports"** per §5.1 of the research doc; only L5 human validation moves it to **"Supports."** The backend used to produce the PDF is implementation detail invisible to the ACR — but the audit trail (§6.6) must record which backend and which gates were run, since that is exactly the kind of evidence a Title II audit or the *Payan v. LACCD* precedent (§1.5 of the research doc — the LACCD-specific case reference, distinct from this PRD's internal §1.5) would expect to see.

---

## 5. Functional requirements

### 5.1 Backend selection

- **FR-1.** `RebuildConfig` (in `config.py`) gains a `backend: Literal["questpdf", "typst"] = "questpdf"` field. Per the existing pattern for every other field in `RebuildConfig` (e.g. `markdown_parser`, loaded at `config.py:373-374` via an explicit `_env("REBUILD_MARKDOWN_PARSER", ...)` call inside the config-loader function), this requires **two** mechanical additions, not one: the dataclass field itself, and a corresponding explicit `_env("REBUILD_BACKEND", ...)` line in the loader function body — the field does not populate itself. Default remains QuestPDF; Typst is opt-in until promoted per the rollout in §11.
- **FR-2.** The renderer selection must be swappable per-job (not just per-deployment), to support A/B validation during the pilot-hardening phases in §11. Concretely, this means: (a) a job-level override parameter accepted alongside the existing `allow_semantic_rebuild` form flag in `backend/app/routes.py`, threaded through to `_rebuild_from_semantics()`; and (b) a branch inside `_rebuild_from_semantics()` (§1.1a) that selects `QuestPdfSidecar` vs. the new `TypstRenderer` (§5.2) at the point where `sidecar.render(request)` is currently called unconditionally. This is **not** a mirror of `faithful_rebuild.pipeline()`'s `force_mode` parameter — that function belongs to the separate, unaffected preserving-mode pipeline (§3, Non-goal 2) — it is new branching logic added directly to the monolithic rebuild-tier function identified in §1.1a.
- **FR-3.** Both backends must accept the identical `RebuildRequest` Pydantic model from `rebuild/ast.py` with **zero schema changes** required to switch backends, for any document that does not require AcroForm fields (see FR-6).

### 5.2 Typst sidecar / invocation shape

- **FR-4.** A new `TypstRenderer` (mirroring the shape of `QuestPdfSidecar` in `rebuild/sidecar.py`, likely `rebuild/typst_renderer.py`) must implement the same effective contract: `async def render(self, request: RebuildRequest) -> bytes`, so it is a drop-in alternative at the call site inside `_rebuild_from_semantics()` (§1.1a) where `sidecar.render(request)` is invoked today.
- **FR-5.** Unlike `QuestPdfSidecar`'s stdin/stdout JSON protocol, `TypstRenderer` must:
  1. Generate `.typ` source text from the `RebuildRequest` (the AST-to-Typst code generator, §5.3) and embed/reference image assets from the `assets` dict (resolving `AssetRef.path` to files Typst's `image()` function can read).
  2. Write the generated `.typ` file and any referenced images to a job-scoped temp directory (mirroring the existing `image_dir` convention already used in `ast_builder.py`'s asset resolution, and in `_rebuild_from_semantics()`'s own `image_dir = cfg.output.output_dir / "images" / doc_job.id[:12]` convention, §1.1a).
  3. Invoke `typst compile <input>.typ <output>.pdf --pdf-standard ua-1` as a subprocess (via `asyncio.create_subprocess_exec`, matching the existing async subprocess pattern in `sidecar.py`) with a configurable timeout (reuse `sidecar_timeout_s`).
  4. On exit code != 0, raise a `TypstCompileError` (mirroring `SidecarError`) carrying Typst's stderr diagnostic verbatim — this is the pilot-verified accessibility gate (missing alt text) surfacing as a **build failure with an actionable message**, not a silent degraded output. At the `_rebuild_from_semantics()` call site, this must be caught and written to `store.update(...)` following the existing `rebuild_*` string-error convention (§1.1a) — e.g. `error=f"rebuild_typst_compile_failed: {exc}"` — exactly parallel to how `SidecarError`/`SidecarTimeout` are handled today.
  5. On success, read the emitted `.pdf` file from disk and return its bytes (validating the `%PDF` magic-byte header exactly as `sidecar.py` does today, since Typst-produced files are read from disk rather than stdout but the same sanity check applies).
  6. Clean up the temp directory (or retain it behind a debug flag, matching whatever convention this repo uses for other subprocess-based intermediate artifacts).
- **FR-6.** `TypstRenderer` must raise a clear, typed "unsupported" error (`rebuild_typst_unsupported_construct`, per the FR-5.4 convention) — not attempt a best-effort render — if the `RebuildRequest` contains constructs Typst cannot yet handle safely: initially, any signal that the source requires AcroForm field regeneration (until §7.4 is resolved), and any explicit opt-out flag for multi-column layouts pending §7.3/§1.5.

### 5.3 AST-to-Typst code generator

- **FR-7.** The generator must be a pure function `RebuildRequest -> str` (Typst source text), symmetrical in shape to how `markdown_parser.parse` and `ast_builder.build` are pure functions today — no I/O beyond what's needed to resolve asset paths for `image()` calls.
- **FR-8.** Per-block mapping (must emit **real Typst semantic constructs**, never manual styling — this is the direct mitigation for Caveat 2, §1.3):

  | AST block (`rebuild/ast.py`) | Typst output | Notes |
  |---|---|---|
  | `HeadingBlock(level, runs)` | `= `/`== `/`=== ` ... (repeat `=` `level` times) followed by the concatenated run text | **Never** `#text(weight: "bold", size: ...)[...]`. Bold/italic `Run` flags within a heading (rare per `markdown_parser.py`'s own comment that images/links in headings "fall through silently") render as Typst `*...*`/`_..._` inline markup nested inside the heading line, not as a substitute for the heading marker itself. |
  | `ParagraphBlock(runs)` | Plain paragraph text, with `Run.bold`/`Run.italic` mapped to Typst `*...*` / `_..._` inline markup per run | Straightforward 1:1. |
  | `ListBlock(ordered, items)` | Typst `+ ` (ordered) or `- ` (unordered) list markup, recursively, for each `ListItem.body` | Typst's list syntax is itself the semantic list construct (compiles to `L/LI/Lbl/LBody`, verified in the pilot) — the generator must use `+`/`-` markup or the `list()`/`enum()` function form, **never** literal `"1. "`/`"•"` text prefixes emitted as plain paragraph runs (this is exactly the bug class Caveat 2 demonstrated: manually-authored numeric prefixes get mis-absorbed into spurious list structures by veraPDF-passing but semantically-wrong documents — the fix here is to make the generator the single source of truth for list semantics, never let a `ListItem.label_runs` string like `"1."` leak into body text as prose). |
  | `SimpleTableBlock(rows)` | Typst `table()` function call, with `table.header()` wrapping rows whose cells have `TableCell.header in {"col","both"}`, mapped to real Typst table syntax (verified in pilot to compile to nested `Table/THead/TBody/TH/TD`) | `TableCell.header == "row"` (row-header semantics) needs explicit handling — confirm in implementation whether Typst's `table.header()` construct or a `table.cell(..., header: true)`-per-cell approach is the correct mapping for row headers specifically, since the pilot verified column/header-row semantics but this AST also models row-header cells, which is a narrower case worth a dedicated unit test. |
  | `FigureBlock(asset_ref, alt, caption)` | Typst `#figure(image("<resolved-path>", alt: "<alt>"), caption: [...])` (or bare `#image(..., alt: "...")` if no caption) | This is the block that Typst's `--pdf-standard ua-1` compiler gate actually checks (per pilot: only `image()`/equation elements are enforced). `FigureBlock.alt`'s Pydantic `min_length=1` guarantees a non-empty string reaches the generator; the generator's job is simply to never drop it, and to fail loudly (not substitute a placeholder) if `alt` is somehow empty at generation time as defense-in-depth. |
  | `ArtifactBlock(asset_ref)` | Typst construct that Typst tags as an artifact/decorative image (needs implementation-time verification of the exact Typst API — e.g. whether Typst's PDF/UA export has a decorative-image annotation, or whether the generator must fall back to an untagged raw image placement outside any `figure()`/semantic wrapper) | **Open implementation question, not yet verified in the pilot** — see §9. |

- **FR-9.** The generator must never accept free-form/pre-styled text as an escape hatch for content that doesn't fit a Block type — every AST node must map to exactly one Typst construct from the table above, with no code path that stringifies a heading/list/table into plain styled text. This constraint is what makes the generator structurally incapable of the Caveat-2 failure mode, rather than merely trained/reviewed not to do it.

### 5.4 Struct-tree assertion pass (post-generation, pre-acceptance)

- **FR-10.** After `TypstRenderer.render()` produces a PDF, before it reaches the existing `pdf_acceptance.py` veraPDF gate, run a dedicated assertion pass (new module, e.g. `rebuild/struct_assert.py`) that opens the PDF with `pikepdf` (the same library already used for struct-tree inspection in the pilot and elsewhere in this repo, e.g. `faithful_rebuild/pipeline.py`) and independently verifies, **per input `RebuildRequest`**:
  - Every `HeadingBlock` in the request has a corresponding `H1`-`H6` struct element in the output struct tree (not `P`).
  - Every `SimpleTableBlock` produced a `Table` element containing `TR`/`TH`/`TD` (and `THead`/`TBody` if the generator emits them), with header cells present where `TableCell.header != "none"`.
  - Every `ListBlock` produced an `L` element containing `LI`/`Lbl`/`LBody`, and that no separately-authored `P` element in the output contains a leading numeral/bullet character pattern that should have been list markup (a direct, automatable check for the exact Caveat-2 failure signature).
  - Every `FigureBlock.asset_ref` maps to a `Figure` struct element whose `/Alt` string matches (or is a superset/paraphrase check TBD, see §9) the AST's `alt` field.
- **FR-11.** A failure of this assertion pass must be treated as a **generator bug**, not a document-specific quirk — it should hard-fail the job (`error="rebuild_struct_assert_failed"`, per the §1.1a convention) and alert engineering (not silently degrade to unstructured output), because by definition it means the generator produced Typst source that "looks right" but tagged wrong, exactly the failure mode this whole gate exists to catch.
- **FR-12.** This pass is Typst-backend-specific in this PRD's initial scope (because it's the newly-introduced risk surface) but should be evaluated for retrofitting onto the QuestPDF path too, since nothing today actually proves QuestPDF's sidecar tags headings/lists/tables correctly beyond veraPDF's tag-well-formedness check — Caveat 2 is a discovery about *validators*, not about Typst specifically, and the existing QuestPDF path has the identical blind spot (a blind spot compounded by the fact that `sidecar/QuestPdfRenderer`'s actual tagging code, per §1.2, has not itself been read for this PRD).

### 5.5 AcroForm handling (explicitly unresolved — functional requirement is to gate, not to solve)

- **FR-13.** Neither backend's `RebuildRequest` schema (`rebuild/ast.py`) currently models form fields at all — there is no `FormFieldBlock` or equivalent. This PRD does **not** require building one. It requires:
  1. A reliable **pre-flight detection** step (before rebuild is attempted) that identifies whether a source document has AcroForm fields (pikepdf can read `/AcroForm` from the source, matching how `pdf_checker.py`/`pdf_fixer.py` already inspect AcroForm today).
  2. If AcroForm fields are detected, the rebuild-tier pipeline (regardless of backend) should **not** silently drop them — until an AST extension + generator mapping for form fields exists, AcroForm-bearing sources should be routed away from the AST rebuild tier entirely (to in-place remediation via the existing `pdf_fixer.py` path, or to the `faithful_rebuild/pipeline.py` "preserving" mode, which operates on the original content stream and could in principle carry AcroForm dictionaries through more naturally than a from-scratch AST rebuild).
  3. This detection-and-route logic is a **shared** requirement for both backends (QuestPDF today already has this gap silently — it's not Typst-specific), but since this PRD is introducing new pipeline branching for backend selection anyway (directly inside `_rebuild_from_semantics()`, per §1.1a), this is the natural point to also close this pre-existing gap.

---

## 6. Non-functional requirements

- **NFR-1 (Determinism).** Given the same `RebuildRequest`, the Typst generator + `typst compile` must produce byte-stable or at least struct-tree-stable output across runs (Typst is a from-source compiler, so this should hold more naturally than QuestPDF's imperative rendering, but must be verified, not assumed).
- **NFR-2 (Latency).** `typst compile` is a single-process, single-pass compile (no JIT warmup like a long-lived .NET AOT process, but also no persistent process to amortize startup cost across jobs like the sidecar could in principle be kept warm). Benchmark cold-start latency per document against the existing `sidecar_timeout_s: 120.0` default and adjust a `typst_timeout_s` config value accordingly rather than assuming parity.
- **NFR-3 (Dependency footprint).** Typst CLI is a single static-ish Rust binary (installed via Homebrew in the pilot); document the equivalent install path for the Docker build (README.md already documents that the current image build includes veraPDF + the QuestPDF sidecar + Playwright/Chromium and takes ~10 minutes — adding Typst should be a small, auditable addition to that same Dockerfile stage, not a new build system).
- **NFR-4 (Failure transparency).** Per FR-4/FR-5, `TypstCompileError` must surface Typst's own stderr diagnostic (which, per the pilot, is specific enough to say alt text is missing) rather than a generic "render failed" — this is a genuine advantage over QuestPDF's current failure mode (a .NET exception translated into opaque JSON on stderr) and should be preserved end-to-end into the `rebuild_typst_compile_failed` error string written via `store.update()` (§1.1a, FR-5.4), matching how `rebuild_sidecar_timeout`/other `rebuild_*` errors already carry the underlying exception text today.
- **NFR-5 (No new attack surface).** Typst's `--pdf-standard ua-1` flag and general compile invocation must be run with the same subprocess-sandboxing posture as the existing sidecar (no shell interpolation of untrusted content into the command line — this repo already uses `asyncio.create_subprocess_exec` with an argv list rather than a shell string in `sidecar.py`; the Typst invocation must follow the identical pattern).
- **NFR-6 (Auditability).** Every rebuild job's audit record (per §7.1 item G of the research doc's "conformance dashboard + audit ledger" concept) must capture which backend (`questpdf` or `typst`) produced the output, the Typst CLI version used, and whether the struct-tree assertion pass (§6.4/FR-10) passed — this is exactly the kind of evidence a Title II audit or the *Payan v. LACCD* precedent would expect (per RESEARCH doc §1.5, "the audit trail is itself the legal product").

---

## 7. Technical design summary (see §5 for full detail; this section is the at-a-glance architecture)

```
allow_semantic_rebuild=True (Form flag, backend/app/routes.py:192)
  + cfg.rebuild.enabled
        │
        ▼
_rebuild_from_semantics()  (backend/app/engine_service.py — EXISTING, MONOLITHIC,
                             the only real caller of the stages below; runs only
                             as a fallback after a primary fix-and-verify attempt)
        │
        ├─ step 1: ContentExtractor.extract() -> ocr_markdown
        ├─ step 2 (parallel fan-in): markdown_parser.parse() + vision_enricher.enrich()
        ├─ step 2b: metadata_ast/page_settings/conformance (currently HARDCODED inline,
        │           §1.1a — a new `backend` selector must thread alongside these)
        ├─ step 2c: ast_builder.build(...) -> RebuildRequest (rebuild/ast.py — UNCHANGED)
        │
        ├──[backend=questpdf, default]──> QuestPdfSidecar.render()  (existing, unchanged)
        │                                   stdin JSON → .NET AOT subprocess → stdout PDF bytes
        │
        └──[backend=typst, NEW branch]──> TypstRenderer.render()  (NEW)
                                            1. typst_generator.generate(request) -> .typ source (NEW, pure fn)
                                            2. write .typ + resolved image assets to job temp dir
                                            3. subprocess: typst compile in.typ out.pdf --pdf-standard ua-1
                                            4. on exit!=0 -> TypstCompileError(stderr)
                                               -> store.update(error=f"rebuild_typst_compile_failed: {exc}")
                                               [[alt-text gate fires here]]
                                            5. read out.pdf bytes, validate %PDF header
                                                   │
                                                   ▼
                                          struct_assert.verify(request, pdf_bytes)  (NEW)
                                            independently checks H1-H6 / Table / L-LI / Figure+Alt
                                            against the *input* AST, not just veraPDF's tag well-formedness
                                            on failure -> store.update(error="rebuild_struct_assert_failed")
                                                   │
                                                   ▼ (both backends converge here)
                                  pdf_acceptance.evaluate_pdf_acceptance()  (existing, unchanged)
                                            veraPDF PDF/UA-1 + checker + screen-reader sim
                                                   │
                                                   ▼
                                  Remediation-Level assignment (L3/L4, per §4 table)
```

Key design point: **the struct-tree assertion pass sits between backend-specific rendering and the shared veraPDF acceptance gate**, so it applies only once regardless of backend, and both backends converge on the identical, unmodified downstream acceptance/level-assignment logic. The blast radius of adding Typst is a new renderer + generator + assertion module, **plus a new conditional branch and new `rebuild_*` error cases inside the existing `_rebuild_from_semantics()` function** (§1.1a) — with zero changes to `rebuild/ast.py`, `markdown_parser.py`, `vision_enricher.py`, `ast_builder.py`, or `pdf_acceptance.py`. This is a narrower blast radius than a full orchestration rewrite, but it is not a zero-touch addition to the orchestration layer, and the FR/NFR sections above should be read with that correction in mind.

---

## 8. Success metrics / acceptance criteria (machine-checkable where possible)

| # | Criterion | How measured |
|---|---|---|
| 1 | Typst backend produces a veraPDF PDF/UA-1 pass (106/106 rules) on **every** document in a curated corpus-representative test set (not just the pilot's single document) | Automated: run existing veraPDF integration in CI against N sample District/campus forms rebuilt via Typst backend |
| 2 | Struct-tree assertion pass (§6.4) reports zero mismatches between input AST and output struct tree across the same test set | Automated: `struct_assert.py` unit + integration tests |
| 3 | Zero instances of the "manually-styled fake heading" pattern (bold/sized plain-`P` text where a `HeadingBlock` was in the input) in any Typst-backend output | Automated, part of struct-tree assertion (FR-10) |
| 4 | Missing alt text on any `image()`/equation Typst construct causes a hard build failure (exit != 0), verified via a negative test case in CI | Automated regression test mirroring the pilot's manual verification |
| 5 | AcroForm-bearing source documents are never silently routed into the Typst (or QuestPDF) AST rebuild tier without the field data being explicitly accounted for | Automated: pre-flight detection test (FR-13) with a fixture fillable-form PDF |
| 6 | Multi-column source documents are either explicitly excluded from the Typst backend or pass a human-reviewed reading-order check before being marked eligible | Manual/human-review-gated (§4 table), tracked in the review queue per RESEARCH doc §7.1 item E |
| 7 | Latency: p95 Typst compile+generate time for a representative multi-page District form is within an agreed threshold of the existing `sidecar_timeout_s` budget | Automated benchmark in CI |
| 8 | Every rebuild job's audit record captures backend identity + Typst CLI version + struct-assert pass/fail (NFR-6) | Automated: audit-ledger schema test |
| 9 | `_rebuild_from_semantics()` correctly dispatches to `TypstRenderer` vs. `QuestPdfSidecar` per the `backend` selector, and every new failure path writes a `rebuild_typst_*` error string via `store.update()` consistent with the existing `rebuild_*` convention (§1.1a) | Automated: integration test exercising both backends through the real orchestration function, not a mocked call site |

---

## 9. Open questions requiring a human decision

1. **Row-header (`TableCell.header == "row"`) mapping** — does Typst's `table.header()` construct support row headers directly, or does the generator need per-cell `table.cell(header: true)` handling? The pilot verified column/header-row semantics but not row-header cells specifically. Needs an implementation-time spike before FR-8's table mapping is finalized.
2. **`ArtifactBlock` → Typst mapping** — the pilot did not verify how (or whether) Typst's PDF/UA export path supports marking an image as a decorative artifact (as opposed to a tagged `Figure`). If Typst has no first-class "artifact" concept, the generator may need to place `ArtifactBlock` images outside any semantic wrapper and verify via the struct-tree assertion pass that they land as `Artifact` (not `Figure` with empty alt, which would be worse than the status quo). This needs a dedicated pilot before FR-8 is implemented for this block type.
3. **Struct-tree assertion pass strictness for `/Alt` text matching** — should the assertion require byte-exact match between `FigureBlock.alt` and the compiled PDF's `/Alt` string, or allow for Typst-internal whitespace/escaping normalization? Needs a decision once the generator's actual escaping behavior is characterized.
4. **Promotion criteria from "opt-in" to "default"** — who signs off that the pilot gaps (§1.4/§1.5) are sufficiently closed, and what's the bar (e.g., N corpus-representative documents at L4 with zero struct-assert failures, human-reviewed reading order on M multi-column samples)? This is a product/engineering-leadership decision, not a purely technical one, and should probably be tied to a Phase gate in §11 rather than left ambiguous.
5. **AcroForm AST extension ownership and timeline** — building a `FormFieldBlock` (or equivalent) into `rebuild/ast.py` is a prerequisite for either backend to responsibly handle LAMC's fillable forms via the semantic-rebuild tier. Is this in scope for this initiative, or a separate, prerequisite workstream that this PRD should simply depend on and block on?
6. **Should the struct-tree assertion pass (§6.4) be retrofitted onto the existing QuestPDF path?** Per FR-12, Caveat 2 is a discovery about validators in general, not about Typst specifically — the QuestPDF path has an identical unverified assumption today (compounded by the fact that `sidecar/QuestPdfRenderer/Renderer.cs`'s actual tagging logic has not been read as part of this PRD, per §1.2). Worth a deliberate decision on priority/sequencing rather than silently deferring indefinitely.
7. **Multi-validator reconciliation** — per RESEARCH doc §4/§7.1 item H, no single validator is authoritative (~50% cross-tool disagreement rate in published studies). Should Typst-backend output additionally be run through PAC/Adobe Preflight before being trusted at L3/L4, matching the multi-validator reconciliation principle already recommended for the engine generally? This PRD scopes only the veraPDF gate (matching current repo behavior) but the question of whether Typst output specifically warrants extra scrutiny during the hardening phase is open.
8. **Reading-order reorder-machinery parity** — `vision_struct_reorder.py`/`content_stream_reorder.py` (shipped in commit `702d6a2`) currently improve reading order on the QuestPDF-tier's post-render struct tree. Should a Typst backend (a) replicate equivalent reorder logic against Typst-compiled struct trees, (b) explicitly forgo it and rely entirely on generator-time ordering being correct from the AST, or (c) route any document at meaningful reading-order risk away from Typst until (a) is built? This is a design decision this PRD surfaces (§1.5, §4) but does not resolve, and it directly affects whether Typst can be trusted on the multi-column document class without regressing relative to the current QuestPDF-tier baseline.
9. **Does the README's `/v1/pdf/rebuild` route need companion documentation for the AST/Typst tier?** README documents `/v1/pdf/rebuild` (`force_mode=preserving|mode_a|mode_b|simple_font`) as the entry point to the *separate* `faithful_rebuild` pipeline (§3, Non-goal 2), not the AST/QuestPDF/Typst tier this PRD scopes — the AST tier is reached only via the `allow_semantic_rebuild` flag on whatever route calls `_rebuild_from_semantics()` (§1.1a), which is undocumented in README today. Should this PRD's implementation phase also add README coverage for that entry point, independent of the Typst work itself?

---

## 10. Risks and mitigations (specific to this project's findings, not generic)

| Risk | Specific evidence it's grounded in | Mitigation |
|---|---|---|
| Generator emits visually-plausible but semantically-wrong Typst source (the Caveat-2 failure mode) | Directly observed in this session's pilot: manually-styled headings/lists passed veraPDF while being tagged wrong | FR-9 (generator structurally cannot stringify semantic blocks) + FR-10/FR-11 (mandatory struct-tree assertion pass that hard-fails the job, not just logs a warning) |
| Typst's alt-text enforcement is narrower than assumed, giving false confidence | Directly observed: enforcement is scoped only to `image()`/equation, not vector/CeTZ figures | `FigureBlock.alt`'s existing `min_length=1` Pydantic gate remains authoritative regardless of backend (§1.3); never treat "Typst compiled" as proof that all figures have alt text |
| AcroForm fields silently dropped when a fillable LAMC district form is routed through the AST rebuild tier | Neither backend's AST models form fields at all (confirmed by reading `rebuild/ast.py` — no `FormFieldBlock`); many LAMC forms are fillable | FR-13 pre-flight detection + routing away from AST rebuild tier until a form-field AST extension exists (§9, open question 5) |
| Multi-column reading order silently mis-ordered or content silently dropped by the Typst backend specifically, while the QuestPDF-tier baseline has already improved on this exact problem | Not tested in the Typst pilot; this repo has already shipped `vision_struct_reorder.py`/`content_stream_reorder.py` (commit `702d6a2`) that measurably improve QuestPDF-tier reading order on the "Major Sheets" multi-column class, with no known equivalent for Typst-compiled struct trees (§1.5); and a separate pilot this session found a different tool (olmOCR-2) **silently dropped entire sidebar columns** on that exact document class with no ambiguity flag despite being prompted to flag one | Explicit exclusion/human-review gate for multi-column sources until a dedicated Typst reading-order pilot is run (§4 table, §8 metric 6, §9 open question 8); do not assume Typst is safe on this class merely because it wasn't observed to fail — it was never tested on it, and the bar it must clear is now the *improved* QuestPDF-tier baseline, not the original scrambled one |
| Typst backend is quietly promoted to default before the open gaps are closed, because it "worked" on the pilot document | The pilot was one synthetic/test document; the research doc's own §4 principle ("a passing automated-tool report is necessary but not sufficient") applies here too | §9 open question 4 requires an explicit, named human sign-off gate before default-promotion; §11 phased rollout keeps Typst opt-in through at least two phases |
| Struct-tree assertion pass itself becomes a rubber stamp (checks presence of tags but not meaningful correctness, repeating the veraPDF blind spot one layer up) | Directly informed by Caveat 2: the lesson is that *any* automated tag-presence check can be satisfied by a generator that's technically correct but semantically hollow | Assertion pass must check **round-trip fidelity to the specific input AST** (e.g., "this exact `HeadingBlock` produced an `H` element with this exact level"), not just "an `H1` exists somewhere" — scope this precisely during implementation, and keep it in scope for the same human-review-queue skepticism the research doc applies to LLM-emitted verdicts (RESEARCH doc §4) |
| Cross-tool validator disagreement understates real defects | RESEARCH doc §4 cites a published 155-PDF study finding ~50.3% cross-tool disagreement between veraPDF/PAC/Acrobat/CommonLook | Track as open question 7; do not let a Typst-backend veraPDF pass alone override the same multi-validator-reconciliation posture recommended for the engine as a whole |
| Implementation underestimates effort because the design targets a clean call site that doesn't exist | `_rebuild_from_semantics()` (`backend/app/engine_service.py`) is a monolithic, private async function with hardcoded metadata/page/conformance values and no existing strategy-pattern seam for backend selection (§1.1a) | FR-1/FR-2/§7 now describe the actual branch-inside-the-function change required; scope estimates and code review for this PRD's implementation phase should treat this as a modification to shipping orchestration code, not an isolated new-module addition |

---

## 11. Phased rollout (hard constraint: April 26, 2027 LACCD ADA deadline)

Sequenced to sit inside the existing roadmap in §7.2 of the research doc (Phase 0 baseline through Phase 5 monitoring), as an enhancement to the rebuild tier rather than a parallel program:

- **Phase T0 — Harden the pilot (target: align with research doc's Phase 0/1, i.e., early in the burndown, well before bulk auto-remediation depends on it).**
  Close the open gaps from §1.4/§1.5: run the Typst pilot against (a) a multi-column District/campus document, (b) an AcroForm-bearing fillable form (expect failure/gap — document it precisely), (c) a real, multi-page, corpus-representative document, and (d) confirm the reading-order reorder-machinery question (§9, open question 8). Build the AST-to-Typst generator (§5.3), the struct-tree assertion pass (§6.4), and the actual dispatch branch inside `_rebuild_from_semantics()` (§1.1a, §5.1) as the real deliverables of this phase, not just more pilots. **Gate to exit this phase:** success metrics 1–4 and 9 in §8 pass on a curated multi-document test set (not just the original single pilot doc).

- **Phase T1 — Opt-in parallel backend (target: aligns with research doc's Phase 2, bulk auto-remediation).**
  Ship `rebuild.backend="typst"` as an explicit, job-level opt-in config (FR-1/FR-2), default remains `questpdf`. Run both backends side-by-side on a sample of the corpus for comparison (veraPDF pass rate, struct-assert pass rate, latency) without affecting production default behavior. AcroForm-bearing and multi-column sources remain hard-routed away from Typst (and, per FR-13, ideally away from naive AST-rebuild entirely) regardless of this phase's outcome.

- **Phase T2 — Conditional promotion (target: aligns with research doc's Phase 3, Office-source parity, since by then the corpus triage signal for "which documents are Typst-eligible" should be mature).**
  If Phase T1's comparison data clears the bar defined in §9's open question 4 (human sign-off, not automatic), promote Typst to the **preferred** backend for non-AcroForm, non-multi-column, born-digital sources — while QuestPDF remains available and is the automatic fallback for anything that trips FR-6's "unsupported construct" gate. This is explicitly **not** a full replacement (§0): both backends stay in the codebase indefinitely, selected by document eligibility, mirroring the "replace-vs-remediate" branching philosophy already established for the corpus orchestrator (RESEARCH doc §7.1 items C/D).

- **Phase T3 — Monitoring and regression (target: aligns with research doc's Phase 5, continuous re-validation, i.e., ongoing past 4/26/2027).**
  Both backends' output continues to flow through the unchanged, shared veraPDF acceptance gate and (per NFR-6) an audit ledger that records backend identity per document — so a future audit or regression investigation can always answer "which backend produced this, and did the struct-assert pass run and pass" for any document produced after this initiative ships.

**Deadline framing:** because Typst is additive at the schema/renderer/generator/assertion level (zero changes to the existing AST/parsing/vision/acceptance code, per §7), and its integration into the orchestration layer is a scoped, reviewable branch inside one existing function (§1.1a) rather than a rewrite of that function, it carries low risk of regressing the QuestPDF path's contribution to the 4/26/2027 burndown even if Phase T0's hardening takes longer than planned — worst case, Typst simply never exits opt-in status and the existing QuestPDF-only rebuild tier continues carrying its current share of the corpus unaffected.

---

## 12. Appendix — Evidence citations (this session's pilot + repo reads)

**Pilot evidence (cited as given per task instructions; not re-derived).** Note on evidentiary weight: the following bullets are traced precisely to RESEARCH doc §9.1, item 4, and that citation discipline is solid — but there is no artifact in *this* repo (no `.typ` file, no compiled PDF, no veraPDF XML report, no struct-tree dump) to independently re-check any of it against. Treat this section as an accurate transcription of a secondhand pilot narrative, not as independently reproduced evidence:
- Typst CLI v0.15.0, installed via Homebrew.
- `typst compile --pdf-standard ua-1` hard-fails (exit 1, no PDF emitted) on missing image/equation alt text — verified both failing (no alt) and succeeding (with alt) paths.
- Output passed veraPDF PDF/UA-1 validation profile: **106/106 rules, 2891/2891 checks, zero failures**.
- `pikepdf` struct-tree inspection confirmed: correct `<H1>/<H2>/<H3>` from real heading syntax; fully-nested `<Table>/<THead>/<TBody>/<TH>/<TD>` with real header-row semantics; `<Figure>` carrying the exact authored `/Alt` string; correct `<L>/<LI>/<Lbl>/<LBody>` list structure.
- Root-level scaffolding (MarkInfo/Marked, `/Lang`, ViewerPreferences/DisplayDocTitle, XMP title/language) set automatically and correctly.
- Caveat 1: UA-1 enforcement scoped only to `image()`/equation elements, not vector/CeTZ figures.
- Caveat 2: manually-styled fake headings (bold+sized text, literal "1."/"2." prefixes) still passed veraPDF PDF/UA-1, with headings tagged plain `<P>` and numeric prefixes auto-absorbed into spurious `<L>/<LI>` structures — proving veraPDF validates tag well-formedness/clause conformance, not semantic correctness of authoring.
- Not yet tested: multi-column reading order, AcroForm form-field tagging, behavior on a real multi-page corpus-representative document.
- Source: `RESEARCH_remedy_server_ADA_refactor.md` §9.1, item 4 ("Typst PDF/UA-1 pilot — strong positive result"). Note: this PRD's granular per-block generator mapping (FR-8), including the row-header question (§9, open question 1), extrapolates beyond what the pilot itself verified — flagged accordingly at each such point in FR-8 and in the open questions, not asserted as pilot-confirmed.

**Corroborating, independently-documented risk (same research doc, same session, different pilot):** §9.1, item 2 (olmOCR-2 pilot) found that on the exact multi-column "Major Sheets" document class already flagged in this repo's prior work as reading-order-sensitive, a different tool **silently dropped both sidebar columns entirely** with no ambiguity flag despite being explicitly prompted to raise one — cited here only to support the risk framing in §10 that "untested on multi-column" should not be read as "probably fine," since a sibling tool in the same research pass failed silently and severely on that exact document class.

**Repo files read to ground the technical design (§5–§7), including the orchestration layer:**
- `backend/app/engine_service.py` — `_rebuild_from_semantics()` in full: confirmed it is the sole caller of `ast_builder.build`, `markdown_parser.parse`, `vision_enricher.enrich`, and `QuestPdfSidecar`; confirmed the hardcoded `metadata_ast`/`page_settings`/`conformance` construction (`title=input_path.stem`, `language="en-US"`, `pdfua="PDFUA_1"`); confirmed the full `rebuild_*` literal error-string convention (`rebuild_extractor_failed`, `rebuild_extractor_empty`, `rebuild_vision_provider_unavailable`, `rebuild_vision_total_failure`, `rebuild_ast_invariant_violation`, `rebuild_sidecar_not_available`, `rebuild_sidecar_timeout`) written via `store.update(job.id, status="failed", error=...)`.
- `backend/app/routes.py` — confirmed `allow_semantic_rebuild: bool = Form(False)` (line 192) as the gate that makes `_rebuild_from_semantics()` reachable at all, and that it fires only as a fallback after a primary fix-and-verify attempt (`engine_service.py:207,225`).
- `src/project_remedy/rebuild/ast.py` — `RebuildRequest`, `HeadingBlock`, `ParagraphBlock`, `ListBlock`/`ListItem`, `SimpleTableBlock`/`TableRow`/`TableCell`, `FigureBlock`, `ArtifactBlock`, `Metadata`, `Conformance`, `PageSettings`, `AssetRef`, and the two `model_validator`s (`_pdfua_requires_language`, `_asset_refs_resolve`).
- `src/project_remedy/rebuild/sidecar.py` — `QuestPdfSidecar`, its `render()` method, `SidecarError`/`SidecarTimeout`, the stdin/stdout JSON + `%PDF` magic-byte protocol. (Note: `sidecar/QuestPdfRenderer/Renderer.cs` and `Ast.cs`, the actual C# tagging implementation this Python wrapper calls, were **not** read — the "QuestPDF has no compiler-level accessibility gate" claim in §1.2/§0 is an inference from the wrapper's I/O contract, not a verified read of the renderer's tagging logic. Flagged here rather than silently treated as equivalent-strength evidence to the repo reads above.)
- `src/project_remedy/rebuild/markdown_parser.py` — `parse()`, `ImagePlaceholder`, block-level walkers for headings/paragraphs/lists/tables.
- `src/project_remedy/rebuild/vision_enricher.py` — `enrich()`, `ImageSemantics`, per-image failure absorption, `VisionEnrichmentError` on total failure.
- `src/project_remedy/rebuild/ast_builder.py` — `build()`, `_substitute()`, `ASTBuildError`, asset-ref resolution via `image_dir`.
- `src/project_remedy/faithful_rebuild/pipeline.py` — the separate MCID/BDC-EMC content-preserving rebuild mode (confirmed distinct from, and unaffected by, the AST/QuestPDF/Typst rebuild tier this PRD scopes). Note: this file (and its `force_mode` parameter) is unrelated to the AST-tier's actual dispatch point; earlier drafts of this PRD imprecisely suggested FR-2 would "mirror" `force_mode`'s call path — corrected in §5.1/§1.1a to describe the real, different call site.
- `src/project_remedy/config.py` — `RebuildConfig` (existing `enabled`, `text_similarity_threshold`, `vision_concurrency`, `sidecar_timeout_s`, `markdown_parser` fields), and the explicit per-field `_env("REBUILD_MARKDOWN_PARSER", ...)` loader call at `config.py:373-374`, confirmed as the two-part (field + loader-line) pattern the new `backend` field must follow.
- `src/project_remedy/pdf_acceptance.py` — confirmed existence of `evaluate_pdf_acceptance`/`validate_with_verapdf` as the shared, backend-agnostic downstream veraPDF gate both backends must continue to pass through unchanged.
- `README.md` — confirmed the `/v1/pdf/rebuild` API route (`force_mode=preserving|mode_a|mode_b|simple_font`) documents the **separate** `faithful_rebuild` preserving-mode pipeline (§3, Non-goal 2), not the AST/QuestPDF/Typst tier this PRD scopes — the AST tier's actual entry point (`allow_semantic_rebuild`, §1.1a) is not documented in README today (see §9, open question 9). Also confirmed the existing Docker build description (veraPDF + QuestPDF sidecar + Playwright/Chromium, ~10 min build) as the baseline for NFR-3's dependency-footprint framing.
- `src/project_remedy/content_stream_reorder.py`, `src/project_remedy/vision_struct_reorder.py` — confirmed both exist and are wired into `fix_all`, per commit `702d6a2` ("Add reading-order reorder engine + Adobe/BT-ET repair + corpus hardening"), superseding the flatly-scrambled framing of the reading-order risk (§1.5).
- Confirmed via `grep`: **no existing Typst references anywhere in the repo** (`.py`/`.md`/`.cs`) — this is genuinely new work, not an extension of prior art in this codebase.
- Confirmed via `grep`: `RESEARCH_remedy_server_ADA_refactor.md` §5 (Remediation-Level taxonomy, L0–L5 + L-REPLACE), §5.1 (VPAT 2.x ACR mapping), §7.1 (component specs, a single heading containing bolded-letter list items A–H — not addressable `§7.1.A`-style subsections; this PRD cites them as "§7.1 item X" accordingly, not as numbered subheadings), §9/9.1 (OSS-alternatives research and the Typst pilot itself) — used for §4's taxonomy mapping and framing throughout, without contradiction.
