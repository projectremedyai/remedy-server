# Move 3 — contrast & reading_order task-input redesign (design, 2026-07-17)

## Why (eval evidence, commit 7b84f08)
- **contrast** adapter collapsed to ALWAYS-FAIL (54.2% status acc; 11/11 gold-pass
  rows predicted fail). The prompt asks the model to judge WCAG ratios (4.5:1 /
  3.0:1) from a rendered PNG with **zero numeric input**, while the verifier gold
  is computed from exact ratios. Distinguishing 4.4:1 from 4.6:1 visually is not
  a learnable task; the model rationally defaults to "fail".
- **reading_order** adapter collapsed to ALWAYS-PASS (51.7%; missed 29/30 gold
  fails). Its structure list contains **tag names only** (`/TD`, `/LI`, ...) —
  no text snippets, no bboxes — so entries cannot be correlated with the page
  image at all; "order matches layout" is undecidable from the given inputs.
- Counter-evidence that input enrichment works: heading_hierarchy's prompt was
  enriched with MCID text in the main workstream (remedy-server `61e5e26`) and
  it is the near-passing task (86.6%).

## Redesign

### contrast → "measured-candidates judgment" prompt
The remediation engine already computes fg/bg colors and ratios deterministically.
Feed them in; make the model judge what the machine cannot:

```
Candidate regions (machine-measured):
  1. bbox=[l,t,r,b]; fg=#333333; bg=#FFFFFF; ratio=12.63:1; size=normal-text
  2. bbox=[...]; fg=#8A8A8A; bg=#F0F0F0; ratio=3.92:1; size=normal-text
  ...
For each candidate judge: is it real text (vs decorative/artifact)? is the size
class right (large-text >= 18pt/14pt-bold)? is it image-of-text? Apply
thresholds (normal 4.5:1, large 3.0:1, non-text 3.0:1) to the MEASURED ratio.
```
Model output stays schema-identical (issues list with ratio echoed). The vision
task becomes: text-vs-decoration, size class, image-of-text — genuinely visual
judgments. Builder change: `build_contrast_corpus.py` prompt assembly + the
production prompt builder in remedy-server main (`pdf_vision` contrast prompt)
must move in lockstep, or train/serve skew is reintroduced.

### reading_order → MCID-enriched structure list (mirror of 61e5e26)
```
Structure tree order:
  1. /TH  bbox=[..] "Name"
  2. /TH  bbox=[..] "Room"
  3. /TD  bbox=[..] "Adams, J."
  ...
```
Per-entry visible-text snippet (first ~40 chars via MCID walk, exactly as the
heading enrichment does) + normalized bbox. Order-vs-layout becomes checkable:
the model can trace the numbered sequence across the page image.

## Cost & sequencing
- Builder edits + corpus regen: local/free (regen needs source PDFs + render
  path from the main-repo toolchain — run in the remedy-server main checkout).
- Retrain (2 tasks x ~30-60 steps): ~ $1.5-2.5 on one A100 window. **Deferred:**
  ledger at time of writing ~$41 incl. the heading2 window; under the $50 hard
  stop but past the $40 soft line. Needs fresh authorization.
- Do NOT retrain on the old prompts again — more epochs cannot fix missing
  input signal (proven by collapse patterns).

## Acceptance for the redesigned tasks
Same promotion gates (evaluation.py): contrast needs status ≥ 0.90 with
near-threshold ≥ 0.85 (now meaningful: the model sees the measured ratio and
must still judge text-vs-decoration correctly); reading_order needs ≥ 0.80.
