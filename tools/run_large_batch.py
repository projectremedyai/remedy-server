#!/usr/bin/env python3
"""Full structural+font remediation for the LARGE (>50pp) LAMC source corpus.

The 79 large source files (~10,384pp) fail broadly — structure tree (7.1-x),
document/metadata (5-1/6.2-1), tables/headings (7.2-x) AND font residue (7.21.x,
incl. CIDSet 7.21.4.2) — so the font-residue pipeline alone is insufficient.
Calibration (2026-07-02) proved the recipe: the engine's offline structural
fix_all clears the heavy 7.1/7.2/metadata clauses at ~0.11 s/page and
visual_diff_pct=0.0, then the font-residue passes + CIDSet + OCG /Name clean up
what fix_all leaves. Per file, in order:

  -1. project_remedy.pdf_fixer.fix_all(thorough=True)  — structural (deep fixes
      forced via PDF_LARGE_DOC_DEEP_FIXES=1); 7.1-x / 7.2-x / metadata
   0. fix_content_splices.fix                           — 7.21.8 splices/controls
   1. fix_glyph_widths.fix                              — 7.21.5 widths
   2. embed_missing_fonts.embed                         — 7.21.4.1 embeds/upgrades
   3. build_tounicode.build                             — 7.21.7 ToUnicode
   4. fix_content_splices.fix_dead_refs                 — 7.21.8 late dead-ref kerns
   5. fix_cidset.fix                                    — 7.21.4.2 (remove wrong
                                                          optional CIDSet stream)
   6. fix_optional_content_config_names                 — 7.10-1 OCG /Name

Same two gates as run_font_residue_batch: output KEPT only if its failed-clause
set is a SUBSET of the input's (nothing newly broken) AND net word loss is within
tolerance (veraPDF can pass while extraction is destroyed). Originals are never
touched; outputs -> OUTPUT_DIR. Progress is written incrementally to
OUTPUT_DIR/_large_report.json after every file so the run can be monitored.

Usage:
    PDF_LARGE_DOC_DEEP_FIXES=1 uv run python tools/run_large_batch.py \
        --src   ~/code/lamc_district_forms/data/visual_match/downloads/lamc \
        --list  <verify_79.json>                (reads the 79 file names) \
        --out   ~/code/lamc_district_forms/lamc_remediated/large_fixed
"""
import sys, os, json, time, shutil, tempfile, glob, subprocess, signal
from pathlib import Path

PER_FILE_TIMEOUT = int(os.environ.get("LARGE_BATCH_TIMEOUT", "1200"))  # hard cap/file (s); 319pp ~55s

os.environ.setdefault("PDF_LARGE_DOC_DEEP_FIXES", "1")   # force deferred deep fixes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_font_residue_batch as R          # reuse word_fidelity + failed_clauses + passes
import fix_glyph_widths
import build_tounicode
import embed_missing_fonts
import fix_content_splices
import fix_cidset
import fix_widths_from_verapdf


def _clause_of(c):
    """'7.21.4.2-1' -> '7.21.4.2'; '7.1-3' -> '7.1'; '7.10-1' -> '7.10'."""
    return c.rsplit("-", 1)[0]


def _is_font_ocg(c):
    """Clauses the font-residue passes + CIDSet + OCG one-liner address."""
    cl = _clause_of(c)
    return cl.startswith("7.21") or cl == "7.10"


def _structural(cur, dst):
    """Engine deep structural pass (fix_all requires Path — it calls .resolve())."""
    from project_remedy.pdf_fixer import fix_all
    rep = fix_all(Path(cur), Path(dst), thorough=True)
    return {"changes": len(getattr(rep, "changes", [])),
            "visual_diff_pct": getattr(rep, "visual_diff_pct", None)}


def _font_passes(cur, td, meta):
    """Run the deterministic font-residue + CIDSet + OCG passes. Returns final path."""
    p = os.path.join(td, "s.pdf")
    try:
        n, _ = fix_content_splices.fix(cur, p)
        if n: cur = p
    except Exception as e:
        meta["splice_error"] = repr(e)[:160]
    p = os.path.join(td, "w.pdf")
    try:
        if fix_glyph_widths.fix(cur, p): cur = p
    except Exception as e:
        meta["width_error"] = repr(e)[:160]
    p = os.path.join(td, "e.pdf")
    try:
        ch, _ = embed_missing_fonts.embed(cur, p)
        if ch: cur = p
    except Exception as e:
        meta["embed_error"] = repr(e)[:160]
    p = os.path.join(td, "t.pdf")
    try:
        if build_tounicode.build(cur, p): cur = p
    except Exception as e:
        meta["tounicode_error"] = repr(e)[:160]
    p = os.path.join(td, "d.pdf")
    try:
        n, _ = fix_content_splices.fix_dead_refs(cur, p)
        if n: cur = p
    except Exception as e:
        meta["deadref_error"] = repr(e)[:160]
    p = os.path.join(td, "c.pdf")
    try:
        rep = fix_cidset.fix(cur, p)
        if rep["cidset_removed"] or rep["cidset_rebuilt"]:
            cur = p; meta["cidset"] = {k: rep[k] for k in ("cidset_removed", "cidset_rebuilt")}
    except Exception as e:
        meta["cidset_error"] = repr(e)[:160]
    p = os.path.join(td, "v.pdf")
    try:
        # veraPDF-guided width fix: clears residual 7.21.5-1 that fix_glyph_widths
        # can't (symbol-encoded fonts / substitute embeds). No-op if none flagged.
        nv = fix_widths_from_verapdf.fix(cur, p)
        if nv:
            cur = p; meta["guided_widths"] = nv
    except Exception as e:
        meta["guided_width_error"] = repr(e)[:160]
    p = os.path.join(td, "o.pdf")
    try:
        import pikepdf
        from project_remedy.pdf_fixer import fix_optional_content_config_names
        with pikepdf.open(cur) as _p:
            if fix_optional_content_config_names(_p):
                _p.save(p); cur = p
    except Exception as e:
        meta["ocg_error"] = repr(e)[:160]
    return cur


def process(inp, outdir):
    """Adaptive: fix_all ONLY for structural failures; font passes ONLY for
    residual font/OCG clauses; ship the best valid candidate (fewest remaining,
    subset of `before`, fidelity-ok) so a font-pass regression falls back to the
    structural-only output instead of discarding all progress."""
    name = os.path.basename(inp)
    t0 = time.time()
    before = R.failed_clauses(inp)
    if before is None:
        return {"file": name, "status": "VERAPDF_ERROR"}
    needs_struct = any(not _is_font_ocg(c) for c in before)

    with tempfile.TemporaryDirectory() as td:
        meta = {"needs_struct": needs_struct}
        candidates = []            # (after_set, path, label)

        # --- structural candidate (only if non-font/OCG clauses present) ---
        base_for_font, after_struct = inp, before
        if needs_struct:
            sp = os.path.join(td, "struct.pdf")
            try:
                meta.update(_structural(inp, sp))
                after_struct = R.failed_clauses(sp)
                if after_struct is not None:
                    candidates.append((after_struct, sp, "struct"))
                    base_for_font = sp
            except Exception as e:
                meta["struct_error"] = repr(e)[:200]
                after_struct = before

        # short-circuit: structural pass already fully clean -> ship it, no font
        # passes. But STILL gate on text fidelity: a veraPDF-clean output can have
        # destroyed the extractable text layer (the invisible-text regression), and
        # skipping this gate here is exactly how broken files shipped before.
        if after_struct is not None and len(after_struct) == 0:
            hard_lost, total = R.word_fidelity(inp, base_for_font)
            if hard_lost > max(5, 0.02 * total):
                meta.setdefault("rejected", {})["struct"] = f"fidelity {hard_lost}/{total}"
                return {"file": name, "status": "DISCARDED_FIDELITY",
                        "before": sorted(before), "meta": meta,
                        "secs": round(time.time() - t0, 1)}
            return _ship(inp, base_for_font, before, after_struct, outdir, name, meta, t0, "struct")

        # --- full candidate: font passes on top of the (possibly structural) base ---
        # skip only if nothing font/OCG remains to fix
        residual = after_struct if after_struct is not None else before
        if any(_is_font_ocg(c) for c in residual) or not needs_struct:
            full = _font_passes(base_for_font, td, meta)
            if full != base_for_font:
                after_full = R.failed_clauses(full)
                if after_full is not None:
                    candidates.append((after_full, full, "full"))

        # --- choose best valid candidate ---
        best = None
        for after, path, label in candidates:
            if not after <= before:                    # introduced a NEW failure
                meta.setdefault("rejected", {})[label] = sorted(after - before)
                continue
            hard_lost, total = R.word_fidelity(inp, path)
            if hard_lost > max(5, 0.02 * total):
                meta.setdefault("rejected", {})[label] = f"fidelity {hard_lost}/{total}"
                continue
            if best is None or len(after) < len(best[0]):
                best = (after, path, label)

        if best is None:
            return {"file": name, "status": "DISCARDED_NO_VALID",
                    "before": sorted(before), "meta": meta,
                    "secs": round(time.time() - t0, 1)}
        return _ship(inp, best[1], before, best[0], outdir, name, meta, t0, best[2])


def _ship(inp, path, before, after, outdir, name, meta, t0, label):
    outp = os.path.join(outdir, name)
    shutil.copyfile(path, outp)
    return {"file": name, "status": "PASS" if not after else "PARTIAL",
            "route": label, "cleared": sorted(before - after),
            "remaining": sorted(after), "meta": meta, "out": outp,
            "secs": round(time.time() - t0, 1)}


def _load_list(src, listarg):
    if listarg and listarg.endswith(".json"):
        d = json.load(open(listarg))
        names = [r["file"] for r in d.get("files", []) if r.get("status") in ("FAIL", "ERROR")]
        return [os.path.join(src, n) for n in names]
    # else: scan src for >50pp
    import pikepdf
    out = []
    for p in sorted(glob.glob(os.path.join(src, "*.pdf"))):
        try:
            with pikepdf.open(p) as pdf:
                if len(pdf.pages) > 50:
                    out.append(p)
        except Exception:
            pass
    return out


def _run_one_subprocess(f, src, outdir, timeout):
    """Process ONE file in an isolated subprocess with a hard wall-clock cap.

    A pathological file (e.g. fix_all looping on a broken structure tree) is
    SIGKILL'd at `timeout` and recorded TIMEOUT instead of hanging the batch.
    Runs in its own process group so grandchildren (veraPDF) die with it.
    """
    name = os.path.basename(f)
    rp = os.path.join(tempfile.gettempdir(), f"_one_{abs(hash(name)) % 10**9}.json")
    errlog = rp + ".err"
    for p_ in (rp, errlog):
        if os.path.exists(p_):
            os.remove(p_)
    cmd = [sys.executable, os.path.abspath(__file__),
           "--src", src, "--out", outdir, "--one", f, "--result", rp]
    env = dict(os.environ, PDF_LARGE_DOC_DEEP_FIXES="1")
    with open(errlog, "wb") as ef:
        proc = subprocess.Popen(cmd, start_new_session=True, env=env,
                                stdout=subprocess.DEVNULL, stderr=ef)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            proc.wait()
            return {"file": name, "status": "TIMEOUT", "timeout_s": timeout}
    if os.path.exists(rp):
        try:
            r = json.load(open(rp)); os.remove(rp)
            if os.path.exists(errlog): os.remove(errlog)
            return r
        except Exception as e:
            return {"file": name, "status": "RESULT_PARSE_ERR", "err": str(e)[:120]}
    tail = ""
    if os.path.exists(errlog):
        tail = open(errlog, "rb").read()[-400:].decode("utf-8", "replace")
        os.remove(errlog)
    return {"file": name, "status": "CRASH", "returncode": proc.returncode, "stderr_tail": tail}


def main(argv):
    src = listarg = outdir = one = result = None
    for i in range(len(argv)):
        if argv[i] == "--src": src = argv[i + 1]
        elif argv[i] == "--list": listarg = argv[i + 1]
        elif argv[i] == "--out": outdir = argv[i + 1]
        elif argv[i] == "--one": one = argv[i + 1]
        elif argv[i] == "--result": result = argv[i + 1]
    src = os.path.expanduser(src); outdir = os.path.expanduser(outdir)
    os.makedirs(outdir, exist_ok=True)

    # --- single-file worker (spawned by the batch loop) ---
    if one:
        try:
            r = process(one, outdir) if os.path.exists(one) else \
                {"file": os.path.basename(one), "status": "MISSING_SRC"}
        except Exception as e:
            r = {"file": os.path.basename(one), "status": "CRASH", "err": repr(e)[:300]}
        if result:
            json.dump(r, open(result, "w"))
        return

    # --- batch driver: one isolated, time-capped subprocess per file ---
    if listarg: listarg = os.path.expanduser(listarg)
    files = _load_list(src, listarg)
    report = os.path.join(outdir, "_large_report.json")
    results = []
    print(f"large-file remediation: {len(files)} files -> {outdir} "
          f"(per-file timeout {PER_FILE_TIMEOUT}s)", flush=True)
    for idx, f in enumerate(files, 1):
        r = _run_one_subprocess(f, src, outdir, PER_FILE_TIMEOUT)
        results.append(r)
        extra = ""
        if r.get("cleared"): extra += f" cleared={len(r['cleared'])}"
        if r.get("remaining"): extra += f" remaining={r['remaining']}"
        print(f"[{idx:2}/{len(files)}] {r['status']:18} {r['file'][:50]:52}{extra}"
              f"  ({r.get('secs','?')}s)", flush=True)
        json.dump(results, open(report, "w"), indent=1)      # incremental
    from collections import Counter
    c = Counter(r["status"] for r in results)
    print("\nSummary:", dict(c), flush=True)
    print(f"report -> {report}", flush=True)


if __name__ == "__main__":
    if not sys.argv[1:]:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
