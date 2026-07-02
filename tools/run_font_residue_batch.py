#!/usr/bin/env python3
"""Batch-apply the deterministic font-residue passes with a veraPDF safety gate.

For each input PDF:
  0. fix_content_splices.fix   (rule 7.21.8 — marked-content operators spliced
     INSIDE string literals by an earlier tag-injection pass, and control bytes
     in show strings, replaced by exact-width TJ kerns)
  1. fix_glyph_widths.fix      (rule 7.21.5 — width arrays)
  2. embed_missing_fonts.embed (rule 7.21.4.1 — substitute-program embed for
     non-embedded WinAnsi simple fonts, hmtx patched to the PDF /Widths;
     standard-14 /ZaDb+/Helv via URW clones; real-font embeds/upgrades for
     macOS-shipped Microsoft families incl. incomplete-subset upgrades)
  3. build_tounicode.build     (rule 7.21.7 — ToUnicode; trusted sources only.
     Runs AFTER embed so a freshly embedded program's built-in encoding /
     cmap is available as a trusted source)
Then gate: the output is KEPT only if its set of failed veraPDF clauses is a
SUBSET of the input's (i.e. some cleared, none newly introduced). Otherwise the
output is discarded and the file is reported as a regression (should not happen).

These passes CANNOT fix 7.21.8 (.notdef) or 7.21.4.1-2 (embedded font missing a
used glyph); files with those remaining are reported PARTIAL and routed onward.

Usage:
    python run_font_residue_batch.py INPUT_DIR OUTPUT_DIR
    python run_font_residue_batch.py file1.pdf file2.pdf --out OUTPUT_DIR
Originals are never modified; outputs are written to OUTPUT_DIR.
"""
import sys, os, re, glob, shutil, tempfile, subprocess, json
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_glyph_widths
import build_tounicode
import embed_missing_fonts
import fix_content_splices

def word_fidelity(src, out):
    """(hard_lost, total): words present in src but not out, net of additions
    (substitution-style corrections like '0:42'->'0.42' cancel out)."""
    import fitz
    from collections import Counter
    def words(p):
        c = Counter()
        d = fitz.open(p)
        for pno in range(d.page_count):
            for w in d[pno].get_text("words"):
                c[w[4]] += 1
        d.close()
        return c
    ws, wo = words(src), words(out)
    lost = sum((ws - wo).values())
    new = sum((wo - ws).values())
    return max(0, lost - new), (sum(ws.values()) or 1)

def failed_clauses(path):
    """Set of 'clause-testNumber' strings veraPDF reports failed (all clauses)."""
    r = subprocess.run(["verapdf", "-f", "ua1", "--format", "xml", path],
                       capture_output=True, text=True, timeout=600)
    try:
        root = ET.fromstring(r.stdout)
    except Exception:
        return None
    out = set()
    for rule in root.iter():
        if re.sub(r"\{.*?\}", "", rule.tag) == "rule" and rule.attrib.get("status") == "failed":
            out.add(f"{rule.attrib.get('clause')}-{rule.attrib.get('testNumber')}")
    return out

def process(inp, outdir):
    name = os.path.basename(inp)
    before = failed_clauses(inp)
    if before is None:
        return {"file": name, "status": "VERAPDF_ERROR"}
    with tempfile.TemporaryDirectory() as td:
        cur = inp
        s = os.path.join(td, "s.pdf")
        n, _rep = fix_content_splices.fix(cur, s)
        if n:
            cur = s
        a = os.path.join(td, "w.pdf")
        if fix_glyph_widths.fix(cur, a):
            cur = a
        b = os.path.join(td, "e.pdf")
        ch, _warns = embed_missing_fonts.embed(cur, b)
        if ch:
            cur = b
        c = os.path.join(td, "t.pdf")
        if build_tounicode.build(cur, c):
            cur = c
        d = os.path.join(td, "d.pdf")
        n, _rep = fix_content_splices.fix_dead_refs(cur, d)
        if n:
            cur = d
        e = os.path.join(td, "o.pdf")
        try:                                        # rule 7.10-1 (OCG config /Name)
            import pikepdf
            from project_remedy.pdf_fixer import fix_optional_content_config_names
            with pikepdf.open(cur) as _p:
                if fix_optional_content_config_names(_p):
                    _p.save(e)
                    cur = e
        except Exception:
            pass
        if cur == inp:                              # no pass changed anything
            return {"file": name, "status": "NO_APPLICABLE_PASS", "remaining": sorted(before)}
        after = failed_clauses(cur)
        if after is None:
            return {"file": name, "status": "VERAPDF_ERROR_OUT"}
        if not after <= before:                     # introduced a NEW failure -> discard
            return {"file": name, "status": "REGRESSION_DISCARDED",
                    "new": sorted(after - before)}
        # TEXT-FIDELITY GATE: veraPDF can pass while extraction is destroyed
        # (e.g. an OCR text layer kerned away). Discard on net word loss.
        hard_lost, total = word_fidelity(inp, cur)
        if hard_lost > max(5, 0.02 * total):
            return {"file": name, "status": "FIDELITY_DISCARDED",
                    "lost_words": hard_lost, "total_words": total}
        outp = os.path.join(outdir, name)
        shutil.copyfile(cur, outp)
        cleared = sorted(before - after)
        status = "PASS" if not after else "PARTIAL"
        return {"file": name, "status": status, "cleared": cleared,
                "remaining": sorted(after), "out": outp}

def main(argv):
    base = ""
    if "--base" in argv:
        i = argv.index("--base"); base = argv[i+1]; del argv[i:i+2]
    if "--out" in argv:
        i = argv.index("--out"); outdir = argv[i+1]; inputs = argv[:i] + argv[i+2:]
    else:
        inputs = [argv[0]]; outdir = argv[1]
    files = []
    for p in inputs:
        if p.endswith(".txt"):                       # newline-separated file list
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#"):
                    files.append(os.path.join(base, line) if base else line)
        elif os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.pdf")))
        else:
            files.append(os.path.join(base, p) if base else p)
    os.makedirs(outdir, exist_ok=True)
    results = []
    for f in files:
        r = process(f, outdir)
        results.append(r)
        tag = {"PASS": "✅ PASS", "PARTIAL": "◐ PARTIAL", "NO_APPLICABLE_PASS": "· none",
               "REGRESSION_DISCARDED": "✗ REGRESSION", "VERAPDF_ERROR": "! verapdf",
               "FIDELITY_DISCARDED": "✗ FIDELITY",
               "VERAPDF_ERROR_OUT": "! verapdf-out"}.get(r["status"], r["status"])
        extra = ""
        if r.get("cleared"): extra += f" cleared={r['cleared']}"
        if r.get("remaining"): extra += f" remaining={r['remaining']}"
        if r.get("new"): extra += f" NEW={r['new']}"
        if r.get("lost_words"): extra += f" lost_words={r['lost_words']}/{r['total_words']}"
        print(f"{tag:14} {r['file'][:52]:54}{extra}")
    from collections import Counter
    c = Counter(r["status"] for r in results)
    print("\nSummary:", dict(c))
    json.dump(results, open(os.path.join(outdir, "_batch_report.json"), "w"), indent=2)
    print(f"report -> {os.path.join(outdir, '_batch_report.json')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
