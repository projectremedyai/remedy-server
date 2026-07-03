#!/usr/bin/env python3
"""veraPDF-GUIDED fix for 7.21.5-1 that the generic fix_glyph_widths can't clear.

fix_glyph_widths rewrites /Widths by re-deriving each code's glyph from the
encoding + embedded cmap. For symbol-encoded simple fonts (Adobe Symbol,
ZapfDingbats, custom-encoded TrueType) that derivation is unreliable, so a
residual 7.21.5-1 survives — commonly after fix_all embeds a substitute program
whose advances differ from the PDF's declared /Widths.

This pass does NOT guess the encoding. It runs veraPDF, reads each failed 7.21.5
check's own error message ("Glyph width <TARGET> ... is not consistent with the
Widths entry ... (value <DICT>)") plus the used-glyph context (BaseFont + integer
character code), and writes exactly that /Widths[code] = round(TARGET) — using
veraPDF's authoritative number.

Targeting guard (so a legitimately-consistent same-BaseFont subset is not
broken): only a simple font is touched when ALL hold —
  * BaseFont matches the flagged font (subset tag stripped, case-insensitive),
  * FirstChar <= code < FirstChar+len(Widths) and Widths[code] == round(DICT),
  * its embedded program actually contains a glyph whose advance == round(TARGET).
Every change is still subject to the caller's veraPDF-subset + word-fidelity
gate, so a mistargeted edit is discarded, never shipped.

    fix_widths_from_verapdf.py in.pdf out.pdf     # exit 0 wrote, 2 nothing
"""
import sys, os, io, re, subprocess
import xml.etree.ElementTree as ET
import pikepdf
from fontTools.ttLib import TTFont

_SUBSET = re.compile(r"^[A-Z]{6}\+")


def _strip_tag(t):
    return re.sub(r"\{.*?\}", "", t)


def _basefont_key(name):
    return _SUBSET.sub("", str(name).lstrip("/")).lower()


def _verapdf_721_5(path):
    """List of (basefont_token, code:int, target:int, dictval:int) flagged 7.21.5-1."""
    r = subprocess.run(["verapdf", "-f", "ua1", "--format", "xml", path],
                       capture_output=True, text=True, timeout=600)
    try:
        root = ET.fromstring(r.stdout)
    except Exception:
        return None
    out = []
    for rule in root.iter():
        if _strip_tag(rule.tag) != "rule":
            continue
        if rule.attrib.get("clause") != "7.21.5" or rule.attrib.get("status") != "failed":
            continue
        for ch in rule.iter():
            if _strip_tag(ch.tag) != "check" or ch.attrib.get("status") != "failed":
                continue
            ctx = err = ""
            for c in ch:
                if _strip_tag(c.tag) == "context": ctx = c.text or ""
                if _strip_tag(c.tag) == "errorMessage": err = c.text or ""
            m = re.search(r"Glyph width ([0-9.]+).*?value ([0-9.]+)", err)
            ug = re.search(r"usedGlyphs\[\d+\]\(([^)]*)\)", ctx)
            if not (m and ug):
                continue
            toks = ug.group(1).split()
            code = next((int(t) for t in toks if t.isdigit()), None)  # first int = char code
            if code is None:
                continue
            out.append((_basefont_key(toks[0]), code,
                        int(round(float(m.group(1)))), int(round(float(m.group(2))))))
    return out


def _program_has_adv(fd, target):
    """True if any embedded-program glyph has advance == target (/1000 em)."""
    for key in ("/FontFile2", "/FontFile3"):
        if fd is not None and key in fd:
            try:
                tt = TTFont(io.BytesIO(bytes(fd[key].read_bytes())))
                upm = tt["head"].unitsPerEm if "head" in tt else 1000
                gs = tt.getGlyphSet()
                for name in tt.getGlyphOrder():
                    w = gs[name].width
                    if w is not None and int(round(w * 1000.0 / upm)) == target:
                        return True
            except Exception:
                return False
    return False


def fix(inp, outp):
    flags = _verapdf_721_5(inp)
    if not flags:
        return 0
    pdf = pikepdf.open(inp)
    changed = 0
    for bf_key, code, target, dictval in flags:
        for obj in pdf.objects:
            try:
                if obj.get("/Type") != pikepdf.Name("/Font"):
                    continue
                if obj.get("/Subtype") not in (pikepdf.Name("/TrueType"), pikepdf.Name("/Type1")):
                    continue
            except Exception:
                continue
            if _basefont_key(obj.get("/BaseFont", "")) != bf_key:
                continue
            W = obj.get("/Widths")
            fc = obj.get("/FirstChar")
            if W is None or fc is None:
                continue
            fc = int(fc); idx = code - fc
            if not (0 <= idx < len(W)) or int(W[idx]) != dictval:
                continue
            if not _program_has_adv(obj.get("/FontDescriptor"), target):
                continue                              # not the culprit subset
            w = pikepdf.Array([int(x) for x in W])
            w[idx] = target
            obj["/Widths"] = w
            changed += 1
    if changed:
        pdf.save(outp)
    pdf.close()
    return changed


if __name__ == "__main__":
    n = fix(sys.argv[1], sys.argv[2])
    print(f"{'wrote' if n else 'no-change'}: {n} verapdf-guided width entr(y/ies)")
    sys.exit(0 if n else 2)
