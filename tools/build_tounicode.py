#!/usr/bin/env python3
"""Deterministic /ToUnicode (re)builder for veraPDF 7.21.7-1/-2.

Builds a /ToUnicode CMap from a TRUSTED source (never OCR/shape guessing), so a
glyph is only mapped when we can prove its Unicode:
  * Tesseract 'GlyphLessFont'      -> content code == Unicode (identity)
  * Type0 CIDFontType2 (Identity-H) with a Unicode cmap in the program -> invert cmap
  * simple TrueType/Type1          -> /Encoding name -> AGL, or base-encoding codec
Glyphs it cannot prove are left unmapped (so veraPDF still fails them -> the file
is routed onward, never silently mismapped). Content-stream codes are untouched.

Usage: build_tounicode.py in.pdf out.pdf
"""
import sys, io, os
import pikepdf
from fontTools.ttLib import TTFont
from fontTools import agl

BAD = {0x0, 0xFEFF, 0xFFFE}
def _ok(u): return u is not None and u not in BAD and not (0xD800 <= u <= 0xDFFF)

def _cmap_stream(mapping, twobyte):
    """mapping: {code:int -> unistr}. Return CMap bytes (Adobe-Identity-UCS, Type2)."""
    items = sorted((c, u) for c, u in mapping.items() if u and all(_ok(ord(ch)) for ch in u))
    if not items:
        return None
    cw = 4 if twobyte else 2
    def hc(c): return f"{c:0{cw}X}"
    def hu(s): return "".join(f"{ord(ch):04X}" for ch in s)
    hdr = ("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
           "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
           "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
           f"1 begincodespacerange\n<{'0'*cw}> <{'F'*cw}>\nendcodespacerange\n")
    body = []
    for i in range(0, len(items), 100):
        chunk = items[i:i+100]
        body.append(f"{len(chunk)} beginbfchar\n")
        for c, u in chunk:
            body.append(f"<{hc(c)}> <{hu(u)}>\n")
        body.append("endbfchar\n")
    ftr = "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    return (hdr + "".join(body) + ftr).encode("latin-1")

def _glyphless_codes(path):
    """Codes actually drawn by any GlyphLessFont span (code == Unicode)."""
    import fitz
    codes = set()
    doc = fitz.open(path)
    for pno in range(doc.page_count):
        for span in doc[pno].get_texttrace():
            if "GlyphLess" in str(span.get("font", "")):
                for ch in span["chars"]:
                    codes.add(ch[0])
    doc.close()
    return codes

def _invert_program_cmap(fd):
    """Type0 program -> {GID: Unicode int} by inverting the program's Unicode cmap."""
    if "/FontFile2" not in fd:
        return None
    try:
        tt = TTFont(io.BytesIO(bytes(fd["/FontFile2"].read_bytes())))
        best = tt.getBestCmap()            # unicode -> glyphname
        if not best:
            return None
        n2g = {n: i for i, n in enumerate(tt.getGlyphOrder())}
        g2u = {}
        for u, gn in best.items():
            g = n2g.get(gn)
            if g is not None and _ok(u):
                g2u.setdefault(g, u)
        return g2u
    except Exception:
        return None

def _cid_to_gid(cidfont):
    m = cidfont.get("/CIDToGIDMap")
    if m is None or m == pikepdf.Name("/Identity"):
        return lambda cid: cid
    b = bytes(m.read_bytes())
    return lambda cid: ((b[cid*2] << 8) | b[cid*2+1]) if cid*2+1 < len(b) else 0

def _simple_map(obj, fd):
    """simple TrueType/Type1 -> {code:int -> unistr} via /Encoding names -> AGL / codec."""
    enc = obj.get("/Encoding")
    base = enc if isinstance(enc, pikepdf.Name) else (
           enc.get("/BaseEncoding") if isinstance(enc, pikepdf.Dictionary) else None)
    # Only trust an EXPLICIT standard base encoding. Never default-guess (cp1252),
    # or a custom builtin encoding (e.g. Computer Modern) yields wrong control chars.
    codec = {"/WinAnsiEncoding": "cp1252", "/MacRomanEncoding": "mac_roman"}.get(str(base) if base else "")
    diffs = {}
    if isinstance(enc, pikepdf.Dictionary) and "/Differences" in enc:
        cur = 0
        for it in enc["/Differences"]:
            if isinstance(it, int): cur = it
            else: diffs[cur] = str(it).lstrip("/"); cur += 1
    if not diffs and not codec:
        return {}                                    # no trusted source -> map nothing
    out = {}
    lo = int(obj.get("/FirstChar", 0)); hi = int(obj.get("/LastChar", 255))
    for code in range(lo, hi + 1):
        u = None
        if code in diffs:
            t = agl.toUnicode(diffs[code])           # glyph name -> Unicode (AGL only)
            u = t if t else None
        elif codec:
            try: u = bytes([code]).decode(codec)
            except Exception: u = None
        if u and all(_ok(ord(ch)) for ch in u):
            out[code] = u
    return out

def build(inp, outp, targets=None):
    pdf = pikepdf.open(inp)
    gl_codes = None
    written = []
    for obj in pdf.objects:
        try:
            if not isinstance(obj, pikepdf.Dictionary) or obj.get("/Type") != pikepdf.Name("/Font"):
                continue
        except Exception:
            continue
        base = str(obj.get("/BaseFont", ""))
        sub = obj.get("/Subtype")
        is_gl = "GlyphLess" in base
        # Selection (non-destructive by default):
        #  * with --targets: only the named fonts (+ GlyphLess repair)
        #  * without targets: only fonts MISSING /ToUnicode, plus GlyphLess repair.
        #    Overwriting an existing /ToUnicode (e.g. 7.21.7-2 repair) is opt-in via targets.
        if targets:
            if not (is_gl or any(t in base for t in targets)):
                continue
        else:
            if "/ToUnicode" in obj and not is_gl:
                continue
        mapping = None; twobyte = (sub == pikepdf.Name("/Type0"))
        if "GlyphLess" in base:                      # Tesseract identity
            if gl_codes is None: gl_codes = _glyphless_codes(inp)
            mapping = {c: chr(c) for c in gl_codes if _ok(c)}
        elif sub == pikepdf.Name("/Type0"):          # cmap inversion
            for cidfont in (obj.get("/DescendantFonts") or []):
                fd = cidfont.get("/FontDescriptor")
                if fd is None: continue
                g2u = _invert_program_cmap(fd)
                if not g2u: continue
                c2g = _cid_to_gid(cidfont)
                mapping = {}
                # code == CID under Identity-H; map every CID whose GID has a Unicode
                for cid in range(max(g2u) + 1):
                    g = c2g(cid)
                    if g in g2u: mapping[cid] = chr(g2u[g])
                break
        elif sub in (pikepdf.Name("/TrueType"), pikepdf.Name("/Type1")):
            fd = obj.get("/FontDescriptor")
            if fd is not None and any(k in fd for k in ("/FontFile", "/FontFile2", "/FontFile3")):
                mapping = _simple_map(obj, fd)
        if not mapping:
            continue
        data = _cmap_stream(mapping, twobyte)
        if not data:
            continue
        obj["/ToUnicode"] = pdf.make_stream(data)
        written.append((base, len(mapping)))
    if written:
        pdf.save(outp)
    pdf.close()
    return written

if __name__ == "__main__":
    inp, outp = sys.argv[1], sys.argv[2]
    tg = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    w = build(inp, outp, tg)
    for b, n in w:
        print(f"  /ToUnicode built: {b} ({n} entries)")
    print(f"{'wrote' if w else 'no-change'}: {len(w)} font(s) -> {outp if w else '(none)'}")
    sys.exit(0 if w else 2)
