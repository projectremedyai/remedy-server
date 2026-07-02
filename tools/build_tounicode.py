#!/usr/bin/env python3
"""Deterministic /ToUnicode (re)builder for veraPDF 7.21.7-1/-2.

Builds a /ToUnicode CMap from a TRUSTED source (never OCR/shape guessing), so a
glyph is only mapped when we can prove its Unicode:
  * Tesseract 'GlyphLessFont'      -> content code == Unicode (identity)
  * Type0 CIDFontType2 (Identity-H) with a Unicode cmap in the program -> invert cmap
  * simple TrueType/Type1          -> /Encoding name -> AGL, or base-encoding codec
  * simple Type1 with embedded /FontFile -> the program's OWN built-in Encoding
    (t1Lib) -> glyph names -> AGL. This is what clears the Computer Modern math
    tail: the subsetter wrote U+0000 for 'minus' etc., but the embedded program
    still names its glyphs (minus, approxequal, radical, period, fi, ...) and AGL
    resolves those authoritatively. Verified: also fixes latent wrong-but-valid
    maps (period->colon, comma->semicolon, radical->'p') the producer emitted.
Glyphs it cannot prove are left unmapped (so veraPDF still fails them -> the file
is routed onward, never silently mismapped). Content-stream codes are untouched.

Existing /ToUnicode maps are only rewritten when (a) the font is named in
--targets, or (b) the map provably contains bad values (0 / U+FEFF / U+FFFE —
the 7.21.7-2 trigger). Rewrites MERGE: trusted entries override, existing valid
entries for codes we cannot prove are kept, existing BAD entries with no trusted
replacement are dropped (never re-emitted).

Usage: build_tounicode.py in.pdf out.pdf [targets]
"""
import sys, io, os, re, tempfile
import pikepdf
from fontTools.ttLib import TTFont
from fontTools import agl, t1Lib

# 0/FEFF/FFFE are veraPDF 7.21.7-2 failures; U+FFFD (REPLACEMENT CHARACTER) is
# veraPDF-legal but semantically a declared-unknown mapping -- equally garbage
# for screen readers, so we treat it as bad: never write it, and repair maps
# containing it when a trusted source exists.
BAD = {0x0, 0xFEFF, 0xFFFE, 0xFFFD}
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

def _type1_builtin_names(fd):
    """Embedded Type1 program -> {code:int -> glyphname} from its OWN built-in
    Encoding array (t1Lib). The program is the trusted source: its CharStrings
    names are authoritative for what each glyph IS. Returns {} if unparseable."""
    if "/FontFile" not in fd:
        return {}, False
    try:
        ff = fd["/FontFile"]
        raw = bytes(ff.read_bytes())
        l1 = int(ff.get("/Length1", 0)); l2 = int(ff.get("/Length2", 0))
        l3 = int(ff.get("/Length3", 0))
        seg1, seg2 = raw[:l1], raw[l1:l1+l2]
        seg3 = raw[l1+l2:l1+l2+l3] if l3 else raw[l1+l2:]
        def hdr(t, n): return bytes([0x80, t]) + n.to_bytes(4, "little")
        pfb = hdr(1, len(seg1)) + seg1 + hdr(2, len(seg2)) + seg2 + \
              hdr(1, len(seg3)) + seg3 + bytes([0x80, 3])
        with tempfile.NamedTemporaryFile(suffix=".pfb", delete=False) as tf:
            tf.write(pfb); tmp = tf.name
        try:
            font = t1Lib.T1Font(tmp); font.parse()
        finally:
            try: os.unlink(tmp)
            except OSError: pass
        enc = font.font.get("Encoding")
        fname = str(font.font.get("FontName", ""))
        zapf = "zapfdingbats" in fname.lower().replace("+", "")
        if not isinstance(enc, list):
            return {}, zapf
        return {c: n for c, n in enumerate(enc) if n and n != ".notdef"}, zapf
    except Exception:
        return {}, False

def _existing_map(obj):
    """Parse the font's current /ToUnicode -> {code:int -> unistr}. Light regex
    parser (bfchar + simple bfrange); enough to detect bad values and to merge."""
    if "/ToUnicode" not in obj:
        return None
    try:
        text = bytes(obj["/ToUnicode"].read_bytes()).decode("latin-1", "replace")
    except Exception:
        return None
    out = {}
    for m in re.finditer(r"beginbfchar(.*?)endbfchar", text, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", m.group(1)):
            dstb = bytes.fromhex(dst if len(dst) % 2 == 0 else "0" + dst)
            out[int(src, 16)] = dstb.decode("utf-16-be", "replace")
    for m in re.finditer(r"beginbfrange(.*?)endbfrange", text, re.S):
        for lo, hi, start in re.findall(
                r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", m.group(1)):
            base = int.from_bytes(bytes.fromhex(start if len(start) % 2 == 0 else "0"+start), "big")
            for i, code in enumerate(range(int(lo, 16), int(hi, 16) + 1)):
                try: out[code] = chr(base + i)
                except ValueError: out[code] = ""
    return out

def _has_bad_values(mapping):
    """True if any mapped value is empty or contains 0 / U+FEFF / U+FFFE (7.21.7-2)."""
    return any((not u) or any(ord(ch) in BAD for ch in u) for u in mapping.values())

def _simple_map(obj, fd):
    """simple TrueType/Type1 -> {code:int -> unistr}.
    Per-code precedence follows the PDF spec: /Differences name > explicit
    /BaseEncoding codec > the embedded Type1 program's built-in Encoding.
    Never default-guesses a codec: with no trusted source a code stays unmapped."""
    enc = obj.get("/Encoding")
    base = enc if isinstance(enc, pikepdf.Name) else (
           enc.get("/BaseEncoding") if isinstance(enc, pikepdf.Dictionary) else None)
    codec = {"/WinAnsiEncoding": "cp1252", "/MacRomanEncoding": "mac_roman"}.get(str(base) if base else "")
    diffs = {}
    if isinstance(enc, pikepdf.Dictionary) and "/Differences" in enc:
        cur = 0
        for it in enc["/Differences"]:
            if isinstance(it, int): cur = it
            else: diffs[cur] = str(it).lstrip("/"); cur += 1
    builtin, zapf = ({}, False)
    if obj.get("/Subtype") == pikepdf.Name("/Type1"):
        builtin, zapf = _type1_builtin_names(fd)
    # the PDF-level font name also identifies ZapfDingbats (an embedded clone
    # like D050000L keeps aNN glyph names but not the ZapfDingbats FontName)
    if "zapfdingbats" in str(obj.get("/BaseFont", "")).lower():
        zapf = True
    if not diffs and not codec and not builtin:
        return {}                                    # no trusted source -> map nothing
    lo = int(obj.get("/FirstChar", 0)); hi = int(obj.get("/LastChar", 255))
    codes = set(range(lo, hi + 1)) | set(diffs) | set(builtin)
    out = {}
    for code in sorted(codes):
        u = None
        if code in diffs:
            u = agl.toUnicode(diffs[code], isZapfDingbats=zapf) or None
        elif codec and 0 <= code <= 255:
            try: u = bytes([code]).decode(codec)
            except Exception: u = None
        elif code in builtin:
            u = agl.toUnicode(builtin[code], isZapfDingbats=zapf) or None
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
        existing = _existing_map(obj)                # None if no /ToUnicode
        pre_mapping = None                           # trusted map computed during selection
        # Selection (non-destructive by default):
        #  * with --targets: only the named fonts (+ GlyphLess repair)
        #  * without targets: fonts MISSING /ToUnicode, fonts whose existing map
        #    provably contains bad values (0/FEFF/FFFE — the 7.21.7-2 trigger),
        #    fonts whose embedded Type1 program's OWN glyph names CONTRADICT the
        #    existing map (proven-wrong Unicode, e.g. CM period mapped to colon),
        #    and the GlyphLess repair. Otherwise valid maps are never touched.
        if targets:
            if not (is_gl or any(t in base for t in targets)):
                continue
        else:
            if existing is not None and not is_gl and not _has_bad_values(existing):
                # Contradiction probe: only simple Type1 with an embedded program.
                if sub != pikepdf.Name("/Type1"):
                    continue
                fd0 = obj.get("/FontDescriptor")
                if fd0 is None or "/FontFile" not in fd0:
                    continue
                trusted = _simple_map(obj, fd0)
                # single-char vs single-char mismatch = the existing map calls the
                # glyph a different character than the font itself names it.
                # (multi-char conventions like fi -> "fi" are NOT contradictions)
                if not any(c in existing and len(existing[c]) == 1 and len(u) == 1
                           and existing[c] != u for c, u in trusted.items()):
                    continue
                pre_mapping = trusted                # proven wrong -> rewrite below
        mapping = pre_mapping; twobyte = (sub == pikepdf.Name("/Type0"))
        if mapping is not None:
            pass                                     # trusted map from selection probe
        elif "GlyphLess" in base:                    # Tesseract identity
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
        if existing:
            # MERGE: trusted entries override; existing VALID entries for codes we
            # cannot prove are kept; existing BAD entries with no trusted
            # replacement are dropped (leaving the glyph unmapped, never garbage).
            keep = {c: u for c, u in existing.items()
                    if u and all(_ok(ord(ch)) for ch in u)}
            mapping = {**keep, **mapping}
            if mapping == keep == existing:          # nothing to improve
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
