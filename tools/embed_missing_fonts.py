#!/usr/bin/env python3
"""Embed a substitute font program into non-embedded simple fonts (7.21.4.1-1).

Every failing font in this corpus is a simple TrueType with /WinAnsiEncoding
(the v1 'encoding_unresolvable' label was a misdiagnosis), so code->char is
fully determined; the ONLY defect is the missing program. Substitutes:

  BaseFont        program                                          license
  --------        -------                                          -------
  Tahoma          /System/Library/Fonts/Supplemental/Tahoma.ttf    real Tahoma, fsType=8 (editable embed OK)
  Tahoma-Bold     .../Tahoma Bold.ttf                              real Tahoma Bold, fsType=8
  Calibri         assets/fonts/Carlito-Regular.ttf                 Carlito, SIL OFL (metric-compatible)
  CenturyGothic   assets/fonts/texgyreadventor-regular.otf         TeX Gyre Adventor, GFL (Avant-Garde lineage)

To avoid introducing 7.21.5-1 (dict widths vs program advance) and to preserve
layout EXACTLY, the substitute's hmtx advances are rewritten to the PDF's own
/Widths for every declared code (viewers position simple-font text by /Widths,
so rendering positions are unchanged by construction). TrueType programs embed
as /FontFile2; CFF-flavoured OpenType as /FontFile3 /Subtype /OpenType.

Usage: embed_missing_fonts.py in.pdf out.pdf   (exit 0 wrote / 2 no-change)
"""
import io, os, re, sys
import pikepdf
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets", "fonts")

SUBSTITUTES = {
    "Tahoma":        "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "Tahoma-Bold":   "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    "Tahoma,Bold":   "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    "Calibri":       os.path.join(ASSETS, "Carlito-Regular.ttf"),
    "CenturyGothic": os.path.join(ASSETS, "texgyreadventor-regular.otf"),
}

# Standard-14 Type1 fonts (bare AcroForm /ZaDb + /Helv dicts): URW base-35
# metric-compatible Ghostscript clones, embedded as real Type1 programs with
# the program renamed to the standard name for dict/descriptor consistency.
# D050000L's built-in encoding IS the ZapfDingbats encoding (verified:
# 0x34->a20 / 0x6E->a73; all 202 encoded glyphs resolve via the Zapf AGL list);
# NimbusSans ships Helvetica's StandardEncoding + AGL-standard glyph names.
STD14 = {
    "ZapfDingbats": (os.path.join(ASSETS, "D050000L.t1"), "D050000L", 4),
    "Helvetica":    (os.path.join(ASSETS, "NimbusSans-Regular.t1"), "NimbusSans-Regular", 32),
}

# WinAnsi (cp1252) code -> unicode; undefined codes excluded
def _winansi(code):
    try:
        return ord(bytes([code]).decode("cp1252"))
    except Exception:
        return None

def _is_embedded(fd):
    return isinstance(fd, pikepdf.Dictionary) and any(
        k in fd for k in ("/FontFile", "/FontFile2", "/FontFile3"))

def _patched_program(path, widths, firstchar):
    """Load substitute, set hmtx advance = PDF /Widths for each declared WinAnsi
    code, return (font_bytes, is_cff, missing_codes)."""
    tt = TTFont(path, fontNumber=0)
    upem = tt["head"].unitsPerEm
    cmap = tt.getBestCmap()
    hmtx = tt["hmtx"]
    missing = []
    for i, w in enumerate(widths):
        w = int(w)
        if w <= 0:
            continue
        code = firstchar + i
        u = _winansi(code)
        # control / cp1252-undefined codes can't be drawn as text: producers
        # often declare /Widths across 0..255 anyway -- ignore, don't fail.
        if u is None or u < 0x20 or code == 0x7F:
            continue
        gname = cmap.get(u)
        if gname is None:
            missing.append(code)                     # printable gap: warn, gate decides
            continue
        adv = round(w * upem / 1000)
        _, lsb = hmtx[gname]
        hmtx[gname] = (adv, lsb)
    is_cff = "CFF " in tt
    if is_cff:
        # CFF charstrings carry their own width; veraPDF/consumers read hmtx for
        # OpenType, but keep CFF nominal/default widths consistent by clearing
        # them is risky -- leave CFF as-is and let the veraPDF gate arbitrate.
        pass
    buf = io.BytesIO()
    tt.save(buf)
    return buf.getvalue(), is_cff, missing

_STD14_CACHE = {}
def _std14_program(std_name):
    """URW clone as raw Type1 segments, program renamed to the standard name.
    Returns (fontfile_bytes, l1, l2, l3, bbox, code2width)."""
    if std_name in _STD14_CACHE:
        return _STD14_CACHE[std_name]
    import tempfile
    from fontTools import t1Lib
    from fontTools.pens.basePen import NullPen
    path, urw_name, _flags = STD14[std_name]
    f = t1Lib.T1Font(path); f.parse()
    bbox = list(f.font["FontBBox"])
    # per-code advance widths from the program (FontMatrix 0.001 -> already /1000);
    # written into the font dict so the embedded program is width-consistent
    # (7.21.5) BY CONSTRUCTION.
    enc = f.font["Encoding"]
    cs = f.font["CharStrings"]
    code2width = {}
    for code, name in enumerate(enc):
        if name == ".notdef" or name not in cs:
            continue
        g = cs[name]
        g.draw(NullPen())                            # .width populated by drawing
        code2width[code] = int(round(g.width))
    with tempfile.TemporaryDirectory() as td:
        pfb = os.path.join(td, "f.pfb")
        f.saveAs(pfb, "PFB")
        raw = open(pfb, "rb").read()
        segs, i = [], 0
        while i < len(raw) and raw[i] == 0x80:
            t = raw[i+1]
            if t == 3:
                break
            n = int.from_bytes(raw[i+2:i+6], "little")
            segs.append((t, raw[i+6:i+6+n])); i += 6 + n
        ascii1 = segs[0][1].replace(b"/" + urw_name.encode(), b"/" + std_name.encode())
        binary = b"".join(d for t, d in segs[1:] if t == 2)
        trailer = b"".join(d for t, d in segs[1:] if t == 1)
        data = ascii1 + binary + trailer
    _STD14_CACHE[std_name] = (data, len(ascii1), len(binary), len(trailer), bbox, code2width)
    return _STD14_CACHE[std_name]

def _embed_std14(pdf, obj, fd, std_name):
    """Embed a URW clone into a bare standard-14 Type1 dict (/ZaDb, /Helv).
    Builds the missing FontDescriptor and adds FirstChar/LastChar/Widths from
    the program itself when the dict has none."""
    data, l1, l2, l3, bbox, code2width = _std14_program(std_name)
    flags = STD14[std_name][2]
    stream = pdf.make_stream(data)
    stream.stream_dict["/Length1"] = l1
    stream.stream_dict["/Length2"] = l2
    stream.stream_dict["/Length3"] = l3
    if fd is None:
        fd = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/FontDescriptor"),
            "/FontName": pikepdf.Name("/" + std_name),
            "/Flags": flags,                         # 4 symbolic / 32 nonsymbolic
            "/FontBBox": pikepdf.Array(bbox),
            "/ItalicAngle": 0,
            "/Ascent": bbox[3], "/Descent": bbox[1],
            "/CapHeight": bbox[3], "/StemV": 90,
        }))
        obj["/FontDescriptor"] = fd
    fd["/FontFile"] = stream
    if "/Widths" not in obj and code2width:
        lo, hi = min(code2width), max(code2width)
        obj["/FirstChar"] = lo
        obj["/LastChar"] = hi
        obj["/Widths"] = pikepdf.Array(
            [code2width.get(c, 0) for c in range(lo, hi + 1)])

# In-place Type0/CIDFontType2 embed: the content stream's CIDs are FIXED and,
# under /CIDToGIDMap /Identity, they are literal GLYPH IDs of the named font.
# The authoritative fix is therefore embedding the REAL named font program --
# GID alignment is exact by construction (verified on this corpus: drawn
# (gid,unicode) pairs match the system font's cmap inversion). All fonts below
# are macOS-shipped Microsoft fonts with fsType=8 (editable embedding allowed).
CID_REAL = {
    "Arial-Black":   "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "Arial":         "/System/Library/Fonts/Supplemental/Arial.ttf",
    "ArialMT":       "/System/Library/Fonts/Supplemental/Arial.ttf",
    "Arial-BoldMT":  "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "Arial,Bold":    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "Arial-ItalicMT":"/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "ArialNarrow":   "/System/Library/Fonts/Supplemental/Arial Narrow.ttf",
}

# Fallback donor-synthesis (GID==CID glyphs copied from an open substitute via
# the font's existing ToUnicode). Only sound when that ToUnicode is real -- it
# DECLINES if any mapped CID fails to resolve in the donor (an all-FFFD map,
# as found on this corpus, must never yield a font full of empty outlines).
CID_DONORS = {}

# SUBSET-UPGRADE: embedded subsets that MISS glyphs the content uses (drawn
# .notdef -> 7.21.8-1, and 7.21.4.1-2). Replacing the subset with the REAL full
# font resolves the code (simple fonts: code->cmap, no glyph-ID semantics; Type0
# Identity: only behind the empirical GID-alignment gate). fsType: Arial family
# & Times = 8 (editable), HelveticaNeue = 0, GillSans = 6 (preview&print --
# embedding permitted for view/print documents; flagged in the report).
_SUP = "/System/Library/Fonts/Supplemental/"
REAL_UPGRADES = {
    "ArialMT":                    (_SUP + "Arial.ttf", 0),
    "Arial-BoldMT":               (_SUP + "Arial Bold.ttf", 0),
    "Arial-ItalicMT":             (_SUP + "Arial Italic.ttf", 0),
    "Arial-BoldItalicMT":         (_SUP + "Arial Bold Italic.ttf", 0),
    "ArialNarrow":                (_SUP + "Arial Narrow.ttf", 0),
    "Arial Narrow":               (_SUP + "Arial Narrow.ttf", 0),
    "ArialNarrow-Bold":           (_SUP + "Arial Narrow Bold.ttf", 0),
    "Arial Narrow Bold":          (_SUP + "Arial Narrow Bold.ttf", 0),
    "Arial Narrow,Bold":          (_SUP + "Arial Narrow Bold.ttf", 0),
    "ArialNarrow-Italic":         (_SUP + "Arial Narrow Italic.ttf", 0),
    "ArialNarrow-BoldItalic":     (_SUP + "Arial Narrow Bold Italic.ttf", 0),
    "TimesNewRomanPSMT":          (_SUP + "Times New Roman.ttf", 0),
    "TimesNewRomanPS-BoldMT":     (_SUP + "Times New Roman Bold.ttf", 0),
    "TimesNewRomanPS-ItalicMT":   (_SUP + "Times New Roman Italic.ttf", 0),
    "GillSans":                   (_SUP + "GillSans.ttc", 0),
    "GillSans-Bold":              (_SUP + "GillSans.ttc", 1),
    "GillSans-Italic":            (_SUP + "GillSans.ttc", 2),
    "GillSans-BoldItalic":        (_SUP + "GillSans.ttc", 3),
    "HelveticaNeue-Roman":        ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
    "HelveticaNeue":              ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
    "HelveticaNeue-Bold":         ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
    "HelveticaNeue-Italic":       ("/System/Library/Fonts/HelveticaNeue.ttc", 2),
    "HelveticaNeue-BoldItalic":   ("/System/Library/Fonts/HelveticaNeue.ttc", 3),
}

def _code_unicode_map(obj):
    """Simple font dict -> {code:int -> unicode int} from /Encoding
    (Differences names via AGL take precedence over the base codec)."""
    from fontTools import agl
    enc = obj.get("/Encoding")
    base = enc if isinstance(enc, pikepdf.Name) else (
        enc.get("/BaseEncoding") if isinstance(enc, pikepdf.Dictionary) else None)
    codec = {"/WinAnsiEncoding": "cp1252",
             "/MacRomanEncoding": "mac_roman"}.get(str(base) if base else "")
    diffs = {}
    if isinstance(enc, pikepdf.Dictionary) and "/Differences" in enc:
        cur = 0
        for it in enc["/Differences"]:
            if isinstance(it, int): cur = it
            else: diffs[cur] = str(it).lstrip("/"); cur += 1
    out = {}
    lo = int(obj.get("/FirstChar", 0)); hi = int(obj.get("/LastChar", 255))
    widths = [int(w) for w in (obj.get("/Widths") or [])]
    for i, w in enumerate(widths):
        code = lo + i
        if w <= 0:
            continue
        if code in diffs:
            u = agl.toUnicode(diffs[code])
            if len(u) == 1:
                out[code] = ord(u)
        elif codec:
            try:
                u = bytes([code]).decode(codec)
                if u >= " " and code != 0x7F:
                    out[code] = ord(u)
            except Exception:
                pass
    return out

def _program_covers(fd, unicodes):
    """Does the embedded program have a real glyph for every unicode? Handles
    TrueType (cmap) and Type1 (builtin encoding names / AGL names)."""
    import io as _io
    try:
        if "/FontFile2" in fd:
            tt = TTFont(_io.BytesIO(bytes(fd["/FontFile2"].read_bytes())), fontNumber=0)
            cmap = {}
            for t in tt["cmap"].tables:
                if (t.platformID, t.platEncID) in ((3, 1), (0, 3), (0, 4)):
                    cmap.update(t.cmap)
            sym = next((t.cmap for t in tt["cmap"].tables
                        if (t.platformID, t.platEncID) == (3, 0)), {})
            missing = [u for u in unicodes
                       if u not in cmap and (0xF000 | (u & 0xFF)) not in sym]
            return not missing, missing
        if "/FontFile" in fd:
            from fontTools import t1Lib, agl as _agl
            import tempfile as _tf
            ff = fd["/FontFile"]
            raw = bytes(ff.read_bytes())
            l1 = int(ff.get("/Length1", 0)); l2 = int(ff.get("/Length2", 0))
            l3 = int(ff.get("/Length3", 0))
            def hdr(t, n): return bytes([0x80, t]) + n.to_bytes(4, "little")
            seg3 = raw[l1+l2:l1+l2+l3] if l3 else raw[l1+l2:]
            pfb = hdr(1, l1) + raw[:l1] + hdr(2, l2) + raw[l1:l1+l2] + \
                  hdr(1, len(seg3)) + seg3 + bytes([0x80, 3])
            with _tf.NamedTemporaryFile(suffix=".pfb", delete=False) as tf:
                tf.write(pfb); tmp = tf.name
            try:
                f = t1Lib.T1Font(tmp); f.parse()
            finally:
                try: os.unlink(tmp)
                except OSError: pass
            names = set(f.font["CharStrings"].keys())
            missing = []
            for u in unicodes:
                name = _agl.UV2AGL.get(u)
                if name is None or name not in names:
                    missing.append(u)
            return not missing, missing
    except Exception:
        return False, ["unparseable"]
    return True, []

def _upgrade_simple(pdf, obj, fd, path, font_number):
    """Swap an incomplete simple-font subset for the REAL full font.
    Safe: simple fonts select glyphs code->cmap, no glyph-ID semantics.
    hmtx is patched to the PDF /Widths so 7.21.5 holds and layout is unchanged.
    A Type1 source dict is converted to /TrueType (program is TrueType now)."""
    tt = TTFont(path, fontNumber=font_number)
    upem = tt["head"].unitsPerEm
    cmap = tt.getBestCmap()
    hmtx = tt["hmtx"]
    c2u = _code_unicode_map(obj)
    lo = int(obj.get("/FirstChar", 0))
    widths = [int(w) for w in (obj.get("/Widths") or [])]
    for i, w in enumerate(widths):
        code = lo + i
        u = c2u.get(code)
        gname = cmap.get(u) if u is not None else None
        if gname is None or w <= 0:
            continue
        adv = round(w * upem / 1000)
        _, lsb = hmtx[gname]
        hmtx[gname] = (adv, lsb)
    buf = io.BytesIO()
    tt.save(buf)
    data = buf.getvalue()
    stream = pdf.make_stream(data)
    stream.stream_dict["/Length1"] = len(data)
    for k in ("/FontFile", "/FontFile3"):
        if k in fd:
            del fd[k]
    fd["/FontFile2"] = stream
    obj["/Subtype"] = pikepdf.Name("/TrueType")

def _usage_scan(path):
    """Content-level font usage via PyMuPDF texttrace: which font names draw
    .notdef (gid 0), and each font's max drawn gid (Type0 Identity: gid==CID,
    reported raw, so overruns past the subset are visible here and ONLY here --
    /W and ToUnicode routinely under-declare)."""
    import fitz
    notdef, maxgid = {}, {}
    doc = fitz.open(path)
    for pno in range(doc.page_count):
        for span in doc[pno].get_texttrace():
            f = span.get("font", "")
            for ch in span["chars"]:
                gid = ch[1]
                if gid == 0:
                    notdef[f] = notdef.get(f, 0) + 1
                if gid > maxgid.get(f, -1):
                    maxgid[f] = gid
    doc.close()
    return notdef, maxgid

def _font_matches(short, used_name):
    """Match a PDF BaseFont short name against a texttrace font name."""
    a = short.replace(" ", "").lower()
    b = used_name.replace(" ", "").lower()
    return a == b or a.endswith(b) or b.endswith(a)

def _type0_needs_upgrade(top, cidfont, fd, maxgid_by_font):
    """True if the embedded Identity subset provably misses used glyphs:
    a CONTENT-drawn CID is out of the program's range, or a declared CID
    (from /W or ToUnicode) is."""
    try:
        tt = TTFont(io.BytesIO(bytes(fd["/FontFile2"].read_bytes())), fontNumber=0)
    except Exception:
        return False
    n = tt["maxp"].numGlyphs
    short = str(top.get("/BaseFont", "")).lstrip("/").split("+")[-1]
    for used_name, mg in maxgid_by_font.items():
        if _font_matches(short, used_name) and mg >= n:
            return True
    declared = set(_parse_w_array(cidfont.get("/W"), 0))
    declared |= set(_tounicode_cids(top) or {})
    return any(c >= n for c in declared)

def _upgrade_type0(pdf, top, cidfont, fd, path, font_number, warnings):
    """Swap an incomplete Identity CID subset for the REAL full font -- ONLY if
    the subset's CIDs empirically ARE the real font's glyph IDs: >=90% of the
    existing ToUnicode's known (cid,uni) pairs must match the real font's cmap
    inversion (>=5 usable pairs). Misaligned subsets are declined."""
    m = cidfont.get("/CIDToGIDMap")
    if m is not None and m != pikepdf.Name("/Identity"):
        warnings.append("upgrade: CIDToGIDMap not Identity -> declined")
        return False
    cid2uni = _tounicode_cids(top) or {}
    pairs = [(c, ord(u)) for c, u in cid2uni.items()
             if len(u) == 1 and ord(u) not in (0, 0xFEFF, 0xFFFE, 0xFFFD)]
    if len(pairs) < 5:
        warnings.append("upgrade: <5 usable ToUnicode pairs -> declined")
        return False
    tt = TTFont(path, fontNumber=font_number)
    best = tt.getBestCmap()
    n2g = {n: i for i, n in enumerate(tt.getGlyphOrder())}
    g2u = {}
    for u, gn in best.items():
        g2u.setdefault(n2g[gn], set()).add(u)
    ok = sum(1 for c, u in pairs if u in g2u.get(c, ()))
    if ok / len(pairs) < 0.90:
        warnings.append(f"upgrade: GID alignment {ok}/{len(pairs)} -> declined")
        return False
    upem = tt["head"].unitsPerEm
    hmtx = tt["hmtx"]
    order = tt.getGlyphOrder()
    dw = int(cidfont.get("/DW", 1000))
    for cid, w in _parse_w_array(cidfont.get("/W"), dw).items():
        if cid < len(order):
            _, lsb = hmtx[order[cid]]
            hmtx[order[cid]] = (round(w * upem / 1000), lsb)
    buf = io.BytesIO()
    tt.save(buf)
    data = buf.getvalue()
    return _attach_cid_program(pdf, top, cidfont, fd, m, data)

def _parse_w_array(w, dw):
    """/W array -> {cid: width}. Handles both [c [w..]] and [c1 c2 w] forms."""
    out = {}
    if w is None:
        return out
    items = list(w)
    i = 0
    while i < len(items):
        a = int(items[i])
        nxt = items[i+1]
        if isinstance(nxt, pikepdf.Array) or isinstance(nxt, list):
            for j, ww in enumerate(nxt):
                out[a + j] = int(ww)
            i += 2
        else:
            b, ww = int(nxt), int(items[i+2])
            for c in range(a, b + 1):
                out[c] = ww
            i += 3
    return out

def _tounicode_cids(font_obj):
    """Existing /ToUnicode of a Type0 font -> {cid: unistr} (bfchar + bfrange)."""
    if "/ToUnicode" not in font_obj:
        return None
    text = bytes(font_obj["/ToUnicode"].read_bytes()).decode("latin-1", "replace")
    out = {}
    for m in re.finditer(r"beginbfchar(.*?)endbfchar", text, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", m.group(1)):
            dstb = bytes.fromhex(dst if len(dst) % 2 == 0 else "0" + dst)
            out[int(src, 16)] = dstb.decode("utf-16-be", "replace")
    for m in re.finditer(r"beginbfrange(.*?)endbfrange", text, re.S):
        for lo, hi, start in re.findall(
                r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", m.group(1)):
            base = int.from_bytes(bytes.fromhex(start if len(start) % 2 == 0 else "0"+start), "big")
            for i, cid in enumerate(range(int(lo, 16), int(hi, 16) + 1)):
                try: out[cid] = chr(base + i)
                except ValueError: pass
    return out

def _build_cid_aligned_ttf(donor_path, cid2uni, cid2w, dw):
    """Synthetic TTF with GID==CID. Returns (bytes, unresolved_cids)."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.pens.recordingPen import DecomposingRecordingPen
    donor = TTFont(donor_path, fontNumber=0)
    upem = donor["head"].unitsPerEm
    dcmap = donor.getBestCmap()
    dgs = donor.getGlyphSet()
    scale = upem / 1000.0
    max_cid = max(cid2uni)
    order = [".notdef"] + [f"cid{c:05d}" for c in range(1, max_cid + 1)]
    glyphs, metrics, cmap, unresolved = {}, {}, {}, []
    # .notdef: empty glyph
    pen0 = TTGlyphPen(None)
    glyphs[".notdef"] = pen0.glyph()
    metrics[".notdef"] = (int(round(dw * scale)), 0)
    for c in range(1, max_cid + 1):
        gname = f"cid{c:05d}"
        adv = int(round(cid2w.get(c, dw) * scale))
        uni = cid2uni.get(c)
        dname = dcmap.get(ord(uni)) if uni and len(uni) == 1 else None
        if dname is None:
            if uni is not None:
                unresolved.append(c)
            pen = TTGlyphPen(None)
            glyphs[gname] = pen.glyph()
            metrics[gname] = (adv, 0)
            continue
        rp = DecomposingRecordingPen(dgs)
        dgs[dname].draw(rp)
        pen = TTGlyphPen(None)
        rp.replay(pen)
        glyphs[gname] = pen.glyph()
        metrics[gname] = (adv, dgs[dname].lsb if hasattr(dgs[dname], "lsb") else 0)
        if uni and len(uni) == 1 and ord(uni) not in cmap:
            cmap[ord(uni)] = gname
    fb = FontBuilder(upem, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    asc = donor["hhea"].ascent; desc = donor["hhea"].descent
    fb.setupHorizontalHeader(ascent=asc, descent=desc)
    fb.setupNameTable({"familyName": "CIDAligned", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=asc, sTypoDescender=desc, usWinAscent=max(asc, 0),
                usWinDescent=max(-desc, 0))
    fb.setupPost()
    buf = io.BytesIO()
    fb.save(buf)
    return buf.getvalue(), unresolved

def _embed_cid_real(pdf, top, cidfont, fd, real_path, warnings):
    """Embed the REAL named font into a non-embedded Identity CIDFontType2.
    CIDs index the program's own glyphs directly -- no synthesis, no remap."""
    m = cidfont.get("/CIDToGIDMap")
    if m is not None and m != pikepdf.Name("/Identity"):
        warnings.append("CIDToGIDMap not Identity -> skipped")
        return False
    data = open(real_path, "rb").read()
    return _attach_cid_program(pdf, top, cidfont, fd, m, data)

def _embed_cid(pdf, top, cidfont, fd, donor_path, warnings):
    """Fallback: GID==CID synthetic program from a donor via existing ToUnicode.
    DECLINES unless every mapped CID resolves to a real donor glyph."""
    m = cidfont.get("/CIDToGIDMap")
    if m is not None and m != pikepdf.Name("/Identity"):
        warnings.append("CIDToGIDMap not Identity -> skipped")
        return False
    cid2uni = _tounicode_cids(top)
    if not cid2uni:
        warnings.append("Type0 without ToUnicode -> cannot align CIDs -> skipped")
        return False
    dw = int(cidfont.get("/DW", 1000))
    cid2w = _parse_w_array(cidfont.get("/W"), dw)
    data, unresolved = _build_cid_aligned_ttf(donor_path, cid2uni, cid2w, dw)
    if unresolved:
        # empty outlines for mapped CIDs would pass veraPDF but render blank
        # text -- strictly worse than failing. Decline.
        warnings.append(f"donor cannot resolve {len(unresolved)} mapped CIDs "
                        f"(e.g. {unresolved[:5]}) -> declined")
        return False
    return _attach_cid_program(pdf, top, cidfont, fd, m, data)

def _attach_cid_program(pdf, top, cidfont, fd, m, data):
    stream = pdf.make_stream(data)
    stream.stream_dict["/Length1"] = len(data)
    # UA-1 7.21.3.2: an EMBEDDED CIDFontType2 must carry an explicit
    # /CIDToGIDMap (the spec default that applied while non-embedded no longer
    # satisfies veraPDF once a program is present).
    if m is None:
        cidfont["/CIDToGIDMap"] = pikepdf.Name("/Identity")
    if fd is None:
        base = str(top.get("/BaseFont", "/Unknown")).lstrip("/")
        fd = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/FontDescriptor"),
            "/FontName": pikepdf.Name("/" + base),
            "/Flags": 32, "/FontBBox": pikepdf.Array([-200, -250, 1200, 1000]),
            "/ItalicAngle": 0, "/Ascent": 1000, "/Descent": -250,
            "/CapHeight": 700, "/StemV": 160,
        }))
        cidfont["/FontDescriptor"] = fd
    fd["/FontFile2"] = stream
    return True

def embed(inp, outp):
    pdf = pikepdf.open(inp)
    changed, warnings = [], []
    try:
        notdef_fonts, maxgid_by_font = _usage_scan(inp)
    except Exception:
        notdef_fonts, maxgid_by_font = {}, {}
    for obj in pdf.objects:
        try:
            if not (isinstance(obj, pikepdf.Dictionary)
                    and obj.get("/Type") == pikepdf.Name("/Font")):
                continue
        except Exception:
            continue
        sub = obj.get("/Subtype")
        # Type0 with a non-embedded CIDFontType2 descendant: embed the REAL named
        # font (GIDs align by construction); donor synthesis only as fallback.
        # An EMBEDDED-but-incomplete Identity subset (declared CIDs beyond the
        # program, or empty outlines for mapped CIDs) is upgraded to the real
        # full font behind the empirical GID-alignment gate.
        if sub == pikepdf.Name("/Type0"):
            for cidfont in (obj.get("/DescendantFonts") or []):
                cfd = cidfont.get("/FontDescriptor")
                if _is_embedded(cfd):
                    cbase = str(obj.get("/BaseFont", "")).lstrip("/").split("+")[-1]
                    up = REAL_UPGRADES.get(cbase)
                    if up and os.path.exists(up[0]) \
                            and _type0_needs_upgrade(obj, cidfont, cfd, maxgid_by_font):
                        if _upgrade_type0(pdf, obj, cidfont, cfd, up[0], up[1], warnings):
                            changed.append((cbase, os.path.basename(up[0]) + " (upgrade)"))
                    continue
                cbase = str(obj.get("/BaseFont", "")).lstrip("/").split("+")[-1]
                real = CID_REAL.get(cbase)
                if real and os.path.exists(real):
                    if _embed_cid_real(pdf, obj, cidfont, cfd, real, warnings):
                        changed.append((cbase, os.path.basename(real) + " (real)"))
                    continue
                donor = CID_DONORS.get(cbase)
                if donor is None or not os.path.exists(donor):
                    if cbase:
                        warnings.append(f"no CID source for {cbase}")
                    continue
                if _embed_cid(pdf, obj, cidfont, cfd, donor, warnings):
                    changed.append((cbase, os.path.basename(donor) + " (GID==CID)"))
            continue
        if sub not in (pikepdf.Name("/TrueType"), pikepdf.Name("/Type1")):
            continue
        fd = obj.get("/FontDescriptor")
        if fd is not None and _is_embedded(fd):
            # embedded-but-incomplete simple subset: upgrade to the real font
            # (safe: simple fonts select glyphs code->cmap, no GID semantics)
            short = str(obj.get("/BaseFont", "")).lstrip("/").split("+")[-1]
            up = REAL_UPGRADES.get(short)
            if up and os.path.exists(up[0]):
                # GUARD: only fonts whose glyph selection is fully determined by
                # an explicit standard encoding may swap programs. A symbolic
                # font (no /Encoding) selects via ITS OWN cmap layout -- a real
                # font's different layout would break selection (verified: the
                # gate discarded exactly such an upgrade as a regression).
                enc = obj.get("/Encoding")
                base_enc = enc if isinstance(enc, pikepdf.Name) else (
                    enc.get("/BaseEncoding") if isinstance(enc, pikepdf.Dictionary) else None)
                std_enc = str(base_enc) in ("/WinAnsiEncoding", "/MacRomanEncoding")
                # trigger 1: content actually draws .notdef through this font
                draws_notdef = any(_font_matches(short, f) for f in notdef_fonts)
                # trigger 2: program lacks a glyph for a declared /Widths code
                needs = draws_notdef
                if not needs:
                    c2u = _code_unicode_map(obj)
                    if c2u:
                        covered, _miss = _program_covers(fd, sorted(set(c2u.values())))
                        needs = not covered
                if needs and not std_enc:
                    warnings.append(f"{short}: no standard /Encoding -> upgrade declined")
                elif needs:
                    _upgrade_simple(pdf, obj, fd, up[0], up[1])
                    changed.append((short, os.path.basename(up[0]) + " (upgrade)"))
            continue
        base = str(obj.get("/BaseFont", "")).lstrip("/")
        short = base.split("+")[-1]
        # standard-14 Type1 (AcroForm /ZaDb, /Helv): URW-clone Type1 route
        if sub == pikepdf.Name("/Type1") and short in STD14 \
                and os.path.exists(STD14[short][0]):
            _embed_std14(pdf, obj, fd, short)
            changed.append((short, os.path.basename(STD14[short][0])))
            continue
        if fd is None:
            continue                                 # bare non-ZaDb std-14: no descriptor to extend
        path = SUBSTITUTES.get(short)
        if path is None or not os.path.exists(path):
            warnings.append(f"no substitute for {short}")
            continue
        enc = obj.get("/Encoding")
        if str(enc) != "/WinAnsiEncoding" and not (
                isinstance(enc, pikepdf.Dictionary)
                and str(enc.get("/BaseEncoding")) == "/WinAnsiEncoding"):
            warnings.append(f"{short}: encoding {enc} not WinAnsi -> skipped")
            continue
        widths = [int(w) for w in (obj.get("/Widths") or [])]
        firstchar = int(obj.get("/FirstChar", 0))
        data, is_cff, missing = _patched_program(path, widths, firstchar)
        if missing:
            # /Widths coverage != usage; embed anyway and let the veraPDF
            # before/after gate reject if a USED glyph is actually absent.
            warnings.append(f"{short}: substitute lacks glyphs for declared codes {missing}")
        stream = pdf.make_stream(data)
        if is_cff:
            stream.stream_dict["/Subtype"] = pikepdf.Name("/OpenType")
            fd["/FontFile3"] = stream
        else:
            stream.stream_dict["/Length1"] = len(data)
            fd["/FontFile2"] = stream
        changed.append((short, os.path.basename(path)))
    if changed:
        pdf.save(outp)
    pdf.close()
    return changed, warnings

if __name__ == "__main__":
    inp, outp = sys.argv[1], sys.argv[2]
    ch, warns = embed(inp, outp)
    for b, p in ch:
        print(f"  embedded: {b} <- {p}")
    for w in warns:
        print(f"  ! {w}")
    print(f"{'wrote' if ch else 'no-change'}: {len(ch)} font(s)")
    sys.exit(0 if ch else 2)
