#!/usr/bin/env python3
"""Repair content-stream text corruption that draws .notdef (veraPDF 7.21.8-1).

Two producer/pipeline defects, both fixed at the parsed-operator level
(pikepdf.parse_content_stream / unparse_content_stream, so every string
encoding -- literal, octal escape, hex -- is handled uniformly):

1. SPLICED MARKED-CONTENT OPERATORS inside a show string. An earlier
   tag-injection pass inserted ``EMC`` / ``/P <</MCID n>> BDC`` at byte offsets
   INSIDE ``(...)Tj`` literals, e.g. ``(Meets Cal-GET\\nEMC\\n/P <</MCID 2>>
   BDC\\nC-3A)Tj``. The newlines render as .notdef and the operator text is
   DRAWN as garbage. The injected scope never worked and its MCID has no parent
   in the structure tree (verified: materializing it just yields 7.1-3), so the
   repair deletes the fragments and merges the halves back into the original
   single string: ``(Meets Cal-GETC-3A)Tj``.

2. CONTROL BYTES (tab, LF, ...) inside show strings. No font has glyphs for
   them; each draws .notdef with the .notdef advance. They are replaced by an
   exact-width TJ kern: ``(\\t\\t\\tText)Tj`` -> ``[ -3*adv (Text) ] TJ`` --
   rendering-identical, zero glyph-0 references. The advance is read from the
   ACTIVE font's embedded program (.notdef hmtx / CharStrings width); if the
   font cannot be resolved the site is skipped and reported.

Idempotent. Usage: fix_content_splices.py in.pdf out.pdf (exit 0 wrote / 2 none)
"""
import io, re, sys
import pikepdf

SPLICE = re.compile(
    rb"\n(?:EMC\n)?(?:/[A-Za-z]+\s*<</MCID \d+>>\s*BDC\n?)"   # .. EMC /P <<MCID>> BDC ..
    rb"|\nEMC\n"                                              # bare EMC
    rb"|/[A-Za-z]+\s+BMC\n*")                                 # /Artifact BMC (mid-word!)
# every control byte draws .notdef (no font maps them); the splice pass above
# consumes operator-fragment newlines first, so whatever remains is a dead draw
CTRL = re.compile(rb"[\x00-\x1f]+")


def _notdef_adv1000(font_obj):
    """.notdef advance of the font's embedded program, in 1/1000 em units."""
    fd = font_obj.get("/FontDescriptor")
    df = font_obj.get("/DescendantFonts")
    if df is not None:
        fd = df[0].get("/FontDescriptor")
    if fd is None:
        return None
    try:
        if "/FontFile2" in fd:
            from fontTools.ttLib import TTFont
            tt = TTFont(io.BytesIO(bytes(fd["/FontFile2"].read_bytes())), fontNumber=0)
            adv = tt["hmtx"][tt.getGlyphOrder()[0]][0]
            return adv * 1000.0 / tt["head"].unitsPerEm
        if "/FontFile" in fd:
            from fontTools import t1Lib
            from fontTools.pens.basePen import NullPen
            import tempfile, os as _os
            ff = fd["/FontFile"]
            raw = bytes(ff.read_bytes())
            l1 = int(ff.get("/Length1", 0)); l2 = int(ff.get("/Length2", 0))
            l3 = int(ff.get("/Length3", 0))
            def hdr(t, n): return bytes([0x80, t]) + n.to_bytes(4, "little")
            seg3 = raw[l1+l2:l1+l2+l3] if l3 else raw[l1+l2:]
            pfb = hdr(1, l1) + raw[:l1] + hdr(2, l2) + raw[l1:l1+l2] + \
                  hdr(1, len(seg3)) + seg3 + bytes([0x80, 3])
            with tempfile.NamedTemporaryFile(suffix=".pfb", delete=False) as tf:
                tf.write(pfb); tmp = tf.name
            try:
                f = t1Lib.T1Font(tmp); f.parse()
            finally:
                try: _os.unlink(tmp)
                except OSError: pass
            g = f.font["CharStrings"].get(".notdef")
            if g is None:
                return None
            g.draw(NullPen())
            return float(g.width)
    except Exception:
        return None
    return None


def _transform_string(b, adv, report):
    """Decoded show-string bytes -> (list of TJ items | None if unchanged).
    Items are (bytes-string) or int kern. Applies splice-merge then ctrl-kern."""
    merged = SPLICE.sub(b"", b)                     # pass 1: drop dead operator text
    has_ctrl = CTRL.search(merged)
    if merged == b and not has_ctrl:
        return None
    if not has_ctrl:
        return [merged]
    if adv is None:
        report.append("ctrl bytes but active font unresolved -> site skipped")
        return None if merged == b else [merged]   # at least apply the merge
    # rebuild as (text, kern, text, ...) preserving exact positions
    items = []
    pos = 0
    for m in CTRL.finditer(merged):
        if m.start() > pos:
            items.append(merged[pos:m.start()])
        items.append(int(round(-len(m.group(0)) * adv)))
        pos = m.end()
    if pos < len(merged):
        items.append(merged[pos:])
    return items


def fix(inp, outp):
    pdf = pikepdf.open(inp)
    report, n_pages = [], 0
    for page in pdf.pages:
        if page.get("/Contents") is None:
            continue
        try:
            ops = pikepdf.parse_content_stream(page)
        except Exception as e:
            report.append(f"unparseable content stream: {e}")
            continue
        fonts = {}
        res = page.get("/Resources")
        if res is not None and "/Font" in res:
            fonts = {str(k).lstrip("/"): v for k, v in res["/Font"].items()}
        cur_adv, cur_simple, changed, new_ops = None, False, False, []
        for operands, op in ops:
            o = str(op)
            if o == "Tf" and len(operands) == 2:
                name = str(operands[0]).lstrip("/")
                fobj = fonts.get(name)
                # ONLY simple (1-byte-code) fonts are transformable: a Type0
                # string is a sequence of 2-byte CIDs whose 0x00/low bytes are
                # NOT control characters -- rewriting them shreds valid glyph
                # references (verified: the gate discarded exactly that).
                cur_simple = fobj is not None and str(fobj.get("/Subtype")) in (
                    "/TrueType", "/Type1", "/Type3", "/MMType1")
                cur_adv = _notdef_adv1000(fobj) if cur_simple else None
                new_ops.append((operands, op))
                continue
            if not cur_simple:
                new_ops.append((operands, op))
                continue
            if o == "Tj" and operands and isinstance(operands[0], pikepdf.String):
                items = _transform_string(bytes(operands[0]), cur_adv, report)
                if items is None:
                    new_ops.append((operands, op))
                    continue
                changed = True
                arr = pikepdf.Array(
                    [pikepdf.String(i) if isinstance(i, bytes) else i for i in items])
                new_ops.append(([arr], pikepdf.Operator("TJ")))
                continue
            if o == "TJ" and operands and isinstance(operands[0], pikepdf.Array):
                out_items, tj_changed = [], False
                for it in operands[0]:
                    if isinstance(it, pikepdf.String):
                        items = _transform_string(bytes(it), cur_adv, report)
                        if items is None:
                            out_items.append(it)
                        else:
                            tj_changed = True
                            out_items.extend(
                                pikepdf.String(i) if isinstance(i, bytes) else i
                                for i in items)
                    else:
                        out_items.append(it)
                if tj_changed:
                    changed = True
                    new_ops.append(([pikepdf.Array(out_items)], pikepdf.Operator("TJ")))
                else:
                    new_ops.append((operands, op))
                continue
            new_ops.append((operands, op))
        if changed:
            page["/Contents"] = pdf.make_stream(
                pikepdf.unparse_content_stream(new_ops))
            n_pages += 1
    if n_pages:
        pdf.save(outp)
    pdf.close()
    return n_pages, report


def _simple_uncovered_codes(font_obj):
    """Codes a simple font's program has NO glyph for (drawn -> .notdef).
    Returns (set of codes, {code: advance/1000}) or (None, None) if unknown."""
    fd = font_obj.get("/FontDescriptor")
    if fd is None:
        return None, None
    lo = int(font_obj.get("/FirstChar", 0))
    widths = [int(w) for w in (font_obj.get("/Widths") or [])]
    if not widths:
        return None, None
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    from embed_missing_fonts import _code_unicode_map, _program_covers
    c2u = _code_unicode_map(font_obj)
    if not c2u:
        return None, None
    covered, missing_unis = _program_covers(fd, sorted(set(c2u.values())))
    if covered:
        return set(), {}
    if missing_unis == ["unparseable"]:
        return None, None
    missing_unis = set(missing_unis)
    codes = {c for c, u in c2u.items() if u in missing_unis}
    advs = {}
    for c in codes:
        i = c - lo
        advs[c] = float(widths[i]) if 0 <= i < len(widths) else 0.0
    return codes, advs

def _cid_dead_set(top, cidfont):
    """CIDs that reference .notdef by construction: CID 0 always; CIDs >= the
    embedded program's glyph count. Returns (dead_predicate, {cid->adv})."""
    import io as _io
    from fontTools.ttLib import TTFont as _TT
    fd = cidfont.get("/FontDescriptor")
    n = None
    if fd is not None and "/FontFile2" in fd:
        try:
            n = _TT(_io.BytesIO(bytes(fd["/FontFile2"].read_bytes())), fontNumber=0)["maxp"].numGlyphs
        except Exception:
            n = None
    dw = float(cidfont.get("/DW", 1000))
    w = {}
    arr = cidfont.get("/W")
    if arr is not None:
        items = list(arr); i = 0
        while i < len(items):
            a = int(items[i]); nx = items[i+1]
            if isinstance(nx, pikepdf.Array) or isinstance(nx, list):
                for j, ww in enumerate(nx): w[a+j] = float(ww)
                i += 2
            else:
                for c in range(a, int(nx)+1): w[c] = float(items[i+2])
                i += 3
    def dead(cid):
        return cid == 0 or (n is not None and cid >= n)
    return dead, (lambda cid: w.get(cid, dw))

def fix_dead_refs(inp, outp):
    """LATE pass (after embed/upgrades): kern-out glyph references that are
    dead by construction and that no font work can restore --
      * simple fonts: codes whose program has no glyph (space dropped by the
        subsetter, etc.); advance = the PDF's own /Widths entry, so layout is
        byte-exact;
      * Type0 Identity: CID 0 (always .notdef) and CIDs beyond the program.
    Rendering is unchanged (these draws produce no ink), veraPDF 7.21.8-1 and
    7.21.4.1-2 lose their trigger."""
    # SAFETY GATE (learned the hard way): only fonts OBSERVED drawing .notdef
    # (gid 0 via texttrace) are eligible for simple-mode kerning. Without this,
    # a coverage misfire nukes text that extracts fine -- Tesseract's
    # GlyphLessFont "covers nothing" by program inspection, and kerning it
    # deletes the entire invisible OCR text layer while veraPDF still passes.
    notdef_fonts = set()
    try:
        import fitz
        _doc = fitz.open(inp)
        for _pno in range(_doc.page_count):
            for _span in _doc[_pno].get_texttrace():
                for _ch in _span["chars"]:
                    if _ch[1] == 0:
                        notdef_fonts.add(_span.get("font", ""))
        _doc.close()
    except Exception as e:
        report_boot = [f"usage scan failed ({e}); simple-mode kerning disabled"]
        notdef_fonts = set()
    pdf = pikepdf.open(inp)
    report, n_pages = [], 0
    for page in pdf.pages:
        if page.get("/Contents") is None:
            continue
        try:
            ops = pikepdf.parse_content_stream(page)
        except Exception as e:
            report.append(f"unparseable content stream: {e}")
            continue
        fonts = {}
        res = page.get("/Resources")
        if res is not None and "/Font" in res:
            fonts = {str(k).lstrip("/"): v for k, v in res["/Font"].items()}
        mode = None            # ("simple", codes, advs) | ("cid", dead, advfn)
        changed, new_ops = False, []
        def xform_simple(b, codes, advs):
            runs, pos, items = [], 0, []
            i = 0
            any_hit = False
            while i < len(b):
                if b[i] in codes:
                    j = i
                    total = 0.0
                    while j < len(b) and b[j] in codes:
                        total += advs.get(b[j], 0.0); j += 1
                    if b[pos:i]:
                        items.append(b[pos:i])
                    items.append(int(round(-total)))
                    pos = j; i = j; any_hit = True
                else:
                    i += 1
            if not any_hit:
                return None
            if b[pos:]:
                items.append(b[pos:])
            return items
        def xform_cid(b, dead, advfn):
            if len(b) % 2:
                return None
            items, pos, any_hit = [], 0, False
            for i in range(0, len(b), 2):
                cid = (b[i] << 8) | b[i+1]
                if dead(cid):
                    if b[pos:i]:
                        items.append(b[pos:i])
                    items.append(int(round(-advfn(cid))))
                    pos = i + 2; any_hit = True
            if not any_hit:
                return None
            if b[pos:]:
                items.append(b[pos:])
            return items
        for operands, op in ops:
            o = str(op)
            if o == "Tf" and len(operands) == 2:
                name = str(operands[0]).lstrip("/")
                fobj = fonts.get(name)
                mode = None
                if fobj is not None:
                    sub = str(fobj.get("/Subtype"))
                    base = str(fobj.get("/BaseFont", "")).lstrip("/").split("+")[-1]
                    observed = any(
                        base.replace(" ", "").lower() == n.replace(" ", "").lower()
                        or base.replace(" ", "").lower().endswith(n.replace(" ", "").lower())
                        or n.replace(" ", "").lower().endswith(base.replace(" ", "").lower())
                        for n in notdef_fonts if n)
                    # Tesseract's GlyphLessFont draws gid 0 BY DESIGN (that is
                    # the OCR text layer; veraPDF does not flag it) -- never
                    # touch it: kerning it deletes the entire extractable text.
                    if "glyphless" in base.lower():
                        observed = False
                    if sub in ("/TrueType", "/Type1") and "glyphless" not in base.lower():
                        # ONLY inkless codes are kernable. Printable-glyph
                        # kerning is forbidden -- a coverage misfire deletes
                        # VISIBLE text (fidelity-gate history).
                        #  * control bytes: dead under every consumer's
                        #    semantics (veraPDF's encoding-name path flags them
                        #    even when a cmap resolves them) -> always kernable
                        #  * space: only when OBSERVED drawing .notdef and the
                        #    program provably lacks the glyph
                        codes, advs = set(), {}
                        if observed:
                            unc, uadv = _simple_uncovered_codes(fobj)
                            if unc and 0x20 in unc:
                                codes.add(0x20)
                                advs[0x20] = uadv.get(0x20, 0.0)
                        nd = _notdef_adv1000(fobj)
                        if nd is not None:
                            for c in range(0x20):
                                codes.add(c)
                                advs.setdefault(c, nd)
                        if codes:
                            mode = ("simple", codes, advs)
                    elif sub == "/Type0" and "glyphless" not in base.lower():
                        # GlyphLessFont is Type0 with a 1-glyph program: every
                        # OCR-text CID would look "out of range" and the whole
                        # extractable text layer would be kerned away.
                        df = fobj.get("/DescendantFonts")
                        enc = str(fobj.get("/Encoding"))
                        if df is not None and enc in ("/Identity-H", "/Identity-V"):
                            dead, advfn = _cid_dead_set(fobj, df[0])
                            mode = ("cid", dead, advfn)
                new_ops.append((operands, op))
                continue
            if mode is None or o not in ("Tj", "TJ"):
                new_ops.append((operands, op))
                continue
            def xf(b):
                if mode[0] == "simple":
                    return xform_simple(b, mode[1], mode[2])
                return xform_cid(b, mode[1], mode[2])
            if o == "Tj" and operands and isinstance(operands[0], pikepdf.String):
                items = xf(bytes(operands[0]))
                if items is None:
                    new_ops.append((operands, op))
                else:
                    changed = True
                    arr = pikepdf.Array(
                        [pikepdf.String(i) if isinstance(i, bytes) else i for i in items])
                    new_ops.append(([arr], pikepdf.Operator("TJ")))
                continue
            if o == "TJ" and operands and isinstance(operands[0], pikepdf.Array):
                out_items, tj_changed = [], False
                for it in operands[0]:
                    if isinstance(it, pikepdf.String):
                        items = xf(bytes(it))
                        if items is None:
                            out_items.append(it)
                        else:
                            tj_changed = True
                            out_items.extend(
                                pikepdf.String(i) if isinstance(i, bytes) else i
                                for i in items)
                    else:
                        out_items.append(it)
                if tj_changed:
                    changed = True
                    new_ops.append(([pikepdf.Array(out_items)], pikepdf.Operator("TJ")))
                else:
                    new_ops.append((operands, op))
                continue
            new_ops.append((operands, op))
        if changed:
            page["/Contents"] = pdf.make_stream(
                pikepdf.unparse_content_stream(new_ops))
            n_pages += 1
    if n_pages:
        pdf.save(outp)
    pdf.close()
    return n_pages, report


if __name__ == "__main__":
    inp, outp = sys.argv[1], sys.argv[2]
    late = len(sys.argv) > 3 and sys.argv[3] == "--dead-refs"
    n, rep = (fix_dead_refs if late else fix)(inp, outp)
    for r in rep:
        print(f"  ! {r}")
    print(f"{'wrote' if n else 'no-change'}: {n} page stream(s) repaired")
    sys.exit(0 if n else 2)
