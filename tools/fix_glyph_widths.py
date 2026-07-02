#!/usr/bin/env python3
"""Deterministic PDF/UA fix for veraPDF rule 7.21.5-1 (glyph-width consistency).

Rewrites the width arrays in the font DICTIONARY (/W for CIDFonts, /Widths for
simple fonts) so they match the advance widths in the embedded font PROGRAM.
Does NOT touch content streams, glyph codes, the program, or rendering.

Usage: fix_glyph_widths.py in.pdf out.pdf
Exit 0 if it wrote a file, 2 if nothing to do.
"""
import sys, io, os
import pikepdf
from fontTools.ttLib import TTFont

def _load_program(fd):
    """Return (TTFont, kind) for the embedded program in a FontDescriptor, or (None,None)."""
    for key, kind in (("/FontFile2","tt"), ("/FontFile3","cff")):
        if key in fd:
            try:
                data = bytes(fd[key].read_bytes())
                return TTFont(io.BytesIO(data)), kind
            except Exception:
                return None, None
    return None, None

def _adv_by_gid(tt):
    """GID -> advance width in *glyph-space* units scaled to /1000 em."""
    upm = tt["head"].unitsPerEm if "head" in tt else 1000
    order = tt.getGlyphOrder()
    gs = tt.getGlyphSet()
    out = {}
    for gid, name in enumerate(order):
        try:
            w = gs[name].width
        except Exception:
            continue
        if w is None:
            continue
        out[gid] = int(round(w * 1000.0 / upm))
    return out

def _cid_to_gid(cidfont, nglyphs):
    m = cidfont.get("/CIDToGIDMap")
    if m is None or m == pikepdf.Name("/Identity"):
        return lambda cid: cid                       # Identity
    b = bytes(m.read_bytes())
    def f(cid):
        i = cid * 2
        return (b[i] << 8) | b[i+1] if i+1 < len(b) else 0
    return f

def _build_W(adv_by_cid):
    """Compact PDF /W array: runs of consecutive CIDs -> [c [w w w ...]]."""
    if not adv_by_cid:
        return None
    cids = sorted(adv_by_cid)
    W = []
    i = 0
    while i < len(cids):
        j = i
        run = [adv_by_cid[cids[i]]]
        while j+1 < len(cids) and cids[j+1] == cids[j]+1:
            j += 1; run.append(adv_by_cid[cids[j]])
        W.append(cids[i]); W.append(run)
        i = j+1
    return W

def fix(inp, outp):
    pdf = pikepdf.open(inp)
    changed = 0
    for obj in pdf.objects:
        try:
            if not isinstance(obj, pikepdf.Dictionary) or obj.get("/Type") != pikepdf.Name("/Font"):
                continue
        except Exception:
            continue
        sub = obj.get("/Subtype")
        # Tesseract's GlyphLessFont: /W carries per-OCR-box advances that are
        # inconsistent with the 1-glyph program BY DESIGN (veraPDF does not
        # flag it). "Fixing" them to the program advance collapses the
        # invisible text layer's geometry -> extracted words merge. Skip.
        if "glyphless" in str(obj.get("/BaseFont", "")).lower():
            continue
        # ---- Type0 -> descendant CIDFontType2/0 with /W ----
        if sub == pikepdf.Name("/Type0"):
            for cidfont in (obj.get("/DescendantFonts") or []):
                fd = cidfont.get("/FontDescriptor")
                if fd is None:
                    continue
                tt, kind = _load_program(fd)
                if tt is None:
                    continue
                adv_gid = _adv_by_gid(tt)
                ng = tt["maxp"].numGlyphs if "maxp" in tt else max(adv_gid, default=0)+1
                c2g = _cid_to_gid(cidfont, ng)
                # widths for every CID whose GID has an advance
                adv_cid = {}
                for cid in range(ng):
                    gid = c2g(cid)
                    if gid in adv_gid:
                        adv_cid[cid] = adv_gid[gid]
                newW = _build_W(adv_cid)
                if newW is None:
                    continue
                dw = adv_gid.get(0, 1000)
                arr = pikepdf.Array()
                for el in newW:
                    arr.append(pikepdf.Array(el) if isinstance(el, list) else el)
                # only rewrite if different
                if str(cidfont.get("/W")) != str(arr) or cidfont.get("/DW") != dw:
                    cidfont["/W"] = arr
                    cidfont["/DW"] = dw
                    changed += 1
        # ---- simple TrueType/Type1 with /Widths ----
        elif sub in (pikepdf.Name("/TrueType"), pikepdf.Name("/Type1")):
            fd = obj.get("/FontDescriptor")
            if fd is None or "/FirstChar" not in obj or "/Widths" not in obj:
                continue
            tt, kind = _load_program(fd)
            if tt is None:
                continue
            upm = tt["head"].unitsPerEm if "head" in tt else 1000
            gs = tt.getGlyphSet(); order = set(tt.getGlyphOrder())
            # index cmap subtables by (platformID, platEncID)
            subt = {}
            if "cmap" in tt:
                for t in tt["cmap"].tables:
                    subt.setdefault((t.platformID, t.platEncID), t.cmap)
            from fontTools import agl
            # base-encoding codec + /Encoding Differences overrides -> code->glyphname
            enc = obj.get("/Encoding")
            base = enc if isinstance(enc, pikepdf.Name) else (
                   enc.get("/BaseEncoding") if isinstance(enc, pikepdf.Dictionary) else None)
            codec = {"/WinAnsiEncoding":"cp1252","/MacRomanEncoding":"mac_roman"}.get(str(base) if base else "", "cp1252")
            diffs = {}
            if isinstance(enc, pikepdf.Dictionary) and "/Differences" in enc:
                cur = 0
                for it in enc["/Differences"]:
                    if isinstance(it, int): cur = it
                    else: diffs[cur] = str(it).lstrip("/"); cur += 1

            def glyph_for(code):
                # 1) named-encoding -> Unicode -> (3,1)/(0,3) cmap  [correct for text fonts]
                uni = None
                if code in diffs:
                    uni = agl.toUnicode(diffs[code]) or None
                    uni = uni[0] if uni else None
                else:
                    try: uni = bytes([code]).decode(codec)
                    except Exception: uni = None
                if uni:
                    for key in ((3,1),(0,3),(0,4),(0,6)):
                        c = subt.get(key)
                        if c and ord(uni) in c: return c[ord(uni)]
                # 2) Mac (1,0) by raw code  [symbol dingbat fonts select here]
                c = subt.get((1,0))
                if c and code in c: return c[code]
                # 3) Microsoft symbol (3,0) by 0xF000|code then code
                c = subt.get((3,0))
                if c:
                    for probe in (0xF000 | code, code):
                        if probe in c: return c[probe]
                # 4) last resort: named glyph directly present in program
                if code in diffs and diffs[code] in order: return diffs[code]
                return None

            first = int(obj["/FirstChar"]); last = int(obj["/LastChar"])
            old = [int(x) for x in obj["/Widths"]]
            widths = list(old); dirty = False
            for code in range(first, last+1):
                gname = glyph_for(code)
                if gname is None: continue          # unresolved -> keep original (safe)
                try:
                    w = gs[gname].width
                except Exception:
                    continue
                if w is None: continue
                nv = int(round(w * 1000.0 / upm))
                idx = code - first
                if 0 <= idx < len(widths) and widths[idx] != nv:
                    widths[idx] = nv; dirty = True
            if dirty:
                obj["/Widths"] = pikepdf.Array(widths)
                changed += 1
    if changed:
        pdf.save(outp)
    pdf.close()
    return changed

if __name__ == "__main__":
    inp, outp = sys.argv[1], sys.argv[2]
    n = fix(inp, outp)
    print(f"{'wrote' if n else 'no-change'}: {n} font width array(s) rewritten -> {outp if n else '(none)'}")
    sys.exit(0 if n else 2)
