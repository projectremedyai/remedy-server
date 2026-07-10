#!/usr/bin/env python3
"""fix_cidset.py — clear veraPDF PDF/UA-1 7.21.4.2 (CIDSet completeness).

Clause 7.21.4.2-2: *if* an embedded CID font's FontDescriptor contains a CIDSet
stream, that stream must identify **all** CIDs present in the font program
(referenced or not). 7.21.4.2-1 is the companion presence/consistency check.

On the LAMC large-file corpus these fail because producers wrote a *truncated*
CIDSet (e.g. a 1-byte stream — CIDs 0..7 — for a 4685-glyph subset program).

CIDSet is OPTIONAL in PDF/UA-1 (ISO 14289-1) and carries no accessibility or
rendering information — it is a subset-completeness hint only. An INCORRECT
CIDSet is strictly worse than none. So the default, provably-correct, uniform
fix is to **remove** the wrong optional stream (the conditional rule then passes
vacuously) — zero rendering/extraction risk, identical for TrueType/CFF/CID-keyed.

`--rebuild` instead reconstructs a correct CIDSet from the embedded program for
the common Identity / CID==GID case (bitmap of every CID present, verified
against the program before writing); it DECLINES to rebuild — and falls back to
removal — for any font whose present-CID set it cannot determine with certainty
(never emits a guessed bitmap).

    uv run python tools/fix_cidset.py in.pdf out.pdf [--rebuild]

Gate the output on a veraPDF before/after delta as usual (7.21.4.2 cleared AND
no new failed clause). Rendering and text extraction are untouched either way.
"""
import io
import sys
import pikepdf
from pikepdf import Name


# ---- program CID enumeration -------------------------------------------------

def _numglyphs_truetype(data):
    from fontTools.ttLib import TTFont
    tt = TTFont(io.BytesIO(data), fontNumber=0, lazy=True)
    return tt["maxp"].numGlyphs


def _cff_present_cids(data):
    """Return the set of CIDs present in a bare CFF (/FontFile3) program.

    CID-keyed CFF: charset names are 'cidNNNNN' -> those integers are the CIDs.
    Plain CFF: CID == GID, so 0..nGlyphs-1.
    Returns None if the CFF cannot be parsed with certainty.
    """
    from fontTools.cffLib import CFFFontSet
    try:
        cff = CFFFontSet()
        cff.decompile(io.BytesIO(data), None)
        font = cff[cff.fontNames[0]]
        charset = list(font.charset)
    except Exception:
        return None
    if not charset:
        return None
    if all(n.startswith("cid") for n in charset):
        try:
            return {int(n[3:]) for n in charset}
        except ValueError:
            return None
    # non-CID-keyed subset: CID == GID
    return set(range(len(charset)))


def _present_cids(cidfont):
    """Sorted list of CIDs present in the embedded program, or None if unknown."""
    fd = cidfont.get("/FontDescriptor")
    if fd is None:
        return None
    ng = None
    cff_cids = None
    if "/FontFile2" in fd:                       # CIDFontType2, TrueType
        try:
            ng = _numglyphs_truetype(bytes(fd["/FontFile2"].read_bytes()))
        except Exception:
            return None
    elif "/FontFile3" in fd:                     # CIDFontType0, CFF (bare)
        data = bytes(fd["/FontFile3"].read_bytes())
        cff_cids = _cff_present_cids(data)
        if cff_cids is None:
            try:                                 # OpenType-CFF sfnt wrapper?
                ng = _numglyphs_truetype(data)
            except Exception:
                return None
    else:
        return None

    c2g = cidfont.get("/CIDToGIDMap")
    identity = c2g is None or str(c2g) == "/Identity"

    if cff_cids is not None and identity:
        return sorted(cff_cids)
    if ng is None:
        return None
    if identity:
        return list(range(ng))                   # CID == GID, all present
    # explicit CIDToGIDMap stream: 2 bytes per CID -> GID
    try:
        m = bytes(c2g.read_bytes())
    except Exception:
        return None
    cids = {0}                                    # .notdef always present
    for c in range(len(m) // 2):
        gid = (m[2 * c] << 8) | m[2 * c + 1]
        if 0 < gid < ng:
            cids.add(c)
    return sorted(cids)


def _bitmap(cids):
    """Big-endian CIDSet bitmap: bit (0x80>>c%8) of byte c//8 set iff CID c present."""
    if not cids:
        return b"\x00"
    ba = bytearray((max(cids) // 8) + 1)
    for c in cids:
        ba[c // 8] |= 0x80 >> (c % 8)
    return bytes(ba)


def _verify_bitmap(bm, cids):
    """Assert the bitmap identifies exactly `cids` (self-check before writing)."""
    got = {b * 8 + i for b, byte in enumerate(bm) for i in range(8) if byte & (0x80 >> i)}
    return got == set(cids)


# ---- traversal ---------------------------------------------------------------

def _iter_cidfonts(pdf):
    seen = set()
    for obj in pdf.objects:
        if not isinstance(obj, pikepdf.Dictionary):
            continue
        try:
            if obj.get("/Type") != Name("/Font") or obj.get("/Subtype") != Name("/Type0"):
                continue
        except Exception:
            continue
        df = obj.get("/DescendantFonts")
        if not df:
            continue
        cid = df[0]
        fd = cid.get("/FontDescriptor")
        if fd is None or "/CIDSet" not in fd:     # optional & absent -> rule not triggered
            continue
        try:
            key = fd.objgen                        # stable identity for indirect objects
        except Exception:
            key = id(fd)
        if key in seen:
            continue
        seen.add(key)
        yield cid, fd


def fix(inp, out, rebuild=False):
    pdf = pikepdf.open(str(inp))
    rebuilt = removed = 0
    detail = []
    for cid, fd in _iter_cidfonts(pdf):
        name = str(cid.get("/BaseFont"))
        cids = _present_cids(cid) if rebuild else None
        if rebuild and cids is not None:
            bm = _bitmap(cids)
            if _verify_bitmap(bm, cids):
                fd["/CIDSet"] = pdf.make_stream(bm)
                rebuilt += 1
                detail.append(f"rebuilt {name}: {len(cids)} CIDs, {len(bm)} bytes")
                continue
        # default / fallback: drop the incorrect optional stream
        del fd["/CIDSet"]
        removed += 1
        detail.append(f"removed {name}")
    pdf.save(str(out))
    return {"cidset_rebuilt": rebuilt, "cidset_removed": removed, "detail": detail}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rebuild = "--rebuild" in sys.argv
    if len(args) != 2:
        sys.exit("usage: fix_cidset.py in.pdf out.pdf [--rebuild]")
    rep = fix(args[0], args[1], rebuild=rebuild)
    print(rep)
