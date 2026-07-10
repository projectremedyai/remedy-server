#!/usr/bin/env python3
"""Heal already-shipped PDFs whose body text was mispositioned by the dropped-Tm
regression (content_stream_repair._renormalize inserting bare BT). Reconstructs
each page's absolute text-line matrix by replaying the preserved relative
Td/TD/T*/Tm chain and injects an explicit Tm after any BT that lacks one.

Non-destructive: only INSERTS Tm operators; never removes/reorders/edits operands.
Source-free (works from the broken file alone). Writes to an output dir; never
overwrites the input. Flags BTs it cannot confidently place (mid-line continuation
with no leading Td) instead of guessing.

Usage:
    uv run python tools/heal_tm.py --out <dir> file1.pdf file2.pdf ...
    uv run python tools/heal_tm.py --out <dir> --list broken.txt
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

import pikepdf

logger = logging.getLogger("heal_tm")
_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mul(a, b):
    a0, a1, a2, a3, a4, a5 = a
    b0, b1, b2, b3, b4, b5 = b
    return (a0 * b0 + a1 * b2, a0 * b1 + a1 * b3,
            a2 * b0 + a3 * b2, a2 * b1 + a3 * b3,
            a4 * b0 + a5 * b2 + b4, a4 * b1 + a5 * b3 + b5)


def _f(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _build_widths(page):
    """Return {font_name: (width_fn, is_cid)} where width_fn(code)->glyph width in
    1/1000 text-space units. Simple fonts use /Widths+/FirstChar (1-byte codes);
    Type0 fonts use the descendant /W + /DW (2-byte CIDs, Identity encoding)."""
    fonts = {}
    res = page.get("/Resources")
    fdict = res.get("/Font") if res is not None else None
    if fdict is None:
        return fonts
    for name, fref in fdict.items():
        try:
            f = fref
            sub = str(f.get("/Subtype", ""))
            if sub == "/Type0":
                desc = f.get("/DescendantFonts")[0]
                dw = _f(desc.get("/DW", 1000)) or 1000.0
                w = {}
                warr = desc.get("/W")
                if warr is not None:
                    lst = list(warr); k = 0
                    while k < len(lst):
                        if k + 1 < len(lst) and isinstance(lst[k + 1], pikepdf.Array):
                            c = int(lst[k]); arr = list(lst[k + 1])
                            for m, wv in enumerate(arr):
                                w[c + m] = _f(wv)
                            k += 2
                        elif k + 2 < len(lst):
                            c1, c2, wv = int(lst[k]), int(lst[k + 1]), _f(lst[k + 2])
                            for cid in range(c1, c2 + 1):
                                w[cid] = wv
                            k += 3
                        else:
                            break
                fonts[str(name)] = ((lambda cid, w=w, dw=dw: w.get(cid, dw)), True)
            else:
                first = int(f.get("/FirstChar", 0))
                widths = f.get("/Widths")
                mw = 0.0
                fd = f.get("/FontDescriptor")
                if fd is not None and "/MissingWidth" in fd:
                    mw = _f(fd.get("/MissingWidth"))
                wl = [ _f(x) for x in widths ] if widths is not None else []
                def wf(code, first=first, wl=wl, mw=mw):
                    idx = code - first
                    return wl[idx] if 0 <= idx < len(wl) else mw
                fonts[str(name)] = (wf, False)
        except Exception:
            continue
    return fonts


def _show_advance(operands, op, width_fn, is_cid, tfs, tc, tw, th):
    """Horizontal text-space advance (delta e of Tm) for a Tj/TJ/'/" show op."""
    adv = 0.0

    def _str_adv(b):
        a = 0.0
        if is_cid:
            for k in range(0, len(b) - 1, 2):
                cid = (b[k] << 8) | b[k + 1]
                a += ((width_fn(cid) / 1000.0) * tfs + tc) * th
        else:
            for byte in b:
                wsp = tw if byte == 32 else 0.0
                a += ((width_fn(byte) / 1000.0) * tfs + tc + wsp) * th
        return a

    if op == "TJ" and operands and isinstance(operands[0], pikepdf.Array):
        for el in operands[0]:
            if isinstance(el, pikepdf.String):
                adv += _str_adv(bytes(el))
            else:
                adv += (-_f(el) / 1000.0) * tfs * th
    elif op in ("Tj", "'", '"') and operands:
        s = operands[-1]
        if isinstance(s, pikepdf.String):
            adv += _str_adv(bytes(s))
    return adv


def reinject_text_matrices(pdf: pikepdf.Pdf, page) -> tuple[int, int]:
    """Return (n_tm_injected, n_unplaceable_bt). Rewrites page.Contents if >0 injected.

    Tracks BOTH the text-line matrix (tlm) and the full text matrix (tm, advanced by
    glyph widths on show ops). When a re-opened BT lacks its own Tm, injects tlm if the
    run's first positioning op is a relative Td/TD/T* (the move then lands correctly),
    or tm if the run continues mid-line with a show op (needs the advanced position)."""
    try:
        instrs = list(pikepdf.parse_content_stream(page))
    except Exception:
        return 0, 0
    Op = pikepdf.Operator
    CSI = pikepdf.ContentStreamInstruction
    widths = _build_widths(page)
    tlm = _IDENTITY
    tm = _IDENTITY
    leading = 0.0
    cur_font = None
    tfs = 0.0; tc = 0.0; tw = 0.0; th = 1.0
    out: list = []
    injected = unplaceable = 0

    i, n = 0, len(instrs)
    while i < n:
        ins = instrs[i]
        op = str(ins.operator)
        ops = ins.operands
        if op == "BT":
            out.append(ins)
            # look ahead: first of Tm / positioning / show / ET decides injection
            first = None
            j = i + 1
            while j < n:
                oj = str(instrs[j].operator)
                if oj in ("Tm", "Td", "TD", "T*", "Tj", "TJ", "'", '"', "ET"):
                    first = oj; break
                j += 1
            if first == "Tm" or first == "ET" or first is None:
                pass  # sets its own matrix / empty
            elif first in ("Td", "TD", "T*"):
                if tlm != _IDENTITY:
                    out.append(CSI([float(x) for x in tlm], Op("Tm"))); injected += 1
                else:
                    unplaceable += 1
            else:  # mid-line continuation: restore the advanced text matrix
                if tm != _IDENTITY:
                    out.append(CSI([float(x) for x in tm], Op("Tm"))); injected += 1
                else:
                    unplaceable += 1
            i += 1
            continue
        # --- track text state / matrices ---
        if op == "Tf" and len(ops) >= 2:
            cur_font = str(ops[0]); tfs = _f(ops[1])
        elif op == "Tc" and ops:
            tc = _f(ops[0])
        elif op == "Tw" and ops:
            tw = _f(ops[0])
        elif op == "Tz" and ops:
            th = _f(ops[0]) / 100.0
        elif op == "TL" and ops:
            leading = _f(ops[0])
        elif op == "Tm" and len(ops) >= 6:
            tlm = tuple(_f(x) for x in ops[:6]); tm = tlm
        elif op == "Td" and len(ops) >= 2:
            tlm = _mul((1, 0, 0, 1, _f(ops[0]), _f(ops[1])), tlm); tm = tlm
        elif op == "TD" and len(ops) >= 2:
            leading = -_f(ops[1])
            tlm = _mul((1, 0, 0, 1, _f(ops[0]), _f(ops[1])), tlm); tm = tlm
        elif op == "T*":
            tlm = _mul((1, 0, 0, 1, 0.0, -leading), tlm); tm = tlm
        elif op in ("Tj", "TJ", "'", '"'):
            if op in ("'", '"'):
                tlm = _mul((1, 0, 0, 1, 0.0, -leading), tlm); tm = tlm
            wf, is_cid = widths.get(cur_font, ((lambda c: 500.0), False))
            adv = _show_advance(ops, op, wf, is_cid, tfs, tc, tw, th)
            tm = _mul((1, 0, 0, 1, adv, 0.0), tm)
        out.append(ins)
        i += 1

    if injected:
        page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(out))
    return injected, unplaceable


def heal_file(inp: Path, out: Path) -> dict:
    pdf = pikepdf.open(inp)
    inj = unp = 0
    for pg in pdf.pages:
        a, b = reinject_text_matrices(pdf, pg)
        inj += a; unp += b
    out.parent.mkdir(parents=True, exist_ok=True)
    if inj:
        pdf.save(out)
    else:
        pdf.close()
        import shutil
        shutil.copyfile(inp, out)
        return {"file": inp.name, "tm_injected": 0, "unplaceable_bt": unp, "changed": False}
    pdf.close()
    return {"file": inp.name, "tm_injected": inj, "unplaceable_bt": unp, "changed": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--list", type=Path, help="text file of input paths (one per line)")
    ap.add_argument("files", nargs="*", type=Path)
    args = ap.parse_args(argv)
    paths = list(args.files)
    if args.list:
        paths += [Path(l.strip()) for l in args.list.read_text().splitlines() if l.strip()]
    import json
    results = []
    for p in paths:
        if not p.exists():
            results.append({"file": p.name, "error": "missing"}); continue
        try:
            results.append(heal_file(p, args.out / p.name))
        except Exception as e:
            results.append({"file": p.name, "error": f"{type(e).__name__}: {e}"})
        print(json.dumps(results[-1]))
    (args.out / "_heal_report.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
