"""Content-stream BT/ET repair — makes remediated PDFs render in strict viewers.

The remediation engine's tag-injection corrupts page content streams: it emits
tagged text as ``BDC /Tag <</MCID n>> BT <text> EMC`` but DROPS the closing
``ET`` (and some intermediate ``BT``), leaving text objects unbalanced. Lenient
renderers (Preview/Quartz, poppler) auto-repair this, so the page looks fine;
Acrobat and Ghostscript enforce the spec and reject it ("invalid operator in
text block" / "text operator outside text block"), so the page fails to render.

This renormalizes each page's content stream so that:
  * every text-showing/-positioning/-state operator sits inside a balanced
    ``BT … ET`` (a dropped ``BT`` is re-opened),
  * no illegal operator (path/graphics/XObject) sits inside a text object
    (a dropped ``ET`` is re-inserted before it),
  * marked-content sequences stay properly nested with text objects.

ONLY ``BT``/``ET`` operators are inserted or removed. Operands, fonts, marked
content (``/MCID``), positioning, resources, and the structure tree are
untouched — text extraction and PDF/UA-1 compliance are preserved.

Idempotent: an already-balanced stream is left byte-for-byte unchanged.

This fixes *rendering*; it is independent of and complementary to
``adobe_compliance`` (which fixes the accessibility *structure tree*).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pikepdf

logger = logging.getLogger(__name__)

#: Text-showing / -positioning / -state operators. These REQUIRE a text object;
#: if one appears outside BT/ET the engine dropped the BT, so we re-open one.
_TEXT_TRIGGER = set(
    """Tj TJ ' " Td TD Tm T* Tc Tw Tz TL Tf Tr Ts""".split()
)
#: Operators additionally legal INSIDE a text object (colour + general graphics
#: state). These are legal outside too, so they neither open nor close a block.
_TEXT_OK = _TEXT_TRIGGER | set(
    """g G rg RG k K cs CS sc scn SC SCN gs w J j M d ri i d0 d1""".split()
)

_BT = pikepdf.ContentStreamInstruction([], pikepdf.Operator("BT"))
_ET = pikepdf.ContentStreamInstruction([], pikepdf.Operator("ET"))

_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mat_mul(a, b):
    """Concatenate two PDF affine matrices (6-tuples): a applied first."""
    a0, a1, a2, a3, a4, a5 = a
    b0, b1, b2, b3, b4, b5 = b
    return (a0 * b0 + a1 * b2, a0 * b1 + a1 * b3,
            a2 * b0 + a3 * b2, a2 * b1 + a3 * b3,
            a4 * b0 + a5 * b2 + b4, a4 * b1 + a5 * b3 + b5)


def _tm_instr(m):
    return pikepdf.ContentStreamInstruction(
        [float(x) for x in m], pikepdf.Operator("Tm"))


def _flt(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _build_widths(page):
    """Return {font_name: (width_fn, is_cid)} for glyph-advance tracking. Simple
    fonts use /Widths+/FirstChar (1-byte codes); Type0 use descendant /W + /DW
    (2-byte CIDs, Identity encoding). Returns {} if resources are unreadable."""
    widths: dict = {}
    try:
        res = page.get("/Resources")
        fdict = res.get("/Font") if res is not None else None
    except Exception:
        return widths
    if fdict is None:
        return widths
    for name, f in fdict.items():
        try:
            sub = str(f.get("/Subtype", ""))
            if sub == "/Type0":
                desc = f.get("/DescendantFonts")[0]
                dw = _flt(desc.get("/DW", 1000)) or 1000.0
                w: dict = {}
                warr = desc.get("/W")
                if warr is not None:
                    lst = list(warr); k = 0
                    while k < len(lst):
                        if k + 1 < len(lst) and isinstance(lst[k + 1], pikepdf.Array):
                            c = int(lst[k])
                            for m, wv in enumerate(list(lst[k + 1])):
                                w[c + m] = _flt(wv)
                            k += 2
                        elif k + 2 < len(lst):
                            c1, c2, wv = int(lst[k]), int(lst[k + 1]), _flt(lst[k + 2])
                            for cid in range(c1, c2 + 1):
                                w[cid] = wv
                            k += 3
                        else:
                            break
                widths[str(name)] = ((lambda cid, w=w, dw=dw: w.get(cid, dw)), True)
            else:
                first = int(f.get("/FirstChar", 0))
                warr = f.get("/Widths")
                wl = [_flt(x) for x in warr] if warr is not None else []
                mw = 0.0
                fd = f.get("/FontDescriptor")
                if fd is not None and "/MissingWidth" in fd:
                    mw = _flt(fd.get("/MissingWidth"))
                widths[str(name)] = (
                    (lambda code, first=first, wl=wl, mw=mw:
                        wl[code - first] if 0 <= code - first < len(wl) else mw),
                    False,
                )
        except Exception:
            continue
    return widths


def _show_advance(operands, op, width_fn, is_cid, tfs, tc, tw, th):
    """Horizontal text-space advance (delta-e of the text matrix) for a show op."""
    def _str_adv(b):
        a = 0.0
        if is_cid:
            for k in range(0, len(b) - 1, 2):
                cid = (b[k] << 8) | b[k + 1]
                a += ((width_fn(cid) / 1000.0) * tfs + tc) * th
        else:
            for byte in b:
                a += ((width_fn(byte) / 1000.0) * tfs + tc + (tw if byte == 32 else 0.0)) * th
        return a

    adv = 0.0
    if op == "TJ" and operands and isinstance(operands[0], pikepdf.Array):
        for el in operands[0]:
            if isinstance(el, pikepdf.String):
                adv += _str_adv(bytes(el))
            else:
                adv += (-_flt(el) / 1000.0) * tfs * th
    elif op in ("Tj", "'", '"') and operands and isinstance(operands[-1], pikepdf.String):
        adv += _str_adv(bytes(operands[-1]))
    return adv


@dataclass
class RepairResult:
    """Aggregate outcome of repairing one or more PDFs."""

    files: int = 0
    files_changed: int = 0
    ops_changed: int = 0          # BT/ET inserted or dropped across all pages
    errors: int = 0
    error_files: list[str] = None  # type: ignore[assignment]
    changed_files: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.error_files is None:
            self.error_files = []
        if self.changed_files is None:
            self.changed_files = []


def _renormalize(instructions, widths=None) -> tuple[list, int]:
    """Return (new_instructions, n_changes) with balanced, well-formed BT/ET.

    Pure transform over a list of pikepdf content-stream instructions, so it is
    unit-testable without a PDF. *n_changes* counts BT/ET inserted or dropped.

    *widths* is an optional {font_name: (width_fn, is_cid)} map (from
    :func:`_build_widths`). When provided, the running text matrix is advanced by
    glyph widths on show ops, so a re-opened text object whose first op is a show
    (mid-line continuation, no leading Td) is restored to the advanced position —
    not just the line origin. Without it, only relative-Td runs are placed.
    """
    instrs = list(instructions)
    n = len(instrs)
    out = []
    in_text = False
    mc_stack: list[bool] = []   # per open BDC/BMC: was it opened while in_text?
    changes = 0
    # Running text-LINE matrix (tlm, reset by relative moves) and full text matrix
    # (tm, additionally advanced by glyph widths on show ops). When we RE-OPEN a text
    # object the engine dropped, its fresh BT resets both to identity, stranding the
    # run at the origin (~1pt -> invisible). We re-emit the accumulated matrix as an
    # explicit Tm at that inserted BT: the line matrix for a leading relative Td, or
    # the advanced text matrix for a mid-line show. Only Tm is added; an already-
    # balanced stream inserts no BT, hence no Tm, and is left byte-identical.
    tlm = _IDENTITY
    tm = _IDENTITY
    leading = 0.0
    cur_font = None
    tfs = 0.0; tc = 0.0; tw = 0.0; th = 1.0

    def _wf():
        return widths.get(cur_font, ((lambda c: 500.0), False)) if widths \
            else ((lambda c: 500.0), False)

    def _track(s, instr):
        nonlocal tlm, tm, leading, cur_font, tfs, tc, tw, th
        ops = instr.operands
        if s == "Tf" and len(ops) >= 2:
            cur_font = str(ops[0]); tfs = _flt(ops[1])
        elif s == "Tc" and ops:
            tc = _flt(ops[0])
        elif s == "Tw" and ops:
            tw = _flt(ops[0])
        elif s == "Tz" and ops:
            th = _flt(ops[0]) / 100.0
        elif s == "TL" and ops:
            leading = _flt(ops[0])
        elif s == "Tm" and len(ops) >= 6:
            tlm = tuple(_flt(x) for x in ops[:6]); tm = tlm
        elif s == "Td" and len(ops) >= 2:
            tlm = _mat_mul((1.0, 0.0, 0.0, 1.0, _flt(ops[0]), _flt(ops[1])), tlm); tm = tlm
        elif s == "TD" and len(ops) >= 2:
            leading = -_flt(ops[1])
            tlm = _mat_mul((1.0, 0.0, 0.0, 1.0, _flt(ops[0]), _flt(ops[1])), tlm); tm = tlm
        elif s == "T*":
            tlm = _mat_mul((1.0, 0.0, 0.0, 1.0, 0.0, -leading), tlm); tm = tlm
        elif s in ("Tj", "TJ", "'", '"'):
            if s in ("'", '"'):
                tlm = _mat_mul((1.0, 0.0, 0.0, 1.0, 0.0, -leading), tlm); tm = tlm
            wf, is_cid = _wf()
            adv = _show_advance(ops, s, wf, is_cid, tfs, tc, tw, th)
            tm = _mat_mul((1.0, 0.0, 0.0, 1.0, adv, 0.0), tm)

    i = 0
    while i < n:
        instr = instrs[i]
        s = str(instr.operator)
        if s in ("BDC", "BMC"):
            mc_stack.append(in_text)
            out.append(instr); i += 1
            continue
        if s == "EMC":
            opened_in_text = mc_stack.pop() if mc_stack else False
            # A marked-content sequence opened OUTSIDE the text object must not
            # close inside it: end the text object first.
            if in_text and not opened_in_text:
                out.append(_ET); changes += 1; in_text = False
            out.append(instr); i += 1
            continue
        if s in ("MP", "DP"):
            out.append(instr); i += 1
            continue
        if s == "BT":
            if not in_text:
                out.append(instr); in_text = True
                tlm = _IDENTITY; tm = _IDENTITY   # a real BT resets the text matrix
            else:
                changes += 1   # redundant nested BT -> drop
            i += 1
            continue
        if s == "ET":
            if in_text:
                out.append(instr); in_text = False
            else:
                changes += 1   # orphaned ET (no open text object) -> drop
            i += 1
            continue
        if s in _TEXT_TRIGGER:
            if not in_text:
                out.append(_BT); changes += 1; in_text = True  # engine dropped the BT
                # Look ahead to the run's first positioning/show op to decide which
                # matrix to restore: the line matrix (relative Td lands correctly) or
                # the advanced text matrix (mid-line show continuation).
                first = None
                k = i
                while k < n:
                    ok = str(instrs[k].operator)
                    if ok in ("Tm", "Td", "TD", "T*", "Tj", "TJ", "'", '"', "ET"):
                        first = ok; break
                    k += 1
                if first in ("Td", "TD", "T*"):
                    if tlm != _IDENTITY:
                        out.append(_tm_instr(tlm))
                elif first in ("Tj", "TJ", "'", '"'):
                    if tm != _IDENTITY:
                        out.append(_tm_instr(tm))
                # first == "Tm" / "ET" / None -> the run sets its own matrix or is empty
            out.append(instr)
            _track(s, instr)
            i += 1
            continue
        if s in _TEXT_OK:
            out.append(instr); i += 1   # colour / graphics-state: legal either side
            continue
        # any other operator (q Q cm path-ops Do sh BI …) is illegal in a text object
        if in_text:
            out.append(_ET); changes += 1; in_text = False
        out.append(instr); i += 1
    if in_text:
        out.append(_ET); changes += 1
    return out, changes


def repair_page(pdf: pikepdf.Pdf, page) -> int:
    """Renormalize one page's content stream. Returns # of BT/ET ops changed
    (0 → already well-formed, stream left untouched)."""
    try:
        widths = _build_widths(page)
    except Exception:
        widths = None
    new_instructions, changes = _renormalize(pikepdf.parse_content_stream(page), widths)
    if changes:
        page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_instructions))
    return changes


def repair_pdf(
    pdf_path: Path | str,
    *,
    out_path: Path | str | None = None,
    dry_run: bool = False,
) -> int:
    """Repair every page of one PDF. Returns total BT/ET ops changed.

    By default rewrites in place; pass *out_path* to write elsewhere or
    *dry_run=True* to count changes without writing. Raises on unopenable PDFs.
    """
    pdf_path = Path(pdf_path)
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        total = sum(repair_page(pdf, page) for page in pdf.pages)
        if total and not dry_run:
            pdf.save(Path(out_path) if out_path else pdf_path)
    return total


def process_directory(
    directory: Path | str,
    *,
    pattern: str = "*.pdf",
    dry_run: bool = False,
) -> RepairResult:
    """Repair every PDF in *directory* (non-recursive, sorted)."""
    directory = Path(directory)
    result = RepairResult()
    for pdf_path in sorted(directory.glob(pattern)):
        try:
            changed = repair_pdf(pdf_path, dry_run=dry_run)
            result.files += 1
            result.ops_changed += changed
            if changed:
                result.files_changed += 1
                result.changed_files.append(pdf_path.name)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            result.errors += 1
            result.error_files.append(pdf_path.name)
            logger.warning("content_stream_repair failed for %s: %s", pdf_path, exc)
    return result


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="repair_content_streams",
        description="Repair unbalanced BT/ET in PDF page content streams so they "
                    "render in Acrobat/Ghostscript (not just Preview).",
    )
    parser.add_argument("path", type=Path, help="a PDF file or a directory of PDFs")
    parser.add_argument("--dry-run", action="store_true",
                        help="report changes that would be made without writing")
    parser.add_argument("--glob", default="*.pdf",
                        help="glob when PATH is a directory (default: *.pdf)")
    args = parser.parse_args(argv)

    if not args.path.exists():
        parser.error(f"path does not exist: {args.path}")

    if args.path.is_dir():
        r = process_directory(args.path, pattern=args.glob, dry_run=args.dry_run)
        print(json.dumps({
            "mode": "dry-run" if args.dry_run else "applied",
            "directory": str(args.path),
            "files": r.files,
            "files_changed": r.files_changed,
            "ops_changed": r.ops_changed,
            "errors": r.errors,
            "error_files": r.error_files,
        }, indent=2))
        return 1 if r.errors else 0

    try:
        changed = repair_pdf(args.path, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"file": str(args.path), "error": str(exc)}))
        return 1
    print(json.dumps({
        "file": str(args.path),
        "mode": "dry-run" if args.dry_run else "applied",
        "ops_changed": changed,
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
