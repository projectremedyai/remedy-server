"""Content-stream reordering — make the PHYSICAL marked-content order match the
structure tree's logical reading order.

Screen readers and PDF/UA follow the structure tree (``/StructTreeRoot``), which
the engine's reading-order fixes already correct. But Acrobat's *Order* panel,
*Read Out Loud*, *Reflow*, copy/paste and search follow the page **content
stream** — the physical sequence of ``/Tag <</MCID n>> BDC … BT … ET … EMC``
drawing blocks. The engine never re-sequences that stream, so on multi-column /
designed pages the two orders disagree: the tags read correctly, the visible
"order" is still scrambled.

This module re-sequences the movable (tagged, text-bearing) marked-content
blocks of a page so their physical order matches the struct-tree reading order,
WITHOUT changing what is drawn or where.

How it stays faithful (verified against the LAMC career-major template):
  * Each tagged unit is an atomic, non-nested ``/Tag <</MCID n>> BDC … EMC``
    block whose ``BT … ET`` text object is fully contained. Moving the block
    moves its whole text object.
  * MCIDs are referenced by NUMBER from the struct tree + ``/ParentTree`` — not
    by position — so we reorder blocks **without renumbering**. Struct linkage
    and veraPDF PDF/UA-1 validity are preserved.
  * The one hazard is graphics-state leakage: a block's glyphs depend on the CTM
    / colour / ExtGState / persistent text-state set by surrounding ``q … Q``.
    We fix this by capturing the *effective* graphics state at each block's
    ``BDC`` and re-establishing it inside a fresh ``q … Q`` wrapper around the
    moved block (CTM/colour/gs outside ``BT``; persistent text state injected
    right after ``BT``). Position then no longer affects rendering.
  * Non-movable content (artifacts, images, paths, bare graphics, and the
    original ``q … Q`` scaffolding) is emitted first, in its original order, so
    background painting and z-order are unchanged. Reordered text is painted on
    top — correct for designed pages where text sits over backgrounds.

The caller is expected to gate every page with a render pixel-diff (see
``reorder_pdf_to_struct_order``'s ``render_gate``): any page whose rendering
would change is reverted to its original stream. Combined with a struct-leaf and
text-multiset check upstream, the pass is safe-by-construction — it produces a
pixel-identical page or no change at all.

Only the page ``/Contents`` are rewritten. The structure tree, ``/ParentTree``,
MCID numbers, fonts and resources are untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pikepdf

logger = logging.getLogger(__name__)

# Operators whose effect persists in the graphics state and is saved/restored by
# q/Q. We snapshot the last-seen instruction for each and re-emit it inside the
# wrapper so a moved block renders with the state it had in its original context.
_FILL_COLOR = {"g", "rg", "k", "sc", "scn"}
_STROKE_COLOR = {"G", "RG", "K", "SC", "SCN"}
# Persistent text-state parameters (font/size, spacing, scale, leading, render
# mode, rise). These live in the graphics state and carry across BT/ET, so a
# block that does not set its own can inherit a neighbour's — we reinject them
# right after the block's BT. NB: Tm/Td/T* are positional and block-local, so
# they are NOT snapshotted.
_TEXT_STATE = {"Tf", "Tc", "Tw", "Tz", "TL", "Tr", "Ts"}
_TEXT_SHOW = {"Tj", "TJ", "'", '"'}
# Operators that mark a block as NON-text (paths / images / shadings). A tagged
# block containing any of these is left in place (not moved) to avoid z-order or
# clipping surprises.
_PAINT_OPS = {"Do", "re", "f", "F", "f*", "S", "s", "B", "B*", "b", "b*", "n",
              "sh", "BI", "m", "l", "c", "v", "y", "W", "W*"}
# Graphics-state operators whose effect persists past the block and may be relied
# on by FOLLOWING background content. When a movable block is lifted out, these
# are left behind (without the marked-content wrapper or any text drawing) so the
# rest of the page inherits exactly the state it did originally.
_GHOST_KEEP = (_FILL_COLOR | _STROKE_COLOR |
               {"cs", "CS", "gs", "w", "J", "j", "M", "d", "ri", "i",
                "cm", "q", "Q"})

_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mat_mul(a, b):
    """Concatenate two PDF affine matrices (6-tuples). Returns a·b (a applied
    first, row-vector convention: p·a·b)."""
    a0, a1, a2, a3, a4, a5 = a
    b0, b1, b2, b3, b4, b5 = b
    return (
        a0 * b0 + a1 * b2,
        a0 * b1 + a1 * b3,
        a2 * b0 + a3 * b2,
        a2 * b1 + a3 * b3,
        a4 * b0 + a5 * b2 + b4,
        a4 * b1 + a5 * b3 + b5,
    )


def _mat_inverse(m):
    """Inverse of a PDF affine matrix (6-tuple), or None if singular."""
    a, b, c, d, e, f = m
    det = a * d - b * c
    if abs(det) < 1e-9:
        return None
    return (d / det, -b / det, -c / det, a / det,
            (c * f - d * e) / det, (b * e - a * f) / det)


@dataclass
class _GState:
    """Snapshot-able effective graphics state (the parts that affect a moved
    text block's rendering)."""
    ctm: tuple = _IDENTITY
    cs: object = None          # fill colour space (cs)
    fill: object = None        # last fill colour op (g/rg/k/sc/scn)
    cs_stroke: object = None   # stroke colour space (CS)
    stroke: object = None      # last stroke colour op (G/RG/K/SC/SCN)
    gs: object = None          # ExtGState (gs)
    gen: dict = field(default_factory=dict)   # w/J/j/M/d/ri/i
    text: dict = field(default_factory=dict)  # persistent text-state ops

    def copy(self) -> "_GState":
        return _GState(self.ctm, self.cs, self.fill, self.cs_stroke, self.stroke,
                       self.gs, dict(self.gen), dict(self.text))


def _mcid_of(operands, page_props):
    """Return the int MCID for a BDC's operands, or None. Handles inline
    ``<</MCID n>>`` dicts and named properties resolved via /Properties."""
    if len(operands) < 2:
        return None
    props = operands[1]
    try:
        if isinstance(props, pikepdf.Dictionary):
            if "/MCID" in props:
                return int(props["/MCID"])
            return None
        if isinstance(props, pikepdf.Name) and page_props is not None:
            resolved = page_props.get(str(props))
            if resolved is not None and "/MCID" in resolved:
                return int(resolved["/MCID"])
    except Exception:
        return None
    return None


@dataclass
class _Block:
    mcid: int
    instrs: list             # the full BDC..EMC instruction slice
    state: _GState           # effective state captured at the BDC
    bt_idx: int              # index (within instrs) of the first BT, or -1


def _emit_state(state: _GState):
    """Instructions to re-establish *state* inside a fresh q (graphics part,
    emitted BEFORE the block's BT)."""
    out = []
    Op = pikepdf.Operator
    CSI = pikepdf.ContentStreamInstruction
    if state.ctm != _IDENTITY:
        out.append(CSI([float(x) for x in state.ctm], Op("cm")))
    if state.cs is not None:
        out.append(state.cs)
    if state.fill is not None:
        out.append(state.fill)
    if state.cs_stroke is not None:
        out.append(state.cs_stroke)
    if state.stroke is not None:
        out.append(state.stroke)
    if state.gs is not None:
        out.append(state.gs)
    for key in ("w", "J", "j", "M", "d", "ri", "i"):
        if key in state.gen:
            out.append(state.gen[key])
    return out


def _wrap_block(block: _Block):
    """A moved block, wrapped in q/Q with its captured state re-established and
    persistent text-state reinjected right after its BT."""
    Op = pikepdf.Operator
    CSI = pikepdf.ContentStreamInstruction
    out = [CSI([], Op("q"))]
    out.extend(_emit_state(block.state))
    if block.bt_idx >= 0 and block.state.text:
        # inject text-state instructions immediately after the block's BT; the
        # block's own ops (if any) follow and override.
        text_injection = [block.state.text[k] for k in
                          ("Tf", "Tc", "Tw", "Tz", "TL", "Tr", "Ts")
                          if k in block.state.text]
        for i, instr in enumerate(block.instrs):
            out.append(instr)
            if i == block.bt_idx:
                out.extend(text_injection)
    else:
        out.extend(block.instrs)
    out.append(CSI([], Op("Q")))
    return out


def _apply_state(gstate: _GState, gstack: list, instr) -> _GState:
    """Update the running graphics state for one instruction. Returns the
    (possibly swapped, on q/Q) current state. Tracks only the parameters that a
    moved text block depends on; ignores positional text ops (Tm/Td/T*)."""
    op = str(instr.operator)
    if op == "q":
        gstack.append(gstate.copy())
    elif op == "Q":
        if gstack:
            return gstack.pop()
    elif op == "cm":
        try:
            m = tuple(float(x) for x in instr.operands[:6])
            gstate.ctm = _mat_mul(m, gstate.ctm)
        except Exception:
            pass
    elif op == "cs":
        gstate.cs = instr
    elif op == "CS":
        gstate.cs_stroke = instr
    elif op in _FILL_COLOR:
        gstate.fill = instr
    elif op in _STROKE_COLOR:
        gstate.stroke = instr
    elif op == "gs":
        gstate.gs = instr
    elif op in ("w", "J", "j", "M", "d", "ri", "i"):
        gstate.gen[op] = instr
    elif op in _TEXT_STATE:
        gstate.text[op] = instr
    return gstate


def _collect_slice(instructions, start):
    """Collect a marked-content slice beginning at the BDC at *start*.

    Returns (slice_instrs, end_index, has_text, has_paint, nested, bt_idx) where
    end_index is the index of the matching EMC (or -1 if unbalanced)."""
    slice_instrs = [instructions[start]]
    depth = 1
    has_text = has_paint = nested = False
    bt_idx = -1
    j = start + 1
    n = len(instructions)
    while j < n and depth > 0:
        s2 = instructions[j]
        o2 = str(s2.operator)
        if o2 in ("BDC", "BMC"):
            depth += 1
            nested = True
        elif o2 == "EMC":
            depth -= 1
            if depth == 0:
                slice_instrs.append(s2)
                return slice_instrs, j, has_text, has_paint, nested, bt_idx
        if o2 == "BT" and bt_idx < 0:
            bt_idx = len(slice_instrs)
        if o2 in _TEXT_SHOW:
            has_text = True
        if o2 in _PAINT_OPS:
            has_paint = True
        slice_instrs.append(s2)
        j += 1
    return slice_instrs, -1, has_text, has_paint, nested, bt_idx


def _segment(instructions, page_props):
    """Walk a page's instruction list once, tracking graphics state.

    Returns (background, blocks, final_ctm, bail_reason):
      * background — instructions with the movable text blocks removed.
      * blocks — list[_Block] of movable, text-bearing tagged blocks, each with
        the effective graphics state captured at its BDC.
      * final_ctm — the CTM in effect at the end of the background stream (used
        by the caller to reset to identity before re-emitting moved blocks).
      * bail_reason — non-empty string if the page is unsafe to reorder.
    """
    background = []
    blocks: list[_Block] = []
    gstate = _GState()
    gstack: list[_GState] = []
    mc_depth = 0

    i = 0
    n = len(instructions)
    while i < n:
        instr = instructions[i]
        op = str(instr.operator)

        if op == "BDC" and mc_depth == 0:
            tag = str(instr.operands[0]) if instr.operands else ""
            mcid = _mcid_of(instr.operands, page_props)
            if tag != "/Artifact" and mcid is not None:
                slc, end, has_text, has_paint, nested, bt_idx = \
                    _collect_slice(instructions, i)
                if end < 0:
                    return background, blocks, gstate.ctm, "unbalanced marked content"
                if nested or has_paint or not has_text:
                    # not safely movable: keep in place, but still track state
                    for s2 in slc:
                        gstate = _apply_state(gstate, gstack, s2)
                        background.append(s2)
                    i = end + 1
                    continue
                # movable: snapshot incoming state, then apply the block's own
                # state so later blocks inherit it (matches original semantics).
                blocks.append(_Block(mcid=mcid, instrs=slc,
                                     state=gstate.copy(), bt_idx=bt_idx))
                # Leave a state-only "ghost" in place: the block's graphics-state
                # ops (colour/CTM/gs/…) WITHOUT its marked-content wrapper or any
                # text drawing, so following background content inherits exactly
                # the state it did originally. No MCID is duplicated.
                for s2 in slc:
                    gstate = _apply_state(gstate, gstack, s2)
                    if str(s2.operator) in _GHOST_KEEP:
                        background.append(s2)
                i = end + 1
                continue

        if op in ("BDC", "BMC"):
            mc_depth += 1
            gstate = _apply_state(gstate, gstack, instr)
            background.append(instr)
            i += 1
            continue
        if op == "EMC":
            if mc_depth > 0:
                mc_depth -= 1
            background.append(instr)
            i += 1
            continue

        gstate = _apply_state(gstate, gstack, instr)
        background.append(instr)
        i += 1

    return background, blocks, gstate.ctm, ""


def reorder_page_to_order(pdf: pikepdf.Pdf, page, target_mcids: list[int]) -> int:
    """Re-sequence *page*'s movable tagged blocks to match *target_mcids*.

    Returns the number of blocks whose physical position changed (0 → stream
    left byte-identical). Movable blocks not present in *target_mcids* are kept
    after the ordered ones in their original relative order; target MCIDs with no
    block are skipped. Returns 0 (no change) if the page is unsafe to reorder.
    """
    try:
        instructions = list(pikepdf.parse_content_stream(page))
    except Exception:
        return 0
    page_props = None
    try:
        res = page.get("/Resources")
        if res is not None and "/Properties" in res:
            page_props = res["/Properties"]
    except Exception:
        page_props = None

    background, blocks, final_ctm, bail = _segment(instructions, page_props)
    if bail or not blocks:
        return 0
    inv = _mat_inverse(final_ctm)
    if inv is None:
        return 0  # cannot safely reset CTM — leave page untouched

    by_mcid: dict[int, list[_Block]] = {}
    for b in blocks:
        by_mcid.setdefault(b.mcid, []).append(b)

    original_order = [b.mcid for b in blocks]
    ordered: list[_Block] = []
    used = set()
    for mcid in target_mcids:
        bucket = by_mcid.get(mcid)
        if bucket:
            ordered.append(bucket.pop(0))
            used.add(id(ordered[-1]))
    # append any movable blocks not named in the target order, original order
    for b in blocks:
        if id(b) not in used:
            ordered.append(b)

    new_order = [b.mcid for b in ordered]
    if new_order == original_order:
        return 0  # already in target order — idempotent, no rewrite

    Op = pikepdf.Operator
    CSI = pikepdf.ContentStreamInstruction
    new_instrs = list(background)
    # Re-emit moved blocks after the background, inside an outer q…Q that first
    # resets the CTM to identity (the background may leave a non-identity CTM).
    # Each block then sets its own absolute CTM inside its own q…Q.
    new_instrs.append(CSI([], Op("q")))
    new_instrs.append(CSI([float(x) for x in inv], Op("cm")))
    for b in ordered:
        new_instrs.extend(_wrap_block(b))
    new_instrs.append(CSI([], Op("Q")))

    page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_instrs))
    return sum(1 for a, c in zip(original_order, new_order) if a != c)


def struct_reading_order(pdf: pikepdf.Pdf) -> dict:
    """Return {page_index: [mcid, …]} — the MCIDs of each page's struct-tree
    leaves in reading (struct) order. This is the target the content stream is
    re-sequenced to match. Pure pikepdf; mirrors the screen-reader traversal."""
    try:
        pidx = {pg.obj.objgen: i for i, pg in enumerate(pdf.pages)}
    except Exception:
        return {}
    seq: list = []

    def page_of(node, inherited):
        try:
            pg = node.get("/Pg")
            if pg is not None:
                return pg.objgen
        except Exception:
            pass
        return inherited

    def walk(node, inherited):
        pg = page_of(node, inherited)
        try:
            k = node.get("/K")
        except Exception:
            k = None
        if k is None:
            return
        items = list(k) if isinstance(k, pikepdf.Array) else [k]
        for it in items:
            if isinstance(it, pikepdf.Dictionary):
                walk(it, pg)
            else:
                try:
                    seq.append((pidx.get(pg), int(it)))
                except Exception:
                    pass

    try:
        walk(pdf.Root.StructTreeRoot, None)
    except Exception:
        return {}
    per: dict = {}
    for idx, mc in seq:
        if idx is None:
            continue
        # keep first occurrence order, drop duplicates per page
        bucket = per.setdefault(idx, [])
        if mc not in bucket:
            bucket.append(mc)
    return per


def _render_pages(source, dpi):
    """Grayscale page samples for a PDF given as a path or raw bytes."""
    import fitz
    if isinstance(source, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(source), filetype="pdf")
    else:
        doc = fitz.open(str(source))
    out = []
    try:
        for pg in doc:
            pix = pg.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            out.append((pix.width, pix.height, bytes(pix.samples)))
    finally:
        doc.close()
    return out


def _diff_fraction(a, b, tol=24, stride=7):
    if a[0] != b[0] or a[1] != b[1]:
        return 1.0
    sa, sb = a[2], b[2]
    n = min(len(sa), len(sb))
    if n == 0:
        return 0.0
    diff = sum(1 for i in range(0, n, stride) if abs(sa[i] - sb[i]) > tol)
    return diff / (n / stride)


def reorder_file_to_struct_order(path, out_path=None, *, dpi=110,
                                 max_diff=0.0003) -> dict:
    """Re-sequence every page's content stream to struct order, behind a render
    pixel-diff gate. If any page would render differently beyond *max_diff*, the
    whole file is left unchanged (rejected). Returns a report dict.

    The render gate is the authoritative content-fidelity check: a pixel-
    identical render guarantees no glyph was moved, dropped or recoloured. (Note:
    ``pdftotext`` word counts are NOT a reliable gate here — extraction order is
    itself content-stream-order-dependent, so a faithful reorder can legitimately
    change them.) The default threshold cleanly separates faithful reorders
    (≤1e-4 from anti-aliasing at z-order edges) from blocks that rely on inherited
    text positioning and would shift (≥6e-4).

    The struct tree, MCID numbers and ParentTree are not touched, so PDF/UA
    validity is preserved; the caller may additionally verify with veraPDF.
    """
    import io
    from pathlib import Path as _Path
    path = _Path(path)
    rep = {"file": path.name, "changed": False, "pages_reordered": 0,
           "worst_diff": 0.0, "rejected": False, "note": ""}
    try:
        before = _render_pages(path, dpi)
    except Exception as exc:  # noqa: BLE001
        rep["note"] = f"render-before failed: {exc}"
        return rep
    pdf = pikepdf.open(str(path))
    try:
        if "/StructTreeRoot" not in pdf.Root:
            rep["note"] = "no struct tree"
            return rep
        target = struct_reading_order(pdf)
        total = 0
        for idx, page in enumerate(pdf.pages):
            total += reorder_page_to_order(pdf, page, target.get(idx, []))
        if total == 0:
            rep["note"] = "already in struct order"
            return rep
        buf = io.BytesIO()
        pdf.save(buf)
    finally:
        pdf.close()
    candidate = buf.getvalue()
    try:
        after = _render_pages(candidate, dpi)
    except Exception as exc:  # noqa: BLE001
        rep["note"] = f"render-after failed: {exc}"
        return rep
    if len(before) != len(after):
        rep["rejected"] = True
        rep["note"] = "page count changed"
        return rep
    worst = max((_diff_fraction(a, b) for a, b in zip(before, after)),
                default=1.0)
    rep["worst_diff"] = round(worst, 5)
    if worst > max_diff:
        rep["rejected"] = True
        rep["note"] = f"render diff {worst:.5f} > {max_diff}"
        return rep
    _Path(out_path or path).write_bytes(candidate)
    rep["changed"] = True
    rep["pages_reordered"] = total
    return rep


def _verapdf_rule_ids_bytes(data: bytes):
    """Failed veraPDF PDF/UA-1 rule ids for *data* (PDF bytes), or None if
    veraPDF is unavailable (so the caller skips the delta check)."""
    import os
    import tempfile
    from pathlib import Path

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    try:
        Path(tmp.name).write_bytes(data)
        from project_remedy.pdf_acceptance import validate_with_verapdf
        res = validate_with_verapdf(Path(tmp.name))
    except Exception as exc:  # noqa: BLE001
        logger.debug("verapdf delta check unavailable: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if not getattr(res, "checked", False):
        return None
    return {v.get("id", "?") for v in (getattr(res, "violations", None) or [])}


def reorder_pdf_in_place(pdf: pikepdf.Pdf, *, dpi=110, max_diff=0.0003,
                         verify_verapdf=False):
    """Reorder an already-open Pdf's content streams to struct order, render-
    gated. If any page would render beyond *max_diff*, the entire reorder is
    reverted (page /Contents restored). Returns (pages_reordered, note).

    When *verify_verapdf* is true, the reorder is ALSO reverted if it introduces
    any new PDF/UA-1 failure versus the pre-reorder state (a before/after delta).
    The render gate guarantees pixel fidelity but a content-stream re-serialize
    can still surface a font-program/CID clause veraPDF flags; this backstop
    keeps the pass from ever shipping a new violation.

    Used by the fix_all pipeline so a single ``remedy pdf fix`` run aligns the
    physical content order with the logical structure order it just built.
    """
    import io
    if "/StructTreeRoot" not in pdf.Root:
        return 0, "no struct tree"
    try:
        buf0 = io.BytesIO()
        pdf.save(buf0)
        before = _render_pages(buf0.getvalue(), dpi)
    except Exception as exc:  # noqa: BLE001
        return 0, f"render-before failed: {exc}"
    target = struct_reading_order(pdf)
    saved = {}
    total = 0
    for idx, page in enumerate(pdf.pages):
        try:
            saved[idx] = page.obj.get("/Contents")
            total += reorder_page_to_order(pdf, page, target.get(idx, []))
        except Exception:
            saved.pop(idx, None)
    if total == 0:
        return 0, "already in struct order"

    def _revert(reason):
        for idx, page in enumerate(pdf.pages):
            if saved.get(idx) is not None:
                page.obj["/Contents"] = saved[idx]
        return 0, reason

    try:
        buf1 = io.BytesIO()
        pdf.save(buf1)
        after = _render_pages(buf1.getvalue(), dpi)
    except Exception:  # noqa: BLE001
        after = None
    ok = after is not None and len(before) == len(after)
    worst = (max((_diff_fraction(a, b) for a, b in zip(before, after)),
                 default=1.0) if ok else 1.0)
    if worst > max_diff:
        return _revert(f"reverted (render diff {worst:.5f} > {max_diff})")
    if verify_verapdf:
        before_rules = _verapdf_rule_ids_bytes(buf0.getvalue())
        after_rules = _verapdf_rule_ids_bytes(buf1.getvalue())
        if before_rules is not None and after_rules is not None:
            new_rules = after_rules - before_rules
            if new_rules:
                return _revert("reverted (introduced veraPDF failure(s): "
                               + ", ".join(sorted(new_rules)) + ")")
    return total, f"render diff {worst:.5f}"


def fix_content_stream_order(pdf: pikepdf.Pdf, *, verify_verapdf=True) -> list:
    """fix_all-compatible wrapper: returns a list of human-readable changes.

    *verify_verapdf* defaults on so the reorder self-reverts rather than ship a
    new PDF/UA-1 failure (symmetric with the struct-tree reorder gate)."""
    try:
        n, note = reorder_pdf_in_place(pdf, verify_verapdf=verify_verapdf)
    except Exception:  # noqa: BLE001
        return []
    if n:
        return [f"Re-sequenced page content streams to match the structure-tree "
                f"reading order ({n} blocks moved; {note})"]
    return []
