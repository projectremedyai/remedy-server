"""fix_all must run the OCG config /Name fixer (PDF/UA-1 7.10-1).

``fix_optional_content_config_names`` existed but was never registered in
fix_all's FIXES table — so files whose only veraPDF failure was 7.10-1 (an
optional-content /D config without /Name) could never converge: the corpus
runner's refix replayed fix_all, which silently skipped the one fix needed.
"""
from __future__ import annotations

import pikepdf
from pikepdf import Dictionary, Name, Array

from project_remedy.pdf_fixer import fix_all


def _pdf_with_unnamed_ocg(path):
    """OCG actually USED by page content (like the real corpus files) so the
    unused-OCG cleanup can't just strip /OCProperties away."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    ocg = pdf.make_indirect(Dictionary(Type=Name("/OCG"), Name=pikepdf.String("Layer 1")))
    page = pdf.pages[0].obj
    page.Contents = pdf.make_stream(
        b"/OC /MC0 BDC\nBT /F1 12 Tf 72 700 Td (layer text) Tj ET\nEMC\n")
    page.Resources = Dictionary(Properties=Dictionary(MC0=ocg))
    pdf.Root.OCProperties = Dictionary(
        OCGs=Array([ocg]),
        # default config dict WITHOUT /Name -> veraPDF 7.10-1
        D=Dictionary(Order=Array([ocg])),
    )
    pdf.save(path)


def test_fix_all_names_optional_content_configs(tmp_path):
    src = tmp_path / "ocg.pdf"
    out = tmp_path / "ocg_fixed.pdf"
    _pdf_with_unnamed_ocg(src)

    report = fix_all(src, out)

    with pikepdf.open(out) as fixed:
        d = fixed.Root.OCProperties.D
        assert str(d.get("/Name", "")).strip(), \
            "fix_all must set /Name on the default OC config (7.10-1)"
    assert any("optional content" in c.lower() for c in report.changes), \
        "the fix must be reported in changes"
