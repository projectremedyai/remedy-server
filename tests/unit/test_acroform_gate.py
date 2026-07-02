"""FR-13: fillable-form sources must be detected before any AST rebuild."""

from __future__ import annotations

import pikepdf

from project_remedy.rebuild.acroform_gate import has_acroform


def _blank_pdf(path):
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(path)


def test_plain_pdf_has_no_acroform(tmp_path):
    p = tmp_path / "plain.pdf"
    _blank_pdf(p)
    assert has_acroform(p) is False


def test_acroform_pdf_detected(tmp_path):
    p = tmp_path / "form.pdf"
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        field = pdf.make_indirect(pikepdf.Dictionary(FT=pikepdf.Name("/Tx"), T=pikepdf.String("name")))
        pdf.Root.AcroForm = pdf.make_indirect(
            pikepdf.Dictionary(Fields=pikepdf.Array([field]))
        )
        pdf.save(p)
    assert has_acroform(p) is True


def test_empty_acroform_fields_is_not_a_form(tmp_path):
    p = tmp_path / "emptyform.pdf"
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        pdf.Root.AcroForm = pdf.make_indirect(pikepdf.Dictionary(Fields=pikepdf.Array([])))
        pdf.save(p)
    assert has_acroform(p) is False


def test_unreadable_file_is_false_not_raise(tmp_path):
    p = tmp_path / "junk.pdf"
    p.write_bytes(b"not a pdf")
    assert has_acroform(p) is False
