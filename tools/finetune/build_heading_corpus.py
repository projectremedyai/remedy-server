#!/usr/bin/env python3
"""Build a verified heading-hierarchy fine-tune corpus.

This builder implements the T5 heading-hierarchy data plan:

* generate controlled WeasyPrint PDF/UA pages with known H1-H6 labels;
* admit real agency pages only after structure-tree inspection finds /H1-/H6;
* emit balanced corrupt/pass heading_hierarchy records in the production prompt
  shape used by project_remedy.pdf_vision;
* keep PDF Association and veraPDF corpus files as fixtures, not bulk data.

Output is written under tools/finetune/data/heading_hierarchy by default, which
is ignored by tools/finetune/.gitignore.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from project_remedy.tag_tree_reader import read_tag_tree  # noqa: E402
from project_remedy.vision_prompts import heading_hierarchy_quality_prompt  # noqa: E402

HEADING_TAGS = tuple(f"H{i}" for i in range(1, 7))
HEADING_SET = set(HEADING_TAGS)
CONTEXT_TAGS = {
    "P",
    "Span",
    "L",
    "LI",
    "Lbl",
    "LBody",
    "Caption",
    "BlockQuote",
    "TH",
    "TD",
}
DEFAULT_OUT_DIR = Path("tools/finetune/data/heading_hierarchy")


DEFAULT_REAL_SOURCES = [
    {
        "family": "NIST",
        "filename": "NIST.SP.800-88r2.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-88r2.pdf",
        "license": "NIST technical series/public domain or broad reuse grant",
    },
    {
        "family": "NIST",
        "filename": "NIST.SP.800-209.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-209.pdf",
        "license": "NIST technical series/public domain or broad reuse grant",
    },
    {
        "family": "NIST",
        "filename": "NIST.SP.800-204D.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-204D.pdf",
        "license": "NIST technical series/public domain or broad reuse grant",
    },
    {
        "family": "NIST",
        "filename": "NIST.SP.800-53B.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53B.pdf",
        "license": "NIST technical series/public domain or broad reuse grant",
    },
    {
        "family": "RSA",
        "filename": "fy2017-me-g.pdf",
        "url": "https://rsa.ed.gov/sites/default/files/publications/fy2017-me-g.pdf",
        "license": "U.S. Department of Education public-domain website notice; verify third-party inserts",
    },
    {
        "family": "RSA",
        "filename": "fy2017-ma-g.pdf",
        "url": "https://rsa.ed.gov/sites/default/files/publications/fy2017-ma-g.pdf",
        "license": "U.S. Department of Education public-domain website notice; verify third-party inserts",
    },
    {
        "family": "RSA",
        "filename": "fy2017-mi-b.pdf",
        "url": "https://rsa.ed.gov/sites/default/files/publications/fy2017-mi-b.pdf",
        "license": "U.S. Department of Education public-domain website notice; verify third-party inserts",
    },
    {
        "family": "RSA",
        "filename": "fy2024-az-c.pdf",
        "url": "https://rsa.ed.gov/sites/default/files/publications/fy2024-az-c.pdf",
        "license": "U.S. Department of Education public-domain website notice; verify third-party inserts",
    },
    {
        "family": "DOJ",
        "filename": "ada-web-pria.pdf",
        "url": "https://www.ada.gov/assets/pdfs/web-pria.pdf",
        "license": "U.S. Department of Justice website public-domain notice unless indicated",
    },
    {
        "family": "DOJ",
        "filename": "ada-web-fria.pdf",
        "url": "https://www.ada.gov/assets/pdfs/web-fria.pdf",
        "license": "U.S. Department of Justice website public-domain notice unless indicated",
    },
    {
        "family": "EPA",
        "filename": "fy25-annual-report-enforcement-and-compliance.pdf",
        "url": "https://www.epa.gov/system/files/documents/2026-03/fy25-annual-report-enforcement-and-compliance.pdf",
        "license": "EPA-produced public-domain data/content; verify third-party media",
    },
    {
        "family": "EPA",
        "filename": "insecticide-strategy-final_0.pdf",
        "url": "https://www.epa.gov/system/files/documents/2025-04/insecticide-strategy-final_0.pdf",
        "license": "EPA-produced public-domain data/content; verify third-party media",
    },
    {
        "family": "EPA",
        "filename": "environmental-justice-strategic-plan-december-2024.pdf",
        "url": "https://www.epa.gov/system/files/documents/2024-12/environmental-justice-strategic-plan-december-2024.pdf",
        "license": "EPA-produced public-domain data/content; verify third-party media",
    },
    {
        "family": "EPA",
        "filename": "report-of-the-chief-foia-officer-to-the-u.s-department-of-justice-2026.pdf",
        "url": "https://www.epa.gov/system/files/documents/2026-03/report-of-the-chief-foia-officer-to-the-u.s-department-of-justice-2026.pdf",
        "license": "EPA-produced public-domain data/content; verify third-party media",
    },
    {
        "family": "EIA",
        "filename": "aeo2023_narrative.pdf",
        "url": "https://www.eia.gov/outlooks/aeo/pdf/aeo2023_narrative.pdf",
        "license": "EIA U.S. government public-domain reports/data with attribution requested",
    },
    {
        "family": "EIA",
        "filename": "outlookglobalrefining.pdf",
        "url": "https://www.eia.gov/analysis/globalrefining/outlookglobalrefining.pdf",
        "license": "EIA U.S. government public-domain reports/data with attribution requested",
    },
    {
        "family": "EIA",
        "filename": "gasolinepricestudy.pdf",
        "url": "https://www.eia.gov/analysis/studies/gasoline/pdf/gasolinepricestudy.pdf",
        "license": "EIA U.S. government public-domain reports/data with attribution requested",
    },
]


PDF_ASSOCIATION_FIXTURE = Path(
    "/tmp/remedy_heading_research/techniques-for-accessible-pdf/headings/"
    "H_09-Heading-level-7-correctly-rolemapped-to-H6/UA1_Tpdf-H_09.pdf"
)
VERAPDF_FIXTURE = Path(
    "/tmp/remedy_heading_research/veraPDF-corpus/PDF_UA-1/7.4 Headings/"
    "7.4.2 Numbered headings/7.4.2-t01-pass-c.pdf"
)


@dataclass(frozen=True)
class Entry:
    tag: str
    text: str
    is_heading: bool


@dataclass(frozen=True)
class HeadingPage:
    doc_id: str
    page: int
    entries: tuple[Entry, ...]
    source_pdf: Path
    family: str
    provenance: str
    license: str
    url: str
    sha256: str

    @property
    def heading_counts(self) -> dict[str, int]:
        return dict(Counter(e.tag for e in self.entries if e.is_heading))


@contextmanager
def patched_env(**updates: str | None):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text(text: str) -> str:
    text = (text or "").replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:180]


def collect_descendant_text(nodes: list, start_index: int) -> str:
    node = nodes[start_index]
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    if node.alt_text:
        parts.append(node.alt_text)
    j = start_index + 1
    while j < len(nodes) and nodes[j].depth > node.depth:
        if nodes[j].text:
            parts.append(nodes[j].text)
        elif nodes[j].alt_text:
            parts.append(nodes[j].alt_text)
        j += 1
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        cleaned = clean_text(part)
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return clean_text(" ".join(out))


def entries_from_nodes(nodes: list, max_lines: int) -> tuple[Entry, ...]:
    entries: list[Entry] = []
    active_heading_depths: list[int] = []
    for i, node in enumerate(nodes):
        while active_heading_depths and node.depth <= active_heading_depths[-1]:
            active_heading_depths.pop()
        if node.tag in HEADING_SET:
            text = collect_descendant_text(nodes, i)
            entries.append(Entry(tag=node.tag, text=text, is_heading=True))
            active_heading_depths.append(node.depth)
            continue
        if active_heading_depths:
            continue
        text = clean_text(node.text or node.alt_text)
        if text and node.tag in CONTEXT_TAGS:
            entries.append(Entry(tag=node.tag, text=text, is_heading=False))
    return trim_entries(entries, max_lines=max_lines)


def trim_entries(entries: Iterable[Entry], max_lines: int) -> tuple[Entry, ...]:
    entries = list(entries)
    if len(entries) <= max_lines:
        return tuple(entries)

    keep = [False] * len(entries)
    for i, entry in enumerate(entries):
        if entry.is_heading:
            keep[i] = True
            if i > 0 and not entries[i - 1].is_heading:
                keep[i - 1] = True
            if i + 1 < len(entries) and not entries[i + 1].is_heading:
                keep[i + 1] = True

    for i, entry in enumerate(entries):
        if sum(keep) >= max_lines:
            break
        if not entry.is_heading:
            keep[i] = True

    selected = [entry for entry, yes in zip(entries, keep, strict=False) if yes]
    if len(selected) > max_lines:
        selected = [entry for entry in selected if entry.is_heading][:max_lines]
    return tuple(selected)


def numbered_order(entries: tuple[Entry, ...]) -> str:
    lines = []
    for i, entry in enumerate(entries, start=1):
        line = f"{i:3d}. /{entry.tag}"
        if entry.text:
            safe = entry.text.replace('"', "'")
            line += f'  (text: "{safe}")'
        lines.append(line)
    return "\n".join(lines)


def corrupted_entries_and_findings(entries: tuple[Entry, ...]) -> tuple[tuple[Entry, ...], list[dict]]:
    headings = [(i, entry) for i, entry in enumerate(entries) if entry.is_heading]
    if not headings:
        return entries, []

    non_h1 = [(i, entry) for i, entry in headings if entry.tag != "H1"]
    corrupt = list(entries)
    findings: list[dict] = []
    if non_h1:
        for i, entry in non_h1:
            corrupt[i] = Entry(tag="H1", text=entry.text, is_heading=True)
            findings.append(
                {
                    "severity": "warning",
                    "element_index": i + 1,
                    "current_tag": "H1",
                    "visible_text": entry.text,
                    "message": f"heading flattened to H1; visual hierarchy is {entry.tag}",
                    "correct_tag": entry.tag,
                    "suggested_fix": f"Retag as {entry.tag}",
                }
            )
    else:
        i, entry = headings[0]
        corrupt[i] = Entry(tag="P", text=entry.text, is_heading=False)
        findings.append(
            {
                "severity": "error",
                "element_index": i + 1,
                "current_tag": "P",
                "visible_text": entry.text,
                "message": "title/section heading is tagged as body text",
                "correct_tag": entry.tag,
                "suggested_fix": f"Retag as {entry.tag}",
            }
        )
    return tuple(corrupt), findings


def inspect_pdf_heading_pages(
    pdf_path: Path,
    *,
    doc_id: str,
    family: str,
    provenance: str,
    license_text: str = "",
    url: str = "",
    min_headings: int = 1,
    max_lines: int = 90,
    extract_text: bool = True,
) -> list[HeadingPage]:
    env = {
        "PDF_SCREEN_READER_EXTRACT_LARGE_TEXT": "1" if extract_text else None,
        "PDF_SCREEN_READER_ALLOW_LARGE_STREAMS": "1" if extract_text else None,
        "PDF_SCREEN_READER_MAX_CONTENT_OPERATORS": "500000",
    }
    with patched_env(**env):
        report = read_tag_tree(pdf_path)

    if not report.has_structure_tree:
        return []

    digest = sha256_file(pdf_path)
    by_page: dict[int, list] = defaultdict(list)
    for node in report.nodes:
        by_page[node.page].append(node)

    pages: list[HeadingPage] = []
    for page_index in range(report.page_count):
        entries = entries_from_nodes(by_page.get(page_index, []), max_lines=max_lines)
        heading_count = sum(1 for entry in entries if entry.is_heading)
        if heading_count < min_headings:
            continue
        pages.append(
            HeadingPage(
                doc_id=doc_id,
                page=page_index + 1,
                entries=entries,
                source_pdf=pdf_path,
                family=family,
                provenance=provenance,
                license=license_text,
                url=url,
                sha256=digest,
            )
        )
    return pages


def render_page(pdf_path: Path, page_num: int, out_png: Path, dpi: int) -> None:
    if out_png.exists():
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_num - 1]
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(str(out_png))
    finally:
        doc.close()


def record_for_page(
    page: HeadingPage,
    *,
    image_rel: str,
    variant: str,
    target: dict,
    logical_order: str,
) -> dict:
    return {
        "doc_id": page.doc_id,
        "page": page.page,
        "task": "heading_hierarchy",
        "image": image_rel,
        "prompt": heading_hierarchy_quality_prompt(logical_order=logical_order),
        "draft_target": "",
        "target": json.dumps(target, ensure_ascii=False),
        "reviewed": True,
        "provenance": page.provenance,
        "variant": variant,
        "source_family": page.family,
        "source_pdf": str(page.source_pdf.resolve()),
        "source_url": page.url,
        "license": page.license,
        "structure_verified": True,
        "heading_counts": page.heading_counts,
    }


def records_for_page(page: HeadingPage, image_rel: str, emit_pass: bool) -> list[dict]:
    corrupted, findings = corrupted_entries_and_findings(page.entries)
    if not findings:
        return []
    records = [
        record_for_page(
            page,
            image_rel=image_rel,
            variant="corrupt_flattened",
            logical_order=numbered_order(corrupted),
            target={"status": "fail", "findings": findings},
        )
    ]
    if emit_pass:
        records.append(
            record_for_page(
                page,
                image_rel=image_rel,
                variant="verified_pass",
                logical_order=numbered_order(page.entries),
                target={"status": "pass", "findings": []},
            )
        )
    return records


TOPICS = [
    "Student Services Accessibility Guide",
    "Emergency Preparedness Manual",
    "Veterans Resource Handbook",
    "Financial Aid Operations Plan",
    "Program Review Technical Memo",
    "Campus Technology Standards",
    "Facilities Safety Report",
    "Enrollment Services Playbook",
    "Community Partnerships Brief",
    "Instructional Support Procedure",
]
SECTIONS = [
    "Eligibility",
    "Required Documentation",
    "Review Timeline",
    "Data Stewardship",
    "Training Requirements",
    "Quality Assurance",
    "Escalation Path",
    "Annual Review",
]
DETAILS = [
    "Applicants receive clear instructions, a documented review path, and accessible follow-up notices.",
    "The process separates policy, procedure, exceptions, and supporting references.",
    "Staff verify each milestone before publishing the final student-facing document.",
    "Records are retained with concise labels so assistive technology exposes the same hierarchy.",
]


def synthetic_entries(page_num: int) -> tuple[Entry, ...]:
    topic = TOPICS[page_num % len(TOPICS)]
    section = SECTIONS[page_num % len(SECTIONS)]
    detail = DETAILS[page_num % len(DETAILS)]
    entries = [
        Entry("H1", f"{topic} {page_num:03d}", True),
        Entry("P", detail, False),
        Entry("H2", f"{section} Overview", True),
        Entry("P", "This section introduces the policy intent and the audience affected by the rule.", False),
        Entry("H3", "Primary Requirements", True),
        Entry("P", "The requirements are grouped by ownership, timeline, and review evidence.", False),
        Entry("H4", "Step One: Intake", True),
        Entry("P", "Collect forms, validate names, and confirm the document version before processing.", False),
        Entry("H5", "Exception Handling", True),
        Entry("P", "Escalate incomplete records with a short note and a next-action date.", False),
        Entry("H6", "Local Note", True),
        Entry("P", "Use this note for campus-specific implementation details.", False),
    ]
    if page_num % 3 == 0:
        entries += (
            Entry("H2", "Implementation Checklist", True),
            Entry("P", "Confirm ownership, publication date, and review sign-off.", False),
        )
    if page_num % 5 == 0:
        entries += (
            Entry("H3", "Monitoring Signals", True),
            Entry("P", "Watch for missing dates, repeated labels, and unlabeled supplemental sections.", False),
        )
    return tuple(entries)


def synthetic_css() -> str:
    return """
@page { size: Letter; margin: 0.62in 0.7in; }
body { margin: 0; color: #1f2528; font-family: Arial, Helvetica, sans-serif; }
.page { break-after: page; min-height: 8.6in; }
.page:last-child { break-after: auto; }
.deck { font-size: 9pt; color: #56616a; margin-bottom: 0.18in; text-transform: uppercase; letter-spacing: 0.04em; }
h1, h2, h3, h4, h5, h6, p { margin: 0; letter-spacing: 0; }
p { font-size: 10.6pt; line-height: 1.42; margin: 0.08in 0 0.15in; max-width: 6.4in; }
.style-0 h1 { font-size: 29pt; font-weight: 800; border-bottom: 3px solid #245b73; padding-bottom: 0.08in; }
.style-0 h2 { font-size: 20pt; color: #245b73; margin-top: 0.18in; }
.style-0 h3 { font-size: 16pt; font-weight: 700; }
.style-0 h4 { font-size: 13.5pt; color: #344b39; }
.style-0 h5 { font-size: 11.5pt; text-transform: uppercase; color: #71502b; }
.style-0 h6 { font-size: 10.5pt; font-style: italic; color: #5d6470; }
.style-1 { font-family: Georgia, 'Times New Roman', serif; }
.style-1 h1 { font-size: 31pt; font-weight: 700; color: #283747; }
.style-1 h2 { font-size: 18pt; font-weight: 700; margin-top: 0.16in; border-left: 5px solid #8a4f2a; padding-left: 0.12in; }
.style-1 h3 { font-size: 15.5pt; color: #4c5f36; }
.style-1 h4 { font-size: 13pt; font-weight: 700; }
.style-1 h5 { font-size: 11.2pt; color: #6d3f58; }
.style-1 h6 { font-size: 10pt; color: #57626d; }
.style-2 h1 { font-size: 27pt; color: #17324d; background: #e7f0f2; padding: 0.12in; }
.style-2 h2 { font-size: 19pt; margin-top: 0.18in; color: #496321; }
.style-2 h3 { font-size: 15pt; border-bottom: 1px solid #a6b0a2; padding-bottom: 0.04in; }
.style-2 h4 { font-size: 12.8pt; color: #6b4423; }
.style-2 h5 { font-size: 11pt; font-weight: 800; }
.style-2 h6 { font-size: 9.8pt; text-transform: uppercase; color: #5c6670; }
.style-3 h1 { font-size: 30pt; font-weight: 800; color: #23302e; }
.style-3 h2 { font-size: 17.5pt; color: #8b3f30; margin-top: 0.2in; }
.style-3 h3 { font-size: 15pt; font-weight: 700; color: #275468; }
.style-3 h4 { font-size: 13pt; border-left: 4px solid #557a46; padding-left: 0.1in; }
.style-3 h5 { font-size: 11.2pt; color: #4b5561; }
.style-3 h6 { font-size: 10pt; font-style: italic; }
"""


def synthetic_html(page_count: int) -> tuple[str, dict[int, tuple[Entry, ...]]]:
    page_entries: dict[int, tuple[Entry, ...]] = {}
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<title>Remedy Synthetic Heading Hierarchy Corpus</title>",
        f"<style>{synthetic_css()}</style></head><body>",
    ]
    for page_num in range(1, page_count + 1):
        entries = synthetic_entries(page_num)
        page_entries[page_num] = entries
        style = f"style-{page_num % 4}"
        parts.append(f'<section class="page {style}">')
        parts.append(f'<div class="deck">Synthetic heading hierarchy sample {page_num:03d}</div>')
        for entry in entries:
            text = html.escape(entry.text)
            if entry.is_heading:
                parts.append(f"<{entry.tag.lower()}>{text}</{entry.tag.lower()}>")
            else:
                parts.append(f"<p>{text}</p>")
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts), page_entries


def default_weasy_env() -> dict[str, str]:
    env = os.environ.copy()
    if "DYLD_FALLBACK_LIBRARY_PATH" not in env:
        candidates = [
            "/opt/homebrew/lib",
            "/opt/homebrew/opt/glib/lib",
            "/opt/homebrew/opt/pango/lib",
            "/opt/homebrew/opt/harfbuzz/lib",
            "/opt/homebrew/opt/fontconfig/lib",
        ]
        existing = [p for p in candidates if Path(p).exists()]
        if existing:
            env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(existing)
    return env


def run_weasyprint(weasyprint: str, html_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [weasyprint, "--pdf-variant=pdf/ua-1", str(html_path), str(pdf_path)]
    subprocess.run(cmd, check=True, env=default_weasy_env())


def run_verapdf(pdf_path: Path) -> dict:
    exe = shutil.which("verapdf")
    if not exe:
        return {"available": False, "passed": None, "summary": "verapdf not found"}
    proc = subprocess.run(
        [exe, "-f", "ua1", str(pdf_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "available": True,
        "passed": "isCompliant=\"true\"" in proc.stdout or "<isCompliant>true</isCompliant>" in proc.stdout,
        "exit_code": proc.returncode,
        "summary": "\n".join(proc.stdout.splitlines()[:20]),
    }


def build_synthetic_pages(args, out_dir: Path) -> tuple[list[HeadingPage], dict]:
    synthetic_dir = out_dir / "synthetic"
    html_path = synthetic_dir / "weasy_heading_corpus.html"
    pdf_path = synthetic_dir / "weasy_heading_corpus.pdf"
    html_text, page_entries = synthetic_html(args.synthetic_pages)
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")
    run_weasyprint(args.weasyprint, html_path, pdf_path)

    fixture_pages = inspect_pdf_heading_pages(
        pdf_path,
        doc_id="synthetic_weasy",
        family="synthetic",
        provenance="synthetic-weasyprint-verified",
        license_text="generated by Remedy corpus builder",
        url="",
        min_headings=1,
        max_lines=args.max_structure_lines,
        extract_text=False,
    )
    counts_by_page = {page.page: page.heading_counts for page in fixture_pages}
    missing = [
        page_num
        for page_num in range(1, args.synthetic_pages + 1)
        if not all(counts_by_page.get(page_num, {}).get(tag, 0) >= 1 for tag in HEADING_TAGS)
    ]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise RuntimeError(f"WeasyPrint output missing H1-H6 tags on synthetic pages: {preview}")

    digest = sha256_file(pdf_path)
    pages = [
        HeadingPage(
            doc_id=f"synthetic_weasy_{page_num:04d}",
            page=1,
            entries=trim_entries(page_entries[page_num], args.max_structure_lines),
            source_pdf=pdf_path,
            family="synthetic",
            provenance="synthetic-weasyprint-verified",
            license="generated by Remedy corpus builder",
            url="",
            sha256=digest,
        )
        for page_num in range(1, args.synthetic_pages + 1)
    ]
    verapdf_result = run_verapdf(pdf_path) if args.run_verapdf else {"available": False, "passed": None}
    return pages, {
        "html": str(html_path),
        "pdf": str(pdf_path),
        "pages_requested": args.synthetic_pages,
        "pages_verified_h1_h6": len(pages),
        "sha256": digest,
        "verapdf": verapdf_result,
    }


def download_or_copy_sources(args, out_dir: Path) -> list[dict]:
    raw_dir = out_dir / "real_sources"
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict] = []
    source_dir = Path(args.real_source_dir).expanduser() if args.real_source_dir else None

    seen_destinations: set[Path] = set()
    for src in DEFAULT_REAL_SOURCES:
        filename = src["filename"]
        dest = raw_dir / filename
        if dest in seen_destinations:
            continue
        seen_destinations.add(dest)

        local = source_dir / filename if source_dir else None
        if local and local.exists():
            shutil.copy2(local, dest)
        elif not dest.exists():
            try:
                with urllib.request.urlopen(src["url"], timeout=60) as response:
                    dest.write_bytes(response.read())
            except Exception as exc:
                print(f"skip download {src['url']}: {exc}", file=sys.stderr)
                continue
        copied.append({**src, "path": dest, "sha256": sha256_file(dest)})
    return copied


def select_real_pages(all_pages: list[HeadingPage], target: int, per_pdf_cap: int) -> list[HeadingPage]:
    by_family: dict[str, list[HeadingPage]] = defaultdict(list)
    per_pdf_seen: Counter[str] = Counter()
    for page in sorted(all_pages, key=lambda p: (p.family, p.doc_id, p.page)):
        key = str(page.source_pdf)
        if per_pdf_seen[key] >= per_pdf_cap:
            continue
        per_pdf_seen[key] += 1
        by_family[page.family].append(page)

    selected: list[HeadingPage] = []
    families = sorted(by_family)
    while len(selected) < target and any(by_family.values()):
        for family in families:
            if by_family[family]:
                selected.append(by_family[family].pop(0))
                if len(selected) >= target:
                    break
    return selected


def build_real_pages(args, out_dir: Path) -> tuple[list[HeadingPage], list[dict]]:
    source_infos = download_or_copy_sources(args, out_dir)
    inspected: list[dict] = []
    candidates: list[HeadingPage] = []
    for info in source_infos:
        path = Path(info["path"])
        pages = inspect_pdf_heading_pages(
            path,
            doc_id=path.stem,
            family=info["family"],
            provenance="real-structure-verified",
            license_text=info["license"],
            url=info["url"],
            min_headings=args.min_real_headings,
            max_lines=args.max_structure_lines,
            extract_text=True,
        )
        candidates.extend(pages)
        inspected.append(
            {
                "family": info["family"],
                "filename": info["filename"],
                "url": info["url"],
                "license": info["license"],
                "sha256": info["sha256"],
                "verified_heading_pages": len(pages),
                "heading_counts": dict(sum((Counter(p.heading_counts) for p in pages), Counter())),
            }
        )

    selected = select_real_pages(candidates, args.real_pages, args.real_pages_per_pdf_cap)
    if len(selected) < args.min_real_pages:
        raise RuntimeError(
            f"Only found {len(selected)} verified real heading pages; "
            f"minimum requested is {args.min_real_pages}"
        )
    return selected, inspected


def write_records(
    pages: list[HeadingPage],
    *,
    out_dir: Path,
    dpi: int,
    emit_pass: bool,
    synthetic_pdf_page_lookup: dict[str, int],
) -> list[dict]:
    records: list[dict] = []
    renders = out_dir / "renders"
    for page in pages:
        if page.family == "synthetic":
            source_page = synthetic_pdf_page_lookup[page.doc_id]
            png_name = f"{page.doc_id}_p1_{dpi}dpi.png"
            render_page(page.source_pdf, source_page, renders / png_name, dpi)
        else:
            safe_doc = re.sub(r"[^A-Za-z0-9_.-]+", "_", page.doc_id)
            png_name = f"{page.family}_{safe_doc}_p{page.page}_{dpi}dpi.png"
            render_page(page.source_pdf, page.page, renders / png_name, dpi)
        image_rel = f"renders/{png_name}"
        records.extend(records_for_page(page, image_rel=image_rel, emit_pass=emit_pass))
    return records


def to_conversation(rec: dict) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rec["image"]},
                    {"type": "text", "text": rec["prompt"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": rec["target"]}]},
        ],
        "meta": {
            "doc_id": rec.get("doc_id"),
            "page": rec.get("page"),
            "task": rec.get("task"),
            "variant": rec.get("variant"),
            "source_family": rec.get("source_family"),
        },
    }


def split_key(rec: dict) -> float:
    key = f"{rec['doc_id']}|{rec['page']}|{rec['task']}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def fixture_manifest() -> list[dict]:
    out = []
    for name, path in [
        ("PDF Association techniques", PDF_ASSOCIATION_FIXTURE),
        ("veraPDF corpus", VERAPDF_FIXTURE),
    ]:
        if not path.exists():
            out.append({"name": name, "path": str(path), "available": False})
            continue
        pages = inspect_pdf_heading_pages(
            path,
            doc_id=path.stem,
            family="unit-fixture",
            provenance="unit-fixture-not-training",
            min_headings=1,
            max_lines=40,
            extract_text=False,
        )
        out.append(
            {
                "name": name,
                "path": str(path),
                "available": True,
                "sha256": sha256_file(path),
                "verified_heading_pages": len(pages),
                "heading_counts": dict(sum((Counter(p.heading_counts) for p in pages), Counter())),
                "training_use": "unit-test fixture only; excluded from corpus records",
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--synthetic-pages", type=int, default=400)
    ap.add_argument("--real-pages", type=int, default=240)
    ap.add_argument("--min-real-pages", type=int, default=150)
    ap.add_argument("--real-pages-per-pdf-cap", type=int, default=25)
    ap.add_argument("--min-real-headings", type=int, default=1)
    ap.add_argument("--render-dpi", type=int, default=200)
    ap.add_argument("--max-structure-lines", type=int, default=90)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--real-source-dir", type=str, default="/tmp/remedy_heading_research")
    ap.add_argument("--weasyprint", default="/tmp/remedy_heading_research/weasy-venv/bin/weasyprint")
    ap.add_argument("--no-pass-negatives", action="store_true")
    ap.add_argument("--no-verapdf", dest="run_verapdf", action="store_false")
    ap.set_defaults(run_verapdf=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not 300 <= args.synthetic_pages <= 500:
        raise SystemExit("--synthetic-pages must be in the requested 300-500 range")
    if not 150 <= args.real_pages <= 300:
        raise SystemExit("--real-pages must be in the requested 150-300 range")
    if not Path(args.weasyprint).exists() and shutil.which(args.weasyprint) is None:
        raise SystemExit(f"WeasyPrint command not found: {args.weasyprint}")

    random.seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_pages, synthetic_info = build_synthetic_pages(args, out_dir)
    real_pages, real_inspected = build_real_pages(args, out_dir)
    selected_pages = synthetic_pages + real_pages

    synthetic_lookup = {
        f"synthetic_weasy_{page_num:04d}": page_num
        for page_num in range(1, args.synthetic_pages + 1)
    }
    records = write_records(
        selected_pages,
        out_dir=out_dir,
        dpi=args.render_dpi,
        emit_pass=not args.no_pass_negatives,
        synthetic_pdf_page_lookup=synthetic_lookup,
    )

    write_jsonl(out_dir / "drafts.jsonl", records)
    conversations = [to_conversation(rec) for rec in records]
    train = [row for rec, row in zip(records, conversations, strict=False) if split_key(rec) >= args.val_frac]
    val = [row for rec, row in zip(records, conversations, strict=False) if split_key(rec) < args.val_frac]
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    page_counts = Counter(page.family for page in selected_pages)
    record_counts = Counter(rec["source_family"] for rec in records)
    manifest = {
        "task": "heading_hierarchy",
        "created_by": "tools/finetune/build_heading_corpus.py",
        "requested": {
            "synthetic_pages": args.synthetic_pages,
            "real_pages": args.real_pages,
            "render_dpi": args.render_dpi,
            "emit_pass_negatives": not args.no_pass_negatives,
        },
        "actual": {
            "source_pages": dict(page_counts),
            "real_pages_selected": len(real_pages),
            "synthetic_pages_selected": len(synthetic_pages),
            "records": len(records),
            "records_by_family": dict(record_counts),
            "train_records": len(train),
            "val_records": len(val),
        },
        "synthetic": synthetic_info,
        "real_sources": real_inspected,
        "fixtures": fixture_manifest(),
        "gate": "Every selected source page was admitted only after structure-tree inspection found real /H1-/H6 tags.",
        "excluded_bulk_sources": [
            "PDF Association techniques",
            "veraPDF corpus",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["actual"], indent=2, sort_keys=True))
    print(f"wrote {out_dir / 'drafts.jsonl'}")
    print(f"wrote {out_dir / 'train.jsonl'}")
    print(f"wrote {out_dir / 'val.jsonl'}")
    print(f"wrote {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
