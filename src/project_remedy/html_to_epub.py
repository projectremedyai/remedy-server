"""HTML-to-EPUB converter producing EPUB Accessibility 1.1 packages.

This module intentionally uses only the Python standard library for the EPUB
container/package writer, avoiding an EPUB-specific runtime dependency.
BeautifulSoup remains responsible only for HTML parsing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


EPUB_A11Y_CONFORMS_TO = "EPUB Accessibility 1.1 - WCAG 2.1 Level AA"

DEFAULT_A11Y_SUMMARY = (
    "This publication includes EPUB Accessibility 1.1 - WCAG 2.1 Level AA "
    "metadata. Automated EPUBCheck and ACE results are reported separately "
    "when validators are available; manual SMART-style review is required to "
    "certify conformance."
)


@dataclass
class EPUBConversionResult:
    """Outcome of a single HTML-to-EPUB conversion."""

    output_path: Path | None = None
    success: bool = False
    error_message: str = ""
    accessibility_features: list[str] = field(default_factory=list)
    chapters: int = 0


class HTMLToEPUBConverter:
    """Converts accessible HTML to an EPUB 3 package.

    ``start`` and ``close`` are kept for interface parity with
    ``HTMLToPDFConverter``. EPUB generation itself is synchronous file I/O, so
    each conversion runs in a thread behind a concurrency semaphore.
    """

    def __init__(self, max_concurrent: int = 8) -> None:
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def start(self) -> None:
        logger.info("HTMLToEPUBConverter started (max_concurrent=%d).", self._max_concurrent)

    async def close(self) -> None:
        logger.info("HTMLToEPUBConverter closed.")

    async def convert(
        self,
        html: str,
        output_path: Path,
        *,
        title: str = "",
        language: str | None = None,
        identifier: str | None = None,
        accessibility_summary: str = DEFAULT_A11Y_SUMMARY,
    ) -> EPUBConversionResult:
        """Convert an HTML string to an EPUB package."""
        async with self._semaphore:
            try:
                return await asyncio.to_thread(
                    _build_epub,
                    html,
                    output_path,
                    title=title,
                    language=language,
                    identifier=identifier,
                    accessibility_summary=accessibility_summary,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("HTML-to-EPUB conversion failed for %s: %s", output_path.name, exc)
                return EPUBConversionResult(error_message=str(exc))

    async def convert_batch(
        self,
        items: Sequence[tuple[str, Path, str, str]],
    ) -> list[EPUBConversionResult]:
        """Convert ``(html, output_path, title, language)`` tuples concurrently."""
        tasks = [
            self.convert(html, path, title=title, language=lang)
            for html, path, title, lang in items
        ]
        return await asyncio.gather(*tasks)


@dataclass(frozen=True)
class _Chapter:
    id: str
    href: str
    title: str
    content: str


def _build_epub(
    html: str,
    output_path: Path,
    *,
    title: str,
    language: str | None,
    identifier: str | None,
    accessibility_summary: str,
) -> EPUBConversionResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    soup = BeautifulSoup(html, "lxml")

    if not title:
        title = soup.title.get_text(strip=True) if soup.title else output_path.stem
    if language is None:
        html_tag = soup.find("html")
        language = (
            str(html_tag.get("lang"))
            if isinstance(html_tag, Tag) and html_tag.get("lang")
            else "en"
        )
    uid = identifier or f"urn:uuid:{uuid.uuid4()}"

    features = _detect_accessibility_features(soup)
    access_modes = _detect_access_modes(soup)
    chapters = _build_chapters(soup, title=title, language=language)
    modified = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    files = {
        "META-INF/container.xml": _container_xml(),
        "OEBPS/content.opf": _content_opf(
            uid=uid,
            title=title,
            language=language,
            modified=modified,
            features=features,
            access_modes=access_modes,
            accessibility_summary=accessibility_summary,
            chapters=chapters,
        ),
        "OEBPS/nav.xhtml": _nav_xhtml(title=title, language=language, chapters=chapters),
        "OEBPS/toc.ncx": _toc_ncx(uid=uid, title=title, chapters=chapters),
        "OEBPS/style/main.css": _DEFAULT_CSS,
    }
    for chapter in chapters:
        files[f"OEBPS/{chapter.href}"] = chapter.content

    _write_epub_zip(output_path, files)

    logger.debug(
        "Wrote EPUB %s (%d chapters, features=%s).",
        output_path.name,
        len(chapters),
        ",".join(features),
    )
    return EPUBConversionResult(
        output_path=output_path,
        success=True,
        accessibility_features=list(features),
        chapters=len(chapters),
    )


def _detect_accessibility_features(soup: BeautifulSoup) -> list[str]:
    features: list[str] = ["structuralNavigation", "tableOfContents"]

    imgs = [img for img in soup.find_all("img") if isinstance(img, Tag)]
    if (
        imgs
        and all(img.has_attr("alt") for img in imgs)
        and any((img.get("alt") or "").strip() for img in imgs)
    ):
        features.append("alternativeText")

    if soup.find(attrs={"aria-describedby": True}) or soup.find(attrs={"longdesc": True}):
        features.append("longDescription")
    if soup.find("math"):
        features.append("MathML")

    features.append("readingOrder")

    if soup.find("track", kind=re.compile(r"^(captions|subtitles)$", re.I)):
        features.append("captions")
    if soup.find(attrs={"class": re.compile(r"\btranscript\b")}):
        features.append("transcript")
    return features


def _detect_access_modes(soup: BeautifulSoup) -> list[str]:
    modes: list[str] = ["textual"]
    if soup.find("img") or soup.find("video") or soup.find("svg"):
        modes.append("visual")
    if soup.find("audio"):
        modes.append("auditory")
    return modes


def _build_chapters(soup: BeautifulSoup, *, title: str, language: str) -> list[_Chapter]:
    body = soup.body
    if not body:
        return [
            _make_chapter(
                index=1,
                title=title,
                body_html=f"<p>{_xml_escape(title)}</p>",
                language=language,
            )
        ]

    sections = body.find_all("section", recursive=False)
    if not sections:
        sections = [
            section
            for section in body.find_all("section")
            if isinstance(section, Tag) and section.find_parent("section") is None
        ]

    if not sections:
        body_html = body.decode_contents() or f"<p>{_xml_escape(title)}</p>"
        return [_make_chapter(index=1, title=title, body_html=body_html, language=language)]

    chapters: list[_Chapter] = []
    for i, section in enumerate(sections, start=1):
        if not isinstance(section, Tag):
            continue
        heading = section.find(re.compile(r"^h[1-6]$"))
        chapter_title = heading.get_text(strip=True) if heading else f"Section {i}"
        chapters.append(
            _make_chapter(
                index=i,
                title=chapter_title,
                body_html=str(section),
                language=language,
            )
        )
    return chapters


def _make_chapter(index: int, title: str, body_html: str, *, language: str) -> _Chapter:
    href = f"chap_{index}.xhtml"
    return _Chapter(
        id=f"chap_{index}",
        href=href,
        title=title,
        content=_xhtml_document(title=title, language=language, body_html=body_html),
    )


def _xhtml_document(*, title: str, language: str, body_html: str) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{_attr(language)}" xml:lang="{_attr(language)}">
<head>
  <meta charset="utf-8"/>
  <title>{_xml_escape(title)}</title>
  <link rel="stylesheet" type="text/css" href="style/main.css"/>
</head>
<body>
{body_html}
</body>
</html>
"""


def _container_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _content_opf(
    *,
    uid: str,
    title: str,
    language: str,
    modified: str,
    features: list[str],
    access_modes: list[str],
    accessibility_summary: str,
    chapters: list[_Chapter],
) -> str:
    metadata = [
        f'<dc:identifier id="pub-id">{_xml_escape(uid)}</dc:identifier>',
        f"<dc:title>{_xml_escape(title)}</dc:title>",
        f"<dc:language>{_xml_escape(language)}</dc:language>",
        f'<meta property="dcterms:modified">{_xml_escape(modified)}</meta>',
        f'<meta property="dcterms:conformsTo">{_xml_escape(EPUB_A11Y_CONFORMS_TO)}</meta>',
    ]
    metadata.extend(
        f'<meta property="schema:accessMode">{_xml_escape(mode)}</meta>'
        for mode in access_modes
    )
    metadata.append('<meta property="schema:accessModeSufficient">textual</meta>')
    metadata.extend(
        f'<meta property="schema:accessibilityFeature">{_xml_escape(feature)}</meta>'
        for feature in features
    )
    metadata.append('<meta property="schema:accessibilityHazard">none</meta>')
    metadata.append(
        f'<meta property="schema:accessibilitySummary">{_xml_escape(accessibility_summary)}</meta>'
    )

    manifest = [
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<item id="css" href="style/main.css" media-type="text/css"/>',
    ]
    manifest.extend(
        f'<item id="{_attr(chapter.id)}" href="{_attr(chapter.href)}" media-type="application/xhtml+xml"/>'
        for chapter in chapters
    )
    spine = "\n    ".join(
        f'<itemref idref="{_attr(chapter.id)}"/>' for chapter in chapters
    )
    metadata_xml = "\n    ".join(metadata)
    manifest_xml = "\n    ".join(manifest)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="pub-id" prefix="schema: http://schema.org/ dcterms: http://purl.org/dc/terms/">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    {metadata_xml}
  </metadata>
  <manifest>
    {manifest_xml}
  </manifest>
  <spine toc="ncx">
    {spine}
  </spine>
</package>
"""


def _nav_xhtml(*, title: str, language: str, chapters: list[_Chapter]) -> str:
    items = "\n      ".join(
        f'<li><a href="{_attr(chapter.href)}">{_xml_escape(chapter.title)}</a></li>'
        for chapter in chapters
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{_attr(language)}" xml:lang="{_attr(language)}">
<head>
  <meta charset="utf-8"/>
  <title>{_xml_escape(title)} Navigation</title>
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>{_xml_escape(title)}</h1>
    <ol>
      {items}
    </ol>
  </nav>
</body>
</html>
"""


def _toc_ncx(*, uid: str, title: str, chapters: list[_Chapter]) -> str:
    points = []
    for index, chapter in enumerate(chapters, start=1):
        points.append(
            f"""<navPoint id="navPoint-{index}" playOrder="{index}">
      <navLabel><text>{_xml_escape(chapter.title)}</text></navLabel>
      <content src="{_attr(chapter.href)}"/>
    </navPoint>"""
        )
    nav_points = "\n    ".join(points)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="{_attr(uid)}"/>
    <meta name="dtb:depth" content="1"/>
    <meta name="dtb:totalPageCount" content="0"/>
    <meta name="dtb:maxPageNumber" content="0"/>
  </head>
  <docTitle><text>{_xml_escape(title)}</text></docTitle>
  <navMap>
    {nav_points}
  </navMap>
</ncx>
"""


def _write_epub_zip(output_path: Path, files: dict[str, str]) -> None:
    fixed_date = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(output_path, "w") as zf:
        mimetype_info = zipfile.ZipInfo("mimetype", fixed_date)
        mimetype_info.compress_type = zipfile.ZIP_STORED
        zf.writestr(mimetype_info, "application/epub+zip")

        for name, content in files.items():
            info = zipfile.ZipInfo(name, fixed_date)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, content.encode("utf-8"))


def _xml_escape(value: str) -> str:
    return escape(value, {'"': "&quot;", "'": "&apos;"})


def _attr(value: str) -> str:
    return _xml_escape(value)


_DEFAULT_CSS = """\
body { font-family: serif; line-height: 1.5; margin: 1em; }
h1, h2, h3, h4, h5, h6 { font-family: sans-serif; }
img { max-width: 100%; height: auto; }
table { border-collapse: collapse; }
th, td { border: 1px solid #999; padding: 0.25em 0.5em; }
caption { font-weight: bold; text-align: left; }
"""
