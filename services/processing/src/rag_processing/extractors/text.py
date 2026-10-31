"""Plain-text, HTML, and XML extractors.

HTML is walked at the block level (h1-h6, p, li, td, ...) rather than via a flat
``text_content()`` dump — the latter concatenates block elements with no
separators, losing both line structure and heading detection. Each block becomes
a typed ``TextBlock`` so headings survive into chunking and citations.
"""

from __future__ import annotations

import asyncio

from lxml import etree, html

from rag_core.models.content import BlockType, ExtractedContent, Table, TextBlock
from rag_core.models.enums import DocumentType
from rag_processing.extractors.base import ExtractionContext, Extractor
from rag_processing.extractors.layout import paginate_items, paginate_text

# Block-level tags we surface as their own text blocks.
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = _HEADING_TAGS | {"p", "li", "td", "th", "blockquote", "pre", "dd", "dt", "caption"}
_DROP_TAGS = ("//script", "//style", "//noscript", "//head")


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class TextExtractor(Extractor):
    supported = (DocumentType.TXT,)

    async def extract(self, data: bytes, ctx: ExtractionContext) -> ExtractedContent:
        text = await asyncio.to_thread(_decode, data)
        return ExtractedContent(document_id=ctx.document_id, pages=paginate_text(text))


class HtmlExtractor(Extractor):
    supported = (DocumentType.HTML,)

    async def extract(self, data: bytes, ctx: ExtractionContext) -> ExtractedContent:
        items = await asyncio.to_thread(self._parse, data)
        return ExtractedContent(
            document_id=ctx.document_id, pages=paginate_items(items)
        )

    @staticmethod
    def _parse(data: bytes) -> list[TextBlock | Table]:
        tree = html.fromstring(data)
        for path in _DROP_TAGS:
            for node in tree.xpath(path):
                parent = node.getparent()
                if parent is not None:
                    parent.remove(node)

        items: list[TextBlock | Table] = []
        for el in tree.iter():
            tag = el.tag
            if not isinstance(tag, str) or tag not in _BLOCK_TAGS:
                continue
            # Skip nested block elements whose ancestor is also a block tag, to
            # avoid emitting the same text twice (outer text_content includes inner).
            if any(
                isinstance(a.tag, str) and a.tag in _BLOCK_TAGS for a in el.iterancestors()
            ):
                continue
            text = " ".join(el.text_content().split()).strip()
            if not text:
                continue
            if tag in _HEADING_TAGS:
                items.append(
                    TextBlock(text=text, role=BlockType.HEADING, page=1, heading_level=int(tag[1]))
                )
            else:
                items.append(TextBlock(text=text, role=BlockType.PARAGRAPH, page=1))
        return items


class XmlExtractor(Extractor):
    supported = (DocumentType.XML,)

    async def extract(self, data: bytes, ctx: ExtractionContext) -> ExtractedContent:
        def _parse() -> str:
            # Recover from malformed XML rather than failing the whole document.
            parser = etree.XMLParser(recover=True, huge_tree=True)
            root = etree.fromstring(data, parser=parser)
            if root is None:
                return _decode(data)
            parts = [t.strip() for t in root.itertext() if t and t.strip()]
            return "\n".join(parts)

        text = await asyncio.to_thread(_parse)
        return ExtractedContent(document_id=ctx.document_id, pages=paginate_text(text))
