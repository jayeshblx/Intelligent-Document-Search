"""Document-type inference from object keys/filenames.

Extension-based and cheap by design: it runs over millions of keys during
discovery and again per document during processing. True format sniffing (e.g.
is a .pdf actually image-only?) happens in processing with the bytes in hand.
"""

from __future__ import annotations

import re

from rag_core.models.enums import DocumentType

# Extension may be separated by '.' or '~' (some upload pipelines replace the dot,
# e.g. "report~pdf", "TEST~docx"). Match a trailing alphanumeric extension.
_EXT_RE = re.compile(r"[.~]([A-Za-z0-9]+)$")

_EXT_MAP: dict[str, DocumentType] = {
    "pdf": DocumentType.PDF,
    "docx": DocumentType.DOCX,
    "doc": DocumentType.DOCX,
    "pptx": DocumentType.PPTX,
    "ppt": DocumentType.PPTX,
    "xlsx": DocumentType.XLSX,
    "xls": DocumentType.XLSX,
    "xml": DocumentType.XML,
    "html": DocumentType.HTML,
    "htm": DocumentType.HTML,
    "txt": DocumentType.TXT,
    "png": DocumentType.IMAGE,
    "jpg": DocumentType.IMAGE,
    "jpeg": DocumentType.IMAGE,
    "tif": DocumentType.IMAGE,
    "tiff": DocumentType.IMAGE,
    "bmp": DocumentType.IMAGE,
    "gif": DocumentType.IMAGE,
}


def infer_document_type(key: str) -> DocumentType:
    """Infer document type from a key's extension (``.ext`` or ``~ext``);
    UNKNOWN if no recognisable extension is present."""
    match = _EXT_RE.search(key)
    if not match:
        return DocumentType.UNKNOWN
    return _EXT_MAP.get(match.group(1).lower(), DocumentType.UNKNOWN)
