"""Images are skipped at listing, before anything is downloaded.

An image yields no extractable text, so indexing one costs a download and an
extraction attempt, produces no chunks, and still registers a document that
``list_documents`` counts. BRAVO's first run showed the scale: 1,167 of 2,147
indexed documents (54%) were unsearchable jpgs, and they were a large share of
why the run took twice as long as projected.

The key property here is that the skip happens from the KEY alone -- an image
that is never fetched cannot cost anything.
"""

from __future__ import annotations

import pytest
from rag_search.indexer import NON_TEXT_TYPES

from rag_core.models.enums import DocumentType
from rag_core.utils.doctypes import infer_document_type


@pytest.mark.parametrize(
    "key",
    [
        "bravo//00ZQ3BkdLEu5jQ7VEIXuXg_LoadImage6~jpg",
        "bravo//photo~jpeg",
        "bravo//scan~png",
        "foxtrot/COMPDOCCAT/diagram~tiff",
        "foxtrot/plan.bmp",
        "foxtrot/animation.gif",
    ],
)
def test_image_keys_are_classified_as_non_text(key: str) -> None:
    assert infer_document_type(key) in NON_TEXT_TYPES


@pytest.mark.parametrize(
    "key",
    [
        "bravo//0510e0e763dc_PIF-0beddd0c~pdf",
        "foxtrot/supplier_quality_manual.pdf",
        "bravo//Sheet1~xlsx",
        "foxtrot/India-Employee Manual~docx",
        "foxtrot/notes~txt",
        "foxtrot/report~html",
        "foxtrot/feed~xml",
        "foxtrot/deck~pptx",
        "foxtrot/no_extension_at_all",
    ],
)
def test_text_bearing_keys_are_kept(key: str) -> None:
    """Everything that might carry text stays -- including keys with no extension,
    which are rare but do extract."""
    assert infer_document_type(key) not in NON_TEXT_TYPES


def test_only_images_are_excluded() -> None:
    """A narrow rule: excluding anything else would silently drop real documents."""
    assert frozenset({DocumentType.IMAGE}) == NON_TEXT_TYPES


def test_the_summary_can_report_the_skip() -> None:
    """The count is surfaced, so a run that indexes little says why."""
    from rag_search.models import IndexSummary

    summary = IndexSummary(bucket="b", prefix="bravo/", namespace="default")
    assert summary.documents_non_text == 0
    summary.documents_non_text += 3
    assert summary.documents_non_text == 3
