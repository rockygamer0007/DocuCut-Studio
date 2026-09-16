from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf


class PDFEngineError(Exception):
    """Base error for PDF processing."""


class PDFPasswordRequired(PDFEngineError):
    """Raised when an encrypted PDF requires a password."""


class PDFInvalidPassword(PDFEngineError):
    """Raised when the supplied password is incorrect."""


@dataclass
class PDFDocumentInfo:
    path: Path
    pages: int
    encrypted: bool


class PDFEngine:
    """
    PDF loading/rendering layer for DocuCut Studio.

    This module intentionally contains no GUI code.
    """

    def inspect(self, path: str | Path) -> PDFDocumentInfo:
        pdf_path = Path(path)

        if not pdf_path.is_file():
            raise PDFEngineError(f"File not found: {pdf_path}")

        try:
            document = pymupdf.open(str(pdf_path))
        except Exception as exc:
            raise PDFEngineError(
                f"Unable to open PDF: {pdf_path}"
            ) from exc

        try:
            return PDFDocumentInfo(
                path=pdf_path,
                pages=document.page_count,
                encrypted=bool(document.needs_pass),
            )
        finally:
            document.close()

    def open(
        self,
        path: str | Path,
        password: Optional[str] = None,
    ) -> pymupdf.Document:
        pdf_path = Path(path)

        if not pdf_path.is_file():
            raise PDFEngineError(f"File not found: {pdf_path}")

        try:
            document = pymupdf.open(str(pdf_path))
        except Exception as exc:
            raise PDFEngineError(
                f"Unable to open PDF: {pdf_path}"
            ) from exc

        if document.needs_pass:
            if not password:
                document.close()
                raise PDFPasswordRequired(str(pdf_path))

            authenticated = document.authenticate(password)

            if not authenticated:
                document.close()
                raise PDFInvalidPassword(str(pdf_path))

        return document

    def render_page(
        self,
        document: pymupdf.Document,
        page_number: int = 0,
        dpi: int = 300,
    ):
        if dpi <= 0:
            raise ValueError("DPI must be greater than zero.")

        if page_number < 0 or page_number >= document.page_count:
            raise IndexError(
                f"Page {page_number} is outside the document."
            )

        page = document.load_page(page_number)

        scale = dpi / 72.0
        matrix = pymupdf.Matrix(scale, scale)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        return pixmap

    def render_clip(
        self,
        document: pymupdf.Document,
        box: tuple[float, float, float, float],
        page_number: int = 0,
        dpi: int = 300,
    ):
        """Render a normalized page rectangle at the requested DPI."""
        if dpi <= 0:
            raise ValueError("DPI must be greater than zero.")

        if len(box) != 4:
            raise ValueError("box must contain x0, y0, x1, y1.")

        if page_number < 0 or page_number >= document.page_count:
            raise IndexError(
                f"Page {page_number} is outside the document."
            )

        x0, y0, x1, y1 = (float(value) for value in box)

        if not (0.0 <= x0 < x1 <= 1.0):
            raise ValueError(f"Invalid normalized x coordinates: {box!r}")

        if not (0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"Invalid normalized y coordinates: {box!r}")

        page = document.load_page(page_number)
        rect = page.rect

        clip = pymupdf.Rect(
            rect.x0 + x0 * rect.width,
            rect.y0 + y0 * rect.height,
            rect.x0 + x1 * rect.width,
            rect.y0 + y1 * rect.height,
        )

        scale = dpi / 72.0
        matrix = pymupdf.Matrix(scale, scale)

        return page.get_pixmap(
            matrix=matrix,
            clip=clip,
            alpha=False,
        )

    def render_page_to_png(
        self,
        document: pymupdf.Document,
        output_path: str | Path,
        page_number: int = 0,
        dpi: int = 300,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        pixmap = self.render_page(
            document=document,
            page_number=page_number,
            dpi=dpi,
        )

        pixmap.save(str(output))

        return output
