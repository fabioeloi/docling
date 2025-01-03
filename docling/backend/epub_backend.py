"""ePub backend for docling."""
import hashlib
import base64
import io
import mimetypes
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Set, Union

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from PIL import Image as PILImage

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import (
    DoclingDocument,
    DocItemLabel,
    DocumentOrigin,
    GroupItem,
    GroupLabel,
    ImageRef,
    NodeItem,
    PictureItem,
    Size,
    TextItem,
)


class EpubBackend(AbstractDocumentBackend):
    """Backend for ePub files."""

    def __init__(
        self,
        in_doc: Any,
        path_or_stream: Union[str, Path, BinaryIO],
        **kwargs: Any,
    ) -> None:
        """Initialize EpubBackend.

        Args:
            in_doc: Input document
            path_or_stream: Path to ePub file or file-like object
            **kwargs: Additional arguments
        """
        super().__init__(in_doc, path_or_stream)
        self.file = path_or_stream if isinstance(path_or_stream, Path) else Path(str(path_or_stream))
        self.book = epub.read_epub(str(self.file))
        self.document_hash = self._calculate_hash()

    def is_valid(self) -> bool:
        """Check if the document is a valid ePub file."""
        try:
            if not self.book:
                self.book = epub.read_epub(str(self.file))
            return True
        except (ebooklib.epub.EpubException, Exception):
            return False

    @classmethod
    def supports_pagination(cls) -> bool:
        """ePub files don't have fixed pages."""
        return False

    def unload(self):
        """Unload the document."""
        self.book = None
        super().unload()

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """Return the supported formats."""
        return {InputFormat.EPUB}

    def _calculate_hash(self) -> int:
        """Calculate hash of the ePub file."""
        hasher = hashlib.sha256()
        with open(self.file, "rb") as f:
            hasher.update(f.read())
        # Convert first 8 bytes of hash to integer
        return int.from_bytes(hasher.digest()[:8], byteorder="big", signed=False)

    def _initialize_document(self) -> DoclingDocument:
        """Initialize DoclingDocument with basic metadata."""
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/octet-stream",  # Using generic binary MIME type
            binary_hash=self.document_hash,
        )
        return DoclingDocument(name=self.file.stem or "file", origin=origin)

    def _extract_metadata(self, doc: DoclingDocument) -> None:
        """Extract and store all available metadata."""
        metadata_fields = [
            'title', 'creator', 'language', 'publisher', 
            'rights', 'identifier', 'coverage', 'date'
        ]
        
        metadata_texts = []
        for field in metadata_fields:
            if value := self.book.get_metadata('DC', field):
                text = f"{field.title()}: {value[0][0]}"
                metadata_texts.append(text)
        
        if metadata_texts:
            combined_text = "\n".join(metadata_texts)
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=combined_text,
                orig=combined_text
            )

    def _get_heading_level(self, content: str) -> Optional[int]:
        """Get heading level from HTML content."""
        soup = BeautifulSoup(content, 'html.parser')
        for i in range(1, 7):  # h1 to h6
            if heading := soup.find(f'h{i}'):
                return i
        return None

    def _create_document_structure(self, doc: DoclingDocument) -> None:
        """Create document structure."""
        # Add title as section header if available
        if self.book.get_metadata('DC', 'title'):
            title = self.book.get_metadata('DC', 'title')[0][0]
            doc.add_text(
                label=DocItemLabel.SECTION_HEADER,
                text=title,
                orig=title
            )
        
        # Create document structure
        current_section = None
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content().decode('utf-8'), 'html.parser')
                
                # Create a section for this chapter
                current_chapter = GroupItem(
                    name=item.get_name(),
                    label=GroupLabel.SECTION,
                    self_ref=f"#/groups/{len(doc.groups)}",
                    parent=doc.body.get_ref()
                )
                doc.groups.append(current_chapter)
                doc.body.children.append(current_chapter.get_ref())
                
                # Extract text content
                for text_item in self._extract_text_content(str(soup)):
                    text_index = len(doc.texts)
                    text_item.self_ref = f"#/texts/{text_index}"
                    text_item.parent = current_chapter.get_ref()
                    doc.texts.append(text_item)
                    current_chapter.children.append(text_item.get_ref())
                
                # Extract images
                for img in soup.find_all('img'):
                    image_ref = self._extract_image(img)
                    if image_ref:
                        picture_index = len(doc.pictures)
                        picture = PictureItem(
                            label=DocItemLabel.PICTURE,
                            self_ref=f"#/pictures/{picture_index}",
                            parent=current_chapter.get_ref(),
                            image=image_ref
                        )
                        doc.pictures.append(picture)
                        current_chapter.children.append(picture.get_ref())
                
                # Extract footnotes
                for note in soup.find_all('aside', {'epub:type': 'footnote'}):
                    note_id = note.get('id', '')
                    note_text = note.get_text().strip()
                    if note_text:
                        footnote_index = len(doc.texts)
                        footnote = TextItem(
                            label=DocItemLabel.FOOTNOTE,
                            text=self._process_text_content(note_text),
                            orig=note_text,
                            self_ref=f"#/texts/{footnote_index}",
                            parent=current_chapter.get_ref()
                        )
                        doc.texts.append(footnote)
                        current_chapter.children.append(footnote.get_ref())
                        # Link to reference if available
                        ref_id = note.get('href', '').lstrip('#')
                        if ref_id:
                            ref = soup.find(id=ref_id)
                            if ref:
                                ref_text = ref.get_text().strip()
                                ref_index = len(doc.texts)
                                ref_item = TextItem(
                                    label=DocItemLabel.REFERENCE,
                                    text=ref_text,
                                    orig=ref_text,
                                    self_ref=f"#/texts/{ref_index}",
                                    parent=current_chapter.get_ref()
                                )
                                doc.texts.append(ref_item)
                                current_chapter.children.append(ref_item.get_ref())

    def _extract_title(self, content: str) -> str:
        """Extract title from HTML content."""
        soup = BeautifulSoup(content, 'html.parser')
        for i in range(1, 7):  # h1 to h6
            if heading := soup.find(f'h{i}'):
                return heading.get_text().strip()
        return "Untitled"

    def _extract_images(self, doc: DoclingDocument) -> None:
        """Extract and store images from ePub."""
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_IMAGE:
                image_data = item.get_content()
                image_str = base64.b64encode(image_data).decode('utf-8')
                image_uri = f"data:{item.media_type};base64,{image_str}"
                
                # Create a PIL Image to get dimensions
                img = PILImage.open(io.BytesIO(image_data))
                
                image_ref = ImageRef(
                    mimetype=item.media_type,
                    dpi=72,  # Default DPI
                    size=Size(width=img.width, height=img.height),
                    uri=image_uri
                )
                doc.add_picture(
                    image=image_ref,
                    annotations=[],
                    caption=None
                )

    def _extract_toc(self, doc: DoclingDocument) -> None:
        """Extract and preserve table of contents structure."""
        def process_toc_item(item: Any, parent: Optional[NodeItem] = None) -> None:
            if isinstance(item, tuple):
                title = item[0].title if item[0] else "Untitled"
            else:
                title = item.title if hasattr(item, 'title') else "Untitled"
            
            # Create a group for this TOC item
            node = doc.add_group(
                name=title,
                label=GroupLabel.SECTION,
                parent=parent
            )
            
            # Add the TOC text under this group
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=f"[TOC] {title}",
                orig=title,
                parent=node
            )
            
            if isinstance(item, tuple):
                for child in item[1]:
                    process_toc_item(child, node)
            else:
                for child in getattr(item, 'items', []):
                    process_toc_item(child, node)
        
        # Create a root TOC group
        toc_root = doc.add_group(
            name="Table of Contents",
            label=GroupLabel.SECTION
        )
        
        for item in self.book.toc:
            process_toc_item(item, toc_root)

    def _extract_text_with_styles(self, html_content: str) -> List[TextItem]:
        """Extract text while preserving important styles."""
        soup = BeautifulSoup(html_content, 'html.parser')
        items = []
        
        style_map = {
            'code': DocItemLabel.CODE,
            'em': DocItemLabel.TEXT,
            'strong': DocItemLabel.TEXT,
            'blockquote': DocItemLabel.TEXT,
        }
        
        for tag, label in style_map.items():
            for elem in soup.find_all(tag):
                text = elem.get_text().strip()
                if text:
                    if tag == 'code':
                        items.append(TextItem(
                            text=self._process_text_content(text),
                            orig=text,
                            label=label,
                            self_ref="#"  # Will be updated by DoclingDocument
                        ))
                    else:
                        items.append(TextItem(
                            text=f"[{tag}] {self._process_text_content(text)}",
                            orig=text,
                            label=label,
                            self_ref="#"  # Will be updated by DoclingDocument
                        ))
        
        # Get remaining text
        for text in soup.stripped_strings:
            if text and not any(text in item.text for item in items):
                items.append(TextItem(
                    text=self._process_text_content(text),
                    orig=text,
                    label=DocItemLabel.TEXT,
                    self_ref="#"  # Will be updated by DoclingDocument
                ))
        
        return items

    def _extract_text_content(self, content: str) -> List[TextItem]:
        """Extract text content from HTML."""
        soup = BeautifulSoup(content, 'html.parser')
        items = []
        
        # Extract text with styles
        items.extend(self._extract_text_with_styles(content))
        
        # Extract footnotes
        for footnote in soup.find_all('a', {'epub:type': 'noteref'}):
            text = footnote.get_text().strip()
            if text:
                items.append(TextItem(
                    text=text,
                    orig=text,
                    label=DocItemLabel.FOOTNOTE,
                    self_ref="#"  # Will be updated by DoclingDocument
                ))
        
        return items

    def _handle_notes(self, doc: DoclingDocument) -> None:
        """Extract and link footnotes and endnotes."""
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8')
                soup = BeautifulSoup(content, 'html.parser')
                
                # Find footnotes (common patterns in ePubs)
                note_patterns = ['.footnote', '.endnote', '.note', '[id^="fn"]']
                for pattern in note_patterns:
                    for note in soup.select(pattern):
                        note_text = note.get_text().strip()
                        if note_text:
                            footnote = doc.add_text(
                                label=DocItemLabel.FOOTNOTE,
                                text=self._process_text_content(note_text),
                                orig=note_text
                            )
                            # Link to reference if available
                            ref_id = note.get('href', '').lstrip('#')
                            if ref_id:
                                doc.add_reference(footnote, ref_id)

    def _extract_image(self, img_tag) -> Optional[ImageRef]:
        """Extract image data from an img tag."""
        src = img_tag.get('src', '')
        if not src:
            return None
        
        # Find the image item in the ePub book
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_IMAGE and item.get_name() == src:
                image_data = item.get_content()
                
                # Create a PIL Image from the data
                img = PILImage.open(io.BytesIO(image_data))
                
                # Get image dimensions and DPI
                size = Size(width=img.width, height=img.height)
                dpi = int(img.info.get('dpi', (72, 72))[0])  # Convert to int
                
                # Create data URI from image data
                mimetype = mimetypes.guess_type(src)[0] or 'image/jpeg'
                data_uri = f"data:{mimetype};base64,{base64.b64encode(image_data).decode()}"
                
                return ImageRef(
                    mimetype=mimetype,
                    dpi=dpi,
                    size=size,
                    uri=data_uri
                )
        
        return None

    def _process_text_content(self, text: str) -> str:
        """Smart text processing to improve readability."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR issues
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')
        
        # Handle special characters
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        
        return text.strip()

    def _validate_conversion(self, doc: DoclingDocument) -> bool:
        """Validate the converted document."""
        # Check for required elements
        has_title = any(item.label == DocItemLabel.SECTION_HEADER for item in doc.texts)
        has_content = any(item.label == DocItemLabel.TEXT for item in doc.texts)
        
        # Check document structure
        has_structure = len(doc.texts) > 0
        
        # Check for critical errors
        has_errors = False  # Add error checking logic if needed
        
        return has_title and has_content and has_structure and not has_errors

    def convert(self) -> DoclingDocument:
        """Convert ePub document to DoclingDocument."""
        doc = self._initialize_document()
        
        # Extract metadata
        self._extract_metadata(doc)
        
        # Create document structure
        self._create_document_structure(doc)
        
        # Extract images
        self._extract_images(doc)
        
        # Extract TOC
        self._extract_toc(doc)
        
        # Process content with styles
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8')
                for text_item in self._extract_text_with_styles(content):
                    doc.texts.append(text_item)
        
        # Handle footnotes
        self._handle_notes(doc)
        
        # Validate conversion
        if not self._validate_conversion(doc):
            raise ValueError("Document conversion validation failed")
        
        return doc
