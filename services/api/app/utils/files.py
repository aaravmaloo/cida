from fastapi import HTTPException, UploadFile, status
from pypdf import PdfReader
from docx import Document

ALLOWED_CONTENT_TYPES = {
    "text/plain": "txt",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


async def extract_text_from_upload(file: UploadFile, max_upload_bytes: int) -> str:
    raw = await file.read(max_upload_bytes + 1)
    if len(raw) > max_upload_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type")

    ext = ALLOWED_CONTENT_TYPES[content_type]
    if ext == "txt":
        return raw.decode("utf-8", errors="ignore")

    if ext == "pdf":
        import io

        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if ext == "docx":
        import io

        doc = Document(io.BytesIO(raw))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())

    raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type")

