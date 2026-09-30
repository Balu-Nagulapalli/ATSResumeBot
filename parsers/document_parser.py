"""Bounded, in-memory extraction and document-type heuristics; no OCR."""
import hashlib
import re
import unicodedata
import zipfile
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from config import MAX_FILE_BYTES, MAX_PAGES, MAX_TEXT_CHARS, MIN_TEXT_CHARS
from models.schemas import ValidationResult


class DocumentError(ValueError):
    pass


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub('[\u2028\u2029\x0c]', '\n', text)
    text = re.sub('[\u2022\u25cf\u25aa\u2023]', '-', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(s.strip() for s in text.splitlines())).strip()


def check_text(text: str) -> str:
    text = normalize_text(text)
    if not text:
        raise DocumentError('The document is empty.')
    if len(text) < MIN_TEXT_CHARS or len(text.split()) < 12:
        raise DocumentError('The document is too short for reliable analysis.')
    if len(text) > MAX_TEXT_CHARS:
        raise DocumentError(f'The document exceeds the {MAX_TEXT_CHARS:,}-character limit.')
    if '\x00' in text or sum(c.isprintable() or c.isspace() for c in text) / len(text) < .95:
        raise DocumentError('The extracted text is unreadable. Export a text-based document.')
    return text


def extract_text(file: BinaryIO, filename: str | None = None, *, resume: bool = False) -> str:
    name = filename or getattr(file, 'name', '')
    extension = Path(name).suffix.lower()
    allowed = {'.pdf', '.docx'} if resume else {'.pdf', '.docx', '.txt'}
    if extension not in allowed:
        raise DocumentError('Unsupported file extension: use ' + ', '.join(sorted(allowed)))
    file.seek(0)
    data = file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise DocumentError('Files must be 10 MB or smaller.')
    if not data:
        raise DocumentError('The document is empty.')
    try:
        if extension == '.pdf':
            with pymupdf.open(stream=data, filetype='pdf') as pdf:
                if pdf.needs_pass:
                    raise DocumentError('Password-protected PDFs are not supported.')
                if len(pdf) > MAX_PAGES:
                    raise DocumentError(f'PDFs must contain at most {MAX_PAGES} pages.')
                pages = []
                for page in pdf:
                    pages.append(page.get_text(sort=True))
                    if sum(map(len, pages)) > MAX_TEXT_CHARS:
                        raise DocumentError('The PDF contains too much text.')
                text = '\n\n'.join(pages)
                if not text.strip():
                    raise DocumentError('This PDF appears to contain no extractable text.')
        elif extension == '.docx':
            with zipfile.ZipFile(BytesIO(data)) as archive:
                if sum(i.file_size for i in archive.infolist()) > 40 * 1024 * 1024:
                    raise DocumentError('The expanded DOCX is too large.')
            doc = Document(BytesIO(data))
            parts = []
            for block in doc.iter_inner_content():
                if isinstance(block, Paragraph):
                    parts.append(block.text)
                elif isinstance(block, Table):
                    parts.extend(' | '.join(c.text for c in row.cells) for row in block.rows)
            for section in doc.sections:
                parts.extend(p.text for p in section.header.paragraphs)
                parts.extend(p.text for p in section.footer.paragraphs)
            text = '\n'.join(parts)
        else:
            try:
                text = data.decode('utf-8-sig')
            except UnicodeDecodeError:
                text = data.decode('cp1252', errors='replace')
    except DocumentError:
        raise
    except Exception:
        raise DocumentError('The document is corrupted or cannot be read. Re-export it and try again.') from None
    return check_text(text)


def document_hash(text: str) -> str:
    canonical = re.sub(r'\W+', '', normalize_text(text).casefold())
    return hashlib.sha256(canonical.encode()).hexdigest()


def reject_identical_documents(jd: str, resume: str) -> None:
    if document_hash(jd) == document_hash(resume):
        raise DocumentError('The resume is identical to the job description.')


def contact_evidence(text: str) -> list[str]:
    emails = re.findall(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}', text)
    phones = re.findall(r'(?<!\d)\+?\d[\d ()-]{7,22}\d(?!\d)', text)
    return emails + [phone for phone in phones if 9 <= len(re.sub(r'\D', '', phone)) <= 15]


def _validate(text: str, resume: bool) -> ValidationResult:
    try:
        text = check_text(text)
    except DocumentError as exc:
        return ValidationResult(valid=False, confidence=0.0, reasons=[str(exc)])
    patterns = {
        'contact information': r'[\w.+-]+@[\w.-]+\.[a-z]{2,}|linkedin\.com|\+?\d[\d ()-]{8,}\d',
        'skills': r'\b(skills|technologies|proficiencies)\b',
        'employment': r'\b(experience|employment|work history)\b',
        'education': r'\b(education|bachelor|master|university|degree)\b',
        'projects or achievements': r'\b(projects|achievements|certifications|professional summary)\b',
    } if resume else {
        'requirements': r'\b(requirements|required|must have|qualifications)\b',
        'responsibilities': r'\b(responsibilities|duties|you will|the role)\b',
        'preferences': r'\b(preferred|nice to have|bonus)\b',
        'job context': r'\b(job|position|hiring|candidate|seeking|join our|we are looking)\b',
        'experience requirement': r'\b\d+\+?\s+years?\b',
    }
    hits = [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)]
    if resume and 'contact information' in hits and not contact_evidence(text) and 'linkedin.com' not in text.casefold():
        hits.remove('contact information')
    job_language = bool(re.search(r'\b(you will|we are hiring|we are looking|the ideal candidate)\b', text, re.I))
    valid = len(hits) >= 3 and not (resume and job_language and 'contact information' not in hits)
    return ValidationResult(valid=valid, confidence=min(.95, len(hits) / 5),
                            reasons=[f'Found {h}.' for h in hits] + ([] if valid else [
                                'Insufficient independent resume indicators.' if resume else
                                'Insufficient independent job-description indicators.']))


def validate_jd_document(text: str) -> ValidationResult:
    return _validate(text, False)


def validate_resume_document(text: str) -> ValidationResult:
    return _validate(text, True)
