from models.schemas import ResumeData
from parsers.document_parser import validate_resume_document, DocumentError
from services.extraction import extract


def extract_resume_structure(text: str, *, client=None) -> ResumeData:
    validation = validate_resume_document(text)
    if not validation.valid:
        raise DocumentError('This does not appear to be a resume. ' + ' '.join(validation.reasons))
    return extract(ResumeData, text, client=client)
