from models.schemas import JobDescriptionData
from parsers.document_parser import validate_jd_document, DocumentError
from services.extraction import extract


def extract_jd_structure(text: str, *, client=None) -> JobDescriptionData:
    validation = validate_jd_document(text)
    if not validation.valid:
        raise DocumentError('This does not appear to be a job description. ' + ' '.join(validation.reasons))
    return extract(JobDescriptionData, text, client=client)
