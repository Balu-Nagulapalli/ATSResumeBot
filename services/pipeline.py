"""Shared orchestration for UI and benchmark evaluation."""
from datetime import date
import hashlib
import logging
from typing import MutableMapping
from config import SEMANTIC_MATCHING
from parsers.document_parser import reject_identical_documents, normalize_text
from services.jd_extractor import extract_jd_structure
from services.resume_extractor import extract_resume_structure
from services.matcher import match_documents
from services.semantic import refine_semantically
from services.scorer import score_match
from services.feedback import generate_feedback
from services.extraction_cache import ExtractionCache, SCHEMAS, extraction_identity


def extraction_hash(text: str) -> str:
    """SHA-256 preserving meaningful punctuation, e.g. C versus C++ versus C#."""
    return hashlib.sha256(normalize_text(text).encode('utf-8')).hexdigest()


def cached_extraction(kind: str, text: str, cache: MutableMapping):
    if kind not in {'jd', 'resume'}:
        raise ValueError('Unknown document extraction kind.')
    digest = extraction_hash(text)
    identity = extraction_identity()
    key = (*identity, kind, digest)
    persistent = ExtractionCache(identity=identity)
    if key not in cache:
        stored = persistent.load(kind, digest)
        if stored is not None:
            cache[key] = stored
    if identity[0] == 'sarvam' and key in cache:
        from services.extraction_completeness import completeness_problem
        if completeness_problem(cache[key], text):
            # Do not feed old sparse memory/disk entries to matching. Leave the
            # disk entry untouched unless a complete replacement is obtained.
            cache.pop(key)
    hit = key in cache
    logging.getLogger(__name__).warning('%s extraction cache: %s %s',
        identity[0].capitalize(), 'JD' if kind == 'jd' else 'Resume', 'HIT' if hit else 'MISS')
    if not hit:
        result = extract_jd_structure(text) if kind == 'jd' else extract_resume_structure(text)
        result = SCHEMAS[kind].model_validate(result.model_dump())
        if identity[0] == 'sarvam':
            from services.extraction_completeness import completeness_problem
            from services.extraction import ExtractionError
            if completeness_problem(result, text):
                raise ExtractionError('AI extraction omitted substantial document information. Please retry extraction.')
        persistent.save(kind, digest, result)
        cache[key] = result
    return cache[key]


def analyze_pair(jd_text: str, resume_text: str, *, cache: MutableMapping | None = None,
                 as_of: date | None = None, semantic: bool = SEMANTIC_MATCHING,
                 weights: dict[str, float] | None = None) -> dict:
    reject_identical_documents(jd_text, resume_text)
    cache = {} if cache is None else cache
    jd = cached_extraction('jd', jd_text, cache)
    resume = cached_extraction('resume', resume_text, cache)
    as_of = as_of or date.today()
    result = match_documents(jd, resume, resume_text, as_of=as_of)
    # The legacy semantic flag cannot enable cloud calls. All match evidence is local.
    result = refine_semantically(result, jd, resume)
    return {'candidate': resume.candidate_name.value if resume.candidate_name else 'Unnamed candidate',
            'jd': jd, 'resume': resume, 'match': result, 'score': score_match(result, weights),
            'feedback': generate_feedback(result), 'as_of': as_of.isoformat()}
