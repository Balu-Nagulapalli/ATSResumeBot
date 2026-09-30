"""Session-scoped call budgets and local-only matching, without live API usage."""
from unittest.mock import Mock
import pytest
import httpx
from google.genai import errors
from services import pipeline, extraction
from services.semantic import refine_semantically
from tests.test_core import jd_fixture, resume_fixture, match, AS_OF


@pytest.fixture
def counters(monkeypatch):
    jd = Mock(return_value=jd_fixture())
    resume = Mock(return_value=resume_fixture())
    monkeypatch.setattr(pipeline, 'extract_jd_structure', jd)
    monkeypatch.setattr(pipeline, 'extract_resume_structure', resume)
    # Any unexpected cloud request during matching fails this test immediately.
    forbidden = Mock(side_effect=AssertionError('Matching must remain local'))
    monkeypatch.setattr(extraction.genai, 'Client', forbidden)
    return jd, resume, forbidden


def test_same_jd_and_resume_are_extracted_once(counters, caplog):
    jd, resume, forbidden = counters
    cache = {}
    first = pipeline.analyze_pair('private job', 'private candidate', cache=cache, as_of=AS_OF)
    second = pipeline.analyze_pair('private job', 'private candidate', cache=cache, as_of=AS_OF)
    assert jd.call_count == resume.call_count == 1
    assert first['score'] == second['score']
    forbidden.assert_not_called()
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ['Gemini extraction cache: JD MISS', 'Gemini extraction cache: Resume MISS',
                        'Gemini extraction cache: JD HIT', 'Gemini extraction cache: Resume HIT']
    assert 'private' not in caplog.text
    assert pipeline.extraction_hash('private candidate') not in caplog.text


def test_one_jd_three_unique_resumes_costs_four_extractions(counters):
    jd, resume, forbidden = counters
    cache = {}
    for document in ['resume A', 'resume B', 'resume C', 'resume A']:
        pipeline.analyze_pair('job', document, cache=cache, as_of=AS_OF)
    assert jd.call_count == 1
    assert resume.call_count == 3
    forbidden.assert_not_called()


def test_resume_is_reused_across_different_jobs(counters):
    jd, resume, _ = counters
    cache = {}
    for job in ['job A', 'job B']:
        pipeline.analyze_pair(job, 'same resume', cache=cache, as_of=AS_OF)
    assert jd.call_count == 2
    assert resume.call_count == 1


def test_persistent_cache_is_reused_by_new_memory_caches(counters):
    jd, _, _ = counters
    pipeline.cached_extraction('jd', 'job', {})
    pipeline.cached_extraction('jd', 'job', {})
    assert jd.call_count == 1


def test_cache_preserves_punctuation_and_document_kind(counters):
    jd, resume, _ = counters
    cache = {}
    for text in ['C', 'C++', 'C#']:
        pipeline.cached_extraction('jd', text, cache)
    pipeline.cached_extraction('resume', 'C', cache)
    assert jd.call_count == 3
    assert resume.call_count == 1


def test_successful_extractions_are_not_evicted_by_new_documents(counters):
    jd, _, _ = counters
    cache = {}
    for i in range(35):
        pipeline.cached_extraction('jd', f'job {i}', cache)
    pipeline.cached_extraction('jd', 'job 0', cache)
    assert jd.call_count == 35


def test_failed_extractions_are_not_cached(counters):
    jd, _, _ = counters
    jd.side_effect = [extraction.ExtractionError('Unavailable'), jd_fixture()]
    cache = {}
    with pytest.raises(extraction.ExtractionError):
        pipeline.cached_extraction('jd', 'job', cache)
    assert not cache
    pipeline.cached_extraction('jd', 'job', cache)
    assert jd.call_count == 2


def test_local_refinement_never_uses_injected_client():
    client = Mock()
    result = match()
    refined = refine_semantically(result, jd_fixture(), resume_fixture(), client=client)
    assert refined == result
    assert not client.mock_calls


@pytest.mark.parametrize('message,details', [
    ('Daily quota exceeded', []),
    ('Quota exceeded', [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]),
    ('You exceeded your current quota', []),
    ('Quota exhausted', []),
])
def test_exhausted_quota_is_not_retried(monkeypatch, message, details):
    client = Mock()
    client.models.generate_content.side_effect = errors.ClientError(429, {'error': {
        'message': message, 'details': details}}, response=httpx.Response(429))
    sleep = Mock()
    monkeypatch.setattr(extraction.time, 'sleep', sleep)
    with pytest.raises(extraction.ExtractionError, match='quota or rate limit'):
        extraction.structured_request(type(jd_fixture()), 'extract', 'synthetic text', client=client)
    assert client.models.generate_content.call_count == 1
    sleep.assert_not_called()


def test_minute_rate_limit_still_respects_retry_after():
    exc = errors.ClientError(429, {'error': {'message': 'Quota exceeded per minute',
        'details': [{'quotaId': 'RequestsPerMinute'}]}}, response=httpx.Response(429, headers={'Retry-After': '12'}))
    assert extraction._retry_delay(exc, 0) == 12
