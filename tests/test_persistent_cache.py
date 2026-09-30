"""Persistent-cache validation, privacy and failure recovery."""
import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from models.schemas import Fact, JobDescriptionData, ResumeData
from services import extraction_cache, pipeline
from services.extraction import ExtractionError
from services.extraction_cache import ExtractionCache


@pytest.mark.parametrize('kind,schema,extractor', [
    ('jd', JobDescriptionData, 'extract_jd_structure'),
    ('resume', ResumeData, 'extract_resume_structure'),
])
def test_restart_reuses_valid_extraction(monkeypatch, caplog, kind, schema, extractor):
    result = schema()
    call = Mock(return_value=result)
    monkeypatch.setattr(pipeline, extractor, call)
    document = 'Private source document for cache test'
    first = pipeline.cached_extraction(kind, document, {})
    # Both the service instance and the caller's in-memory cache are fresh.
    fresh = ExtractionCache()
    assert fresh.load(kind, pipeline.extraction_hash(document)) == result
    second = pipeline.cached_extraction(kind, document, {})
    assert first == second == result
    assert call.call_count == 1
    label = 'JD' if kind == 'jd' else 'Resume'
    assert [r.getMessage() for r in caplog.records] == [
        f'Gemini extraction cache: {label} MISS', f'Gemini extraction cache: {label} HIT']


@pytest.mark.parametrize('content', ['{broken', '{"unexpected":true}',
    '{"total_experience_years":"seven"}', 'null', '\ufffd'])
def test_corrupt_or_incompatible_cache_falls_back(monkeypatch, content):
    store = ExtractionCache()
    digest = pipeline.extraction_hash('resume')
    path = store.path('resume', digest)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding='utf-8')
    call = Mock(return_value=ResumeData())
    monkeypatch.setattr(pipeline, 'extract_resume_structure', call)
    assert pipeline.cached_extraction('resume', 'resume', {}) == ResumeData()
    call.assert_called_once()
    assert store.load('resume', digest) == ResumeData()


def test_failed_api_response_is_never_saved(monkeypatch):
    monkeypatch.setattr(pipeline, 'extract_jd_structure', Mock(side_effect=ExtractionError('Unavailable')))
    memory = {}
    with pytest.raises(ExtractionError):
        pipeline.cached_extraction('jd', 'private job', memory)
    assert not memory
    assert not list(extraction_cache.CACHE_ROOT.rglob('*.json'))


def test_only_structured_fields_are_saved(monkeypatch):
    result = ResumeData(headline=Fact(value='Engineer', quote='Engineer'))
    monkeypatch.setattr(pipeline, 'extract_resume_structure', Mock(return_value=result))
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-cache-test-secret')
    raw = 'Engineer\nRAW_ONLY_MARKER omitted from the structured extraction\nMore document text'
    pipeline.cached_extraction('resume', raw, {})
    files = list(extraction_cache.CACHE_ROOT.rglob('*.json'))
    assert len(files) == 1
    assert files[0].parent.name == 'resumes'
    assert files[0].stem == pipeline.extraction_hash(raw)
    serialized = files[0].read_text(encoding='utf-8')
    assert json.loads(serialized) == result.model_dump()
    assert 'RAW_ONLY_MARKER' not in serialized
    assert 'fake-cache-test-secret' not in serialized
    assert 'expected_score' not in serialized
    assert raw not in serialized


def test_invalid_fresh_result_is_not_cached(monkeypatch):
    result = Mock()
    result.model_dump.return_value = {'expected_score': 95}
    monkeypatch.setattr(pipeline, 'extract_jd_structure', Mock(return_value=result))
    with pytest.raises(ValidationError):
        pipeline.cached_extraction('jd', 'job', {})
    assert not list(extraction_cache.CACHE_ROOT.rglob('*.json'))


def test_memory_hit_does_not_read_disk(monkeypatch):
    monkeypatch.setattr(pipeline, 'extract_jd_structure', Mock(return_value=JobDescriptionData()))
    memory = {}
    pipeline.cached_extraction('jd', 'job', memory)
    read = Mock(side_effect=AssertionError('Memory must be checked first'))
    monkeypatch.setattr(ExtractionCache, 'load', read)
    pipeline.cached_extraction('jd', 'job', memory)
    read.assert_not_called()


def test_disk_write_failure_keeps_successful_memory_result(monkeypatch):
    monkeypatch.setattr(pipeline, 'extract_jd_structure', Mock(return_value=JobDescriptionData()))
    monkeypatch.setattr(extraction_cache.os, 'replace', Mock(side_effect=OSError('Disk unavailable')))
    memory = {}
    assert pipeline.cached_extraction('jd', 'job', memory) == JobDescriptionData()
    assert memory
    assert not list(extraction_cache.CACHE_ROOT.rglob('*.json'))
