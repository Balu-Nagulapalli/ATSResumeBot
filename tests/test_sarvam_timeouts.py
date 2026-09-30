"""Offline timeout retries and persistent-cache behavior with the real SDK."""
import json
from unittest.mock import Mock, call

import httpx
import pytest

from models.schemas import JobDescriptionData, ResumeData
from services import extraction, extraction_cache, pipeline, sarvam
from tests.test_sarvam import response


@pytest.mark.parametrize('schema,seconds', [(JobDescriptionData, 60), (ResumeData, 90)])
def test_sdk_uses_bounded_request_timeout(schema, seconds):
    def handler(request):
        assert request.extensions['timeout']['read'] == seconds
        assert all(value is not None for value in request.extensions['timeout'].values())
        return response()
    with httpx.Client(transport=httpx.MockTransport(handler), timeout=1) as client:
        assert sarvam.structured_request(schema, 'extract', 'synthetic', client=client) == schema()


@pytest.mark.parametrize('timeouts', [1, 2, 3])
def test_read_timeout_retries_are_bounded(monkeypatch, caplog, timeouts):
    requests = []
    sleep = Mock()
    monkeypatch.setattr(sarvam.time, 'sleep', sleep)
    def handler(request):
        requests.append(request)
        if len(requests) <= timeouts:
            raise httpx.ReadTimeout('PrivateCandidate private@example.test secret-key', request=request)
        return response()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        if timeouts == 3:
            with pytest.raises(extraction.ExtractionError, match='timed out') as exc:
                sarvam.structured_request(ResumeData, 'extract', 'synthetic', client=client)
            assert 'PrivateCandidate' not in str(exc.value)
        else:
            assert sarvam.structured_request(ResumeData, 'extract', 'synthetic', client=client) == ResumeData()
        assert not client.event_hooks['response']
    assert len(requests) == min(timeouts + 1, 3)
    assert sleep.call_args_list == [call(2), call(4)][:min(timeouts, 2)]
    assert all(request.content == requests[0].content for request in requests)
    assert 'exception_type=ReadTimeout' in caplog.text
    for private in ('PrivateCandidate', 'private@example.test', 'secret-key'):
        assert private not in caplog.text


def test_validation_failure_after_timeout_stops_without_retry(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(sarvam.time, 'sleep', sleep)
    handler = Mock(side_effect=[httpx.ReadTimeout('synthetic timeout'), response('{"skills":false}')])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(extraction.ExtractionError, match='schema validation'):
            sarvam.structured_request(ResumeData, 'extract', 'synthetic', client=client)
    assert handler.call_count == 2
    sleep.assert_called_once_with(2)


def test_cached_jd_reused_and_failed_resume_not_cached(monkeypatch, caplog):
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    monkeypatch.setattr(sarvam.time, 'sleep', lambda _: None)
    jd_text = 'Software Engineer. Required skills: Python. Responsibilities: build software. ' * 3
    resume_text = 'Synthetic Candidate. Skills: Python. Experience: developer. Education: bachelor degree. ' * 3
    store = extraction_cache.ExtractionCache()
    digest = pipeline.extraction_hash(jd_text)
    store.save('jd', digest, JobDescriptionData())
    cached_jd = store.path('jd', digest).read_bytes()
    jd_request = Mock(side_effect=AssertionError('Cached JD must never be extracted again'))
    monkeypatch.setattr(pipeline, 'extract_jd_structure', jd_request)
    requests = []
    def handler(request):
        requests.append(request)
        assert json.loads(request.content)['response_format']['json_schema']['name'] == 'ResumeData'
        raise httpx.ReadTimeout('synthetic timeout', request=request)
    constructor = httpx.Client
    monkeypatch.setattr(sarvam.httpx, 'Client', lambda **kwargs:
                        constructor(transport=httpx.MockTransport(handler), **kwargs))
    memory = {}
    with pytest.raises(extraction.ExtractionError, match='timed out'):
        pipeline.analyze_pair(jd_text, resume_text, cache=memory)
    assert len(requests) == 3
    jd_request.assert_not_called()
    assert 'JD HIT' in caplog.text
    assert 'Resume MISS' in caplog.text
    assert len(memory) == 1
    assert all(isinstance(value, JobDescriptionData) for value in memory.values())
    assert store.load('resume', pipeline.extraction_hash(resume_text)) is None
    assert store.path('jd', digest).read_bytes() == cached_jd
    assert list(extraction_cache.CACHE_ROOT.rglob('*.json')) == [store.path('jd', digest)]
