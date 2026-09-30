"""Offline regression tests for the shared Gemini transient retry policy."""
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest
from google import genai
from google.genai import errors, types
from models.schemas import JobDescriptionData, ResumeData
from services.extraction import ExtractionError, structured_request, _retry_delay, extract
from services.semantic import Decisions
from parsers.document_parser import DocumentError


def api_error(code=503, retry_after=None):
    headers = {} if retry_after is None else {'Retry-After': retry_after}
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {'error': {'message': 'Service unavailable'}},
               response=httpx.Response(code, headers=headers))


@pytest.mark.parametrize('schema,stage,result', [
    (JobDescriptionData, 'JD extraction', JobDescriptionData()),
    (ResumeData, 'Resume extraction', ResumeData()),
    (Decisions, 'Semantic matching', Decisions(decisions=[])),
])
def test_all_stages_recover_on_third_attempt(monkeypatch, caplog, schema, stage, result):
    sleep = Mock()
    monkeypatch.setattr('services.extraction.time.sleep', sleep)
    client = Mock()
    client.models.generate_content.side_effect = [api_error(), api_error(),
        SimpleNamespace(candidates=[SimpleNamespace(finish_reason='STOP')], text=result.model_dump_json())]
    assert structured_request(schema, 'extract', 'synthetic text', client=client, stage=stage) == result
    assert client.models.generate_content.call_count == 3
    assert sleep.call_args_list == [call(2.0), call(4.0)]
    assert len(caplog.records) == 2
    assert all(stage in entry.getMessage() for entry in caplog.records)
    requests = client.models.generate_content.call_args_list
    assert requests[0] == requests[1] == requests[2]


def test_exhaustion_preserves_friendly_error_and_diagnostics(monkeypatch, caplog):
    sleep = Mock()
    monkeypatch.setattr('services.extraction.time.sleep', sleep)
    client = Mock()
    client.models.generate_content.side_effect = api_error()
    with pytest.raises(ExtractionError, match='Gemini could not process the request. Check model configuration and retry.'):
        structured_request(ResumeData, 'extract', 'synthetic text', client=client)
    assert client.models.generate_content.call_count == 3
    assert sleep.call_args_list == [call(2.0), call(4.0)]
    assert len(caplog.records) == 3


@pytest.mark.parametrize('header,expected', [('7', 7), ('2.5', 2.5), ('0', 2),
    ('60', 60), ('61', None), ('86400', None), ('-1', 2), ('nan', 2), ('invalid', 2)])
def test_retry_after_seconds_are_bounded(header, expected):
    assert _retry_delay(api_error(retry_after=header), 0) == expected


def test_retry_after_http_date(monkeypatch):
    now = datetime(2026, 9, 27, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr('services.extraction.time.time', lambda: now)
    for offset, expected in [(10, 10), (-10, 2), (61, None)]:
        header = format_datetime(datetime.fromtimestamp(now + offset, timezone.utc), usegmt=True)
        assert _retry_delay(api_error(retry_after=header), 0) == expected


def test_retry_after_controls_actual_wait(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr('services.extraction.time.sleep', sleep)
    client = Mock()
    client.models.generate_content.side_effect = [api_error(429, '7'), api_error(503, '1'),
        SimpleNamespace(candidates=[SimpleNamespace(finish_reason='STOP')], text='{}')]
    structured_request(ResumeData, 'extract', 'synthetic text', client=client)
    assert sleep.call_args_list == [call(7.0), call(4.0)]


def test_validation_failure_after_transient_error_stops_immediately(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr('services.extraction.time.sleep', sleep)
    client = Mock()
    client.models.generate_content.side_effect = [api_error(),
        SimpleNamespace(candidates=[SimpleNamespace(finish_reason='STOP')], text='not JSON')]
    with pytest.raises(ExtractionError, match='malformed JSON'):
        structured_request(ResumeData, 'extract', 'synthetic text', client=client)
    assert client.models.generate_content.call_count == 2
    sleep.assert_called_once_with(2.0)


def test_malformed_document_never_reaches_api():
    client = Mock()
    with pytest.raises(DocumentError):
        extract(ResumeData, '', client=client)
    client.models.generate_content.assert_not_called()


def test_sdk_does_not_multiply_application_retries(monkeypatch):
    requests = []
    monkeypatch.setattr('services.extraction.time.sleep', lambda _: None)
    def handler(request):
        requests.append(request)
        return httpx.Response(503, json={'error': {'code': 503, 'message': 'Service unavailable', 'status': 'UNAVAILABLE'}})
    with genai.Client(api_key='test-placeholder', vertexai=False, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))) as client:
        with pytest.raises(ExtractionError):
            structured_request(ResumeData, 'extract', 'synthetic text', client=client)
    assert len(requests) == 3
