"""Exercise the real SDK with an in-memory HTTP transport."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import httpx
import pytest
from google import genai
from google.genai import errors, types
from models.schemas import ResumeData, Years
from services.extraction import ExtractionError, structured_request, validate_source_evidence


def test_real_sdk_serialization_and_parsing():
    captured = []
    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json={'candidates': [
            {'finishReason': 'STOP', 'content': {'role': 'model',
             'parts': [{'text': ResumeData().model_dump_json()}]}}]})
    with genai.Client(api_key='test-placeholder', vertexai=False, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))) as client:
        result = structured_request(ResumeData, 'extract', 'synthetic document', client=client)
    assert result == ResumeData()
    assert captured[0]['generationConfig']['responseMimeType'] == 'application/json'
    assert captured[0]['generationConfig']['responseJsonSchema'] == ResumeData.model_json_schema()
    assert captured[0]['systemInstruction']['parts'][0]['text'] == 'extract'


def test_timeout_without_http_status_is_not_retried(monkeypatch):
    monkeypatch.setattr('services.extraction.time.sleep', lambda _: None)
    client = Mock()
    client.models.generate_content.side_effect = httpx.ReadTimeout('private transport error')
    with pytest.raises(ExtractionError, match='timed out'):
        structured_request(ResumeData, 'extract', 'document', client=client)
    assert client.models.generate_content.call_count == 1


def test_auth_error_is_sanitized_without_retry():
    client = Mock()
    client.models.generate_content.side_effect = errors.ClientError(400, {'error': {
        'code': 400, 'message': 'API key not valid. private provider error', 'status': 'INVALID_ARGUMENT',
        'details': [{'reason': 'API_KEY_INVALID'}]}})
    with pytest.raises(ExtractionError, match='rejected the configured API key') as error:
        structured_request(ResumeData, 'extract', 'document', client=client)
    assert 'private provider' not in str(error.value)
    assert client.models.generate_content.call_count == 1


def test_experience_citation_requires_duration_not_unrelated_number():
    data = ResumeData(total_experience_years=Years(value=7.0, quote='7 projects in 3 years'))
    with pytest.raises(ExtractionError):
        validate_source_evidence(data, '7 projects in 3 years')
    data = ResumeData(total_experience_years=Years(value=7.0, quote='7 Years Experience'))
    validate_source_evidence(data, '7 Years Experience')


@pytest.mark.parametrize('code,phrase,calls', [
    (401, 'rejected the configured API key', 1), (403, 'access was denied', 1),
    (400, 'could not process', 1), (404, 'model was not found', 1),
    (429, 'quota or rate limit', 3), (500, 'could not process', 3),
    (502, 'could not process', 3), (503, 'could not process', 3),
    (504, 'timed out', 3), (501, 'could not process', 1)])
def test_provider_errors_are_safe_and_retries_bounded(monkeypatch, code, phrase, calls):
    monkeypatch.setattr('services.extraction.time.sleep', lambda _: None)
    client = Mock()
    error_type = errors.ClientError if code < 500 else errors.ServerError
    client.models.generate_content.side_effect = error_type(code, {'error': {
        'code': code, 'message': 'private provider details'}})
    with pytest.raises(ExtractionError, match=phrase) as error:
        structured_request(ResumeData, 'extract', 'document', client=client)
    assert 'private provider' not in str(error.value)
    assert client.models.generate_content.call_count == calls


@pytest.mark.parametrize('text,phrase', [('not JSON', 'malformed JSON'),
    ('{"total_experience_years": "seven"}', 'schema validation'),
    ('{"unknown_field": true}', 'schema validation')])
def test_invalid_responses_are_validated_without_retry(text, phrase):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason='STOP')], text=text)
    with pytest.raises(ExtractionError, match=phrase):
        structured_request(ResumeData, 'extract', 'document', client=client)
    assert client.models.generate_content.call_count == 1


@pytest.mark.parametrize('reason', ['MAX_TOKENS', 'SAFETY'])
def test_incomplete_responses_never_become_empty_resumes(reason):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=reason)], text='{}')
    with pytest.raises(ExtractionError, match='incomplete response'):
        structured_request(ResumeData, 'extract', 'document', client=client)


def test_owned_client_configuration_and_cleanup(monkeypatch):
    from services import extraction
    monkeypatch.setenv('GEMINI_API_KEY', 'test-placeholder')
    monkeypatch.setattr(extraction, 'MODEL', 'gemini-test')
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason='STOP')], text='{}')
    constructor = Mock(return_value=client)
    monkeypatch.setattr(extraction.genai, 'Client', constructor)
    assert structured_request(ResumeData, 'extract', 'document') == ResumeData()
    options = constructor.call_args.kwargs['http_options']
    assert options.timeout == 60000
    assert options.retry_options.attempts == 1
    assert constructor.call_args.kwargs['vertexai'] is False
    client.close.assert_called_once()
