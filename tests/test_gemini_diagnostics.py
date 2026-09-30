"""Diagnostics must not leak provider-echoed documents or credentials."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from google.genai import errors
from models.schemas import JobDescriptionData, ResumeData
from services.extraction import ExtractionError, structured_request


@pytest.mark.parametrize('schema,stage,override', [
    (JobDescriptionData, 'JD extraction', None),
    (ResumeData, 'Resume extraction', None),
    (ResumeData, 'Semantic matching', 'Semantic matching'),
])
def test_api_diagnostics_are_private(caplog, schema, stage, override):
    client = Mock()
    response = httpx.Response(400, headers={'retry-after': '7'})
    client.models.generate_content.side_effect = errors.ClientError(400, {'error': {
        'message': 'Invalid JSON payload received. Unknown name: candidate David PrivateName '
                   'private@example.test fake-secret-value JD confidential resume confidential'}}, response=response)
    with pytest.raises(ExtractionError, match='Check model configuration and retry'):
        structured_request(schema, 'unchanged prompt', 'confidential document', client=client, stage=override)
    record = json.loads(caplog.records[-1].getMessage().split('Gemini diagnostic: ', 1)[1])
    assert record['stage'] == stage
    assert record['exception_type'] == 'ClientError'
    assert record['http_status'] == 400
    assert record['retry_after'] == '7'
    assert 'invalid json payload received' in record['gemini_error_message']
    for private in ['David', 'PrivateName', 'private@example.test', 'fake-secret-value', 'confidential']:
        assert private not in caplog.text


def test_parsing_diagnostics_omit_model_output(caplog):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason='STOP')],
        text='{"total_experience_years": "PrivateCandidate confidential-output"}')
    with pytest.raises(ExtractionError, match='schema validation'):
        structured_request(ResumeData, 'extract', 'confidential input', client=client)
    assert len(caplog.records) == 1  # Schema failures must not be retried.
    for item in caplog.records:
        record = json.loads(item.getMessage().split('Gemini diagnostic: ', 1)[1])
        assert record['stage'] == 'Structured response parsing'
        assert record['exception_type'] == 'ValidationError'
    assert 'PrivateCandidate' not in caplog.text
    assert 'confidential' not in caplog.text


def test_transport_diagnostics_omit_urls_and_untrusted_retry_header(caplog, monkeypatch):
    monkeypatch.setattr('services.extraction.time.sleep', lambda _: None)
    client = Mock()
    client.models.generate_content.side_effect = httpx.ReadTimeout(
        'https://example.test?key=fake-secret PrivateCandidate')
    with pytest.raises(ExtractionError, match='timed out'):
        structured_request(ResumeData, 'extract', 'private input', client=client)
    assert len(caplog.records) == 1
    assert 'fake-secret' not in caplog.text
    assert 'PrivateCandidate' not in caplog.text
    caplog.clear()
    client.models.generate_content.side_effect = errors.ClientError(400, {'error': {'message': 'private data'}},
        response=httpx.Response(400, headers={'retry-after': 'PrivateCandidate'}))
    with pytest.raises(ExtractionError):
        structured_request(ResumeData, 'extract', 'private input', client=client)
    assert 'PrivateCandidate' not in caplog.text
    assert '"retry_after": null' in caplog.text
