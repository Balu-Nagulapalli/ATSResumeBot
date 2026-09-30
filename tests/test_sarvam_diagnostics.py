"""Reproduce the import failure and verify that diagnostics never echo documents."""
import builtins
from unittest.mock import Mock

import httpx
import pytest

from models.schemas import JobDescriptionData
from services import extraction, sarvam
from tests.test_sarvam import response


def test_missing_sdk_logged_before_generic_ui_handler(monkeypatch, caplog):
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    original = builtins.__import__
    def missing_sdk(name, *args, **kwargs):
        if name == 'services.sarvam':
            raise ModuleNotFoundError('PrivateCandidate secret-key', name='sarvamai')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing_sdk)
    with pytest.raises(ModuleNotFoundError):
        extraction.provider_structured_request(JobDescriptionData, 'extract', 'private document')
    assert '[SARVAM ERROR]' in caplog.text
    assert 'stage=JD extraction' in caplog.text
    assert 'exception_type=ModuleNotFoundError' in caplog.text
    assert 'http_status=null' in caplog.text
    assert 'SDK is not installed' in caplog.text
    assert 'PrivateCandidate' not in caplog.text
    assert 'secret-key' not in caplog.text
    assert 'private document' not in caplog.text


@pytest.mark.parametrize('result,stage,detail', [
    (lambda: response('PrivateCandidate secret-key'), 'JD structured parsing', 'json_parse_error='),
    (lambda: response('{"required_skills":"PrivateCandidate", "secret-key":true}'),
     'JD schema validation', 'Input should be a valid list.'),
    (lambda: httpx.Response(200, text='PrivateCandidate secret-key'),
     'JD structured parsing', 'json_parse_error='),
    (lambda: httpx.Response(200, json={'choices': 'PrivateCandidate secret-key'}),
     'JD structured parsing', 'json_parse_error='),
])
def test_successful_http_parsing_and_validation_errors_are_safe(caplog, result, stage, detail):
    handler = Mock(side_effect=lambda _: result())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(extraction.ExtractionError):
            sarvam.structured_request(JobDescriptionData, 'extract', 'private document', client=client)
        assert not client.event_hooks['response']
    assert handler.call_count == 1
    assert f'stage={stage}' in caplog.text
    assert 'http_status=200' in caplog.text
    assert detail in caplog.text
    assert 'PrivateCandidate' not in caplog.text
    assert 'secret-key' not in caplog.text
    assert 'private document' not in caplog.text


def test_provider_error_code_and_safe_excerpt(caplog):
    def handler(_):
        return httpx.Response(400, json={'error': {'code': 'invalid_request_error',
            'message': 'Invalid JSON schema: unsupported keyword anyOf. PrivateCandidate secret-key'}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(extraction.ExtractionError):
            sarvam.structured_request(JobDescriptionData, 'extract', 'private document', client=client)
    assert 'provider_error_code=invalid_request_error' in caplog.text
    assert 'unsupported keyword' in caplog.text
    assert 'http_status=400' in caplog.text
    assert 'PrivateCandidate' not in caplog.text
    assert 'secret-key' not in caplog.text
