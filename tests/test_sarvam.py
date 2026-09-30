"""Offline provider, HTTP contract, privacy, and cache-isolation regressions."""
import json
from unittest.mock import Mock, call

import httpx
import pytest

import config
from models.schemas import Fact, JobDescriptionData, ResumeData
from services import extraction, extraction_cache, pipeline, sarvam
from services.benchmark import evaluate_benchmarks
from tests.test_core import AS_OF, jd_fixture, resume_fixture, resume_text


def response(content='{}', finish='stop'):
    return httpx.Response(200, json={'id': 'offline-response', 'created': 0,
        'model': 'sarvam-105b', 'object': 'chat.completion', 'choices': [
        {'index': 0, 'finish_reason': finish, 'message': {'role': 'assistant', 'content': content}}]})


@pytest.mark.parametrize('provider', ['gemini', 'sarvam'])
def test_selected_provider_only(monkeypatch, provider):
    monkeypatch.setenv('AI_PROVIDER', provider)
    other = 'gemini' if provider == 'sarvam' else 'sarvam'
    monkeypatch.delenv(f'{other.upper()}_API_KEY', raising=False)
    assert config.get_api_key() == 'offline-test-key'
    gemini = Mock(return_value=ResumeData())
    sarvam_call = Mock(return_value=ResumeData())
    monkeypatch.setattr(extraction, 'structured_request', gemini)
    monkeypatch.setattr(sarvam, 'structured_request', sarvam_call)
    assert extraction.provider_structured_request(ResumeData, 'extract', 'text') == ResumeData()
    assert gemini.call_count == (provider == 'gemini')
    assert sarvam_call.call_count == (provider == 'sarvam')


def test_unsupported_provider_rejected_before_cache_or_network(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'secret-invalid-name')
    for operation in [config.get_provider, lambda: pipeline.cached_extraction('jd', 'text', {}),
                      lambda: extraction.provider_structured_request(ResumeData, '', '')]:
        with pytest.raises(config.ConfigurationError, match='Unsupported AI_PROVIDER') as exc:
            operation()
        assert 'secret-invalid-name' not in str(exc.value)


@pytest.mark.parametrize('schema,field', [(JobDescriptionData, 'required_skills'), (ResumeData, 'skills')])
def test_http_contract_and_existing_schemas(monkeypatch, schema, field):
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    monkeypatch.setenv('SARVAM_MODEL', 'sarvam-105b')
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setenv('SARVAM_API_KEY', '  offline-test-key \n')
    expected = schema(**{field: [Fact(value='Python', quote='Python')]})
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        assert str(request.url) == sarvam.ENDPOINT
        assert request.method == 'POST'
        assert request.headers['api-subscription-key'] == 'offline-test-key'
        return response(expected.model_dump_json())
    source = 'Python development experience and project responsibilities. ' * 3
    original_schema = schema.model_json_schema()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = extraction.extract(schema, source, client=client)
    assert result == expected
    assert isinstance(result, schema)
    body = captured[0]
    assert body['model'] == 'sarvam-105b'
    from services.extraction_completeness import sarvam_instructions
    assert body['messages'][0]['content'] == extraction.SYSTEM_PROMPT + '\n\n' + sarvam_instructions(schema)
    assert body['messages'][1]['content'] == source.strip()
    assert body['response_format']['json_schema']['strict'] is True
    wire_schema = body['response_format']['json_schema']['schema']
    assert wire_schema == sarvam.strict_json_schema(schema)
    assert set(wire_schema['required']) == set(original_schema['properties'])
    assert schema.model_json_schema() == original_schema
    assert 'score' not in json.dumps(body['response_format'])
    assert 'Never score' in body['messages'][0]['content']


@pytest.mark.parametrize('status,attempts', [(429, 3), (500, 3), (502, 3), (503, 3), (504, 3),
                                          (400, 1), (401, 1), (403, 1), (422, 1)])
def test_retries_and_private_diagnostics(monkeypatch, caplog, status, attempts):
    requests = []
    sleep = Mock()
    monkeypatch.setattr(sarvam.time, 'sleep', sleep)
    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers={'retry-after': '3'}, json={
            'error': 'PrivateName private@example.test +123456789 offline-test-key full-document'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(extraction.ExtractionError) as exc:
            sarvam.structured_request(ResumeData, 'extract', 'full-document', client=client)
    assert len(requests) == attempts
    assert sleep.call_args_list == ([call(3.0), call(4.0)] if attempts == 3 else [])
    for private in ['PrivateName', 'private@example.test', '+123456789', 'offline-test-key', 'full-document']:
        assert private not in caplog.text + str(exc.value)
    assert 'provider=sarvam' in caplog.text
    assert 'Resume extraction' in caplog.text


@pytest.mark.parametrize('content,finish', [('not-json', 'stop'), ('{"unexpected":true}', 'stop'),
                                          ('{}', 'length'), ('', 'stop')])
def test_bad_responses_never_retry_or_cache(monkeypatch, content, finish):
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    handler = Mock(side_effect=lambda _: response(content, finish))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(pipeline, 'extract_resume_structure', lambda text:
                            extraction.extract(ResumeData, text, client=client))
        with pytest.raises(extraction.ExtractionError):
            pipeline.cached_extraction('resume', 'Synthetic resume with sufficient text. ' * 4, {})
    assert handler.call_count == 1
    assert not list(extraction_cache.CACHE_ROOT.rglob('*.json'))


def test_unsupported_evidence_rejected(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    data = ResumeData(skills=[Fact(value='Java', quote='Java')])
    with httpx.Client(transport=httpx.MockTransport(lambda _: response(data.model_dump_json()))) as client:
        with pytest.raises(extraction.ExtractionError, match='evidence absent'):
            extraction.extract(ResumeData, 'Python developer experience and projects. ' * 4, client=client)


@pytest.mark.parametrize('kind,schema,extractor', [('jd', JobDescriptionData, 'extract_jd_structure'),
                                               ('resume', ResumeData, 'extract_resume_structure')])
def test_provider_model_version_cache_isolation(monkeypatch, caplog, kind, schema, extractor):
    source = 'private document'
    digest = pipeline.extraction_hash(source)
    legacy = extraction_cache.CACHE_ROOT / ('jd' if kind == 'jd' else 'resumes') / f'{digest}.json'
    legacy.parent.mkdir(parents=True)
    legacy.write_text(schema().model_dump_json(), encoding='utf-8')
    legacy_bytes = legacy.read_bytes()
    extract = Mock(return_value=schema())
    monkeypatch.setattr(pipeline, extractor, extract)
    memory = {}
    monkeypatch.setenv('AI_PROVIDER', 'gemini')
    pipeline.cached_extraction(kind, source, memory)
    gemini_path = extraction_cache.ExtractionCache().path(kind, digest)
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    pipeline.cached_extraction(kind, source, memory)
    sarvam_path = extraction_cache.ExtractionCache().path(kind, digest)
    assert gemini_path != sarvam_path
    assert extract.call_count == 2
    pipeline.cached_extraction(kind, source, memory)
    pipeline.cached_extraction(kind, source, {})  # restart disk hit
    assert extract.call_count == 2
    label = 'JD' if kind == 'jd' else 'Resume'
    assert f'Sarvam extraction cache: {label} MISS' in caplog.text
    assert f'Sarvam extraction cache: {label} HIT' in caplog.text
    monkeypatch.setenv('SARVAM_MODEL', 'changed-model')
    pipeline.cached_extraction(kind, source, memory)
    monkeypatch.setattr(extraction_cache, 'EXTRACTION_VERSION', 'next-schema-version')
    pipeline.cached_extraction(kind, source, memory)
    assert extract.call_count == 4
    assert len(memory) == 4
    assert legacy.read_bytes() == legacy_bytes
    assert gemini_path.exists() and sarvam_path.exists()


def test_matching_scoring_make_zero_provider_calls(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    monkeypatch.setattr(pipeline, 'extract_jd_structure', Mock(return_value=jd_fixture()))
    monkeypatch.setattr(pipeline, 'extract_resume_structure', Mock(return_value=resume_fixture()))
    forbidden = Mock(side_effect=AssertionError('Matching/scoring must remain local'))
    monkeypatch.setattr(extraction.genai, 'Client', forbidden)
    monkeypatch.setattr(sarvam.httpx, 'Client', forbidden)
    pipeline.analyze_pair('synthetic job', resume_text(), as_of=AS_OF)
    forbidden.assert_not_called()


def test_benchmark_labels_never_reach_sarvam(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        return response()
    jd = 'Software Engineer job description. Requirements: Python. Responsibilities: build software. ' * 3
    resume = 'Synthetic Candidate. Skills: Python. Experience: software engineer. Education: bachelor degree. ' * 3
    constructor = httpx.Client
    monkeypatch.setattr(sarvam.httpx, 'Client', lambda **kwargs:
                        constructor(transport=httpx.MockTransport(handler), **kwargs))
    evaluate_benchmarks([{'jd': jd, 'resume': resume, 'expected_score': 87.654321}], as_of=AS_OF)
    assert len(captured) == 2
    assert [body['messages'][1]['content'] for body in captured] == [jd.strip(), resume.strip()]
    assert '87.654321' not in json.dumps(captured)
    assert 'expected_score' not in json.dumps(captured)


def test_schema_definition_change_invalidates_identity(monkeypatch):
    old = extraction_cache.extraction_identity()
    class FutureResume(ResumeData):
        added: str | None = None
    monkeypatch.setitem(extraction_cache.SCHEMAS, 'resume', FutureResume)
    assert extraction_cache.extraction_identity() != old


@pytest.mark.parametrize('provider', ['gemini', 'sarvam'])
def test_application_extraction_errors_are_provider_neutral(monkeypatch, provider):
    monkeypatch.setenv('AI_PROVIDER', provider)
    monkeypatch.setenv(f'{provider.upper()}_API_KEY', '')
    with pytest.raises(extraction.ExtractionError) as exc:
        extraction.provider_structured_request(ResumeData, 'extract', 'text')
    assert 'gemini' not in str(exc.value).lower()
    assert 'sarvam' not in str(exc.value).lower()


def test_sarvam_ui_ready_without_gemini_key(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    monkeypatch.setenv('GEMINI_API_KEY', '')
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert not any('AI extraction is currently unavailable' in item.value for item in app.warning)
    assert not next(button for button in app.button if 'Analyze Candidates' in button.label).disabled
