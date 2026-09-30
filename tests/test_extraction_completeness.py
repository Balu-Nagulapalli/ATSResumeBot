"""Source-aware completeness, corrective retries, and targeted cache recovery."""
import json
from unittest.mock import Mock, call

import httpx
import pytest

from models.schemas import Fact, JobDescriptionData, ResumeData
from services import extraction, extraction_cache, pipeline, sarvam
from services.extraction_completeness import completeness_problem, CORRECTIVE_INSTRUCTION
from services.sarvam_cache_recovery import clear_bad_session_entries, BAD_JD, BAD_RESUME
from tests.test_sarvam import response

JD = '''Senior Developer job opening
Requirements:
Required skills: Python and SQL
Preferred tools: Docker
Experience: 5 years developing applications
Education: Bachelor degree in Computer Science
Responsibilities: Build reliable applications and review code.
We are looking for a developer to maintain our internal services and collaborate with colleagues.
The role includes documenting changes, investigating reported problems and testing releases.
'''
RESUME = '''PrivateCandidate
Skills: Python and SQL
Experience: Developer 2020 - Present
Responsibilities: Build reliable applications and review code.
Education: Bachelor degree in Computer Science
Projects: Build a reporting application
Achievements: Improved release testing
Professional summary: Developer with experience maintaining internal services and collaborating with colleagues.
Work includes documenting changes, investigating reported problems and testing releases.
'''


def complete(schema):
    field = 'required_skills' if schema is JobDescriptionData else 'skills'
    return schema(**{field: [Fact(value='Python', quote='Python'), Fact(value='SQL', quote='SQL')]})


@pytest.mark.parametrize('data,text', [
    (JobDescriptionData(), JD),
    (JobDescriptionData(job_title=Fact(value='Senior Developer', quote='Senior Developer'),
                        responsibilities=[Fact(value='Build reliable applications', quote='Build reliable applications')]), JD),
    (ResumeData(), RESUME),
    (ResumeData(candidate_name=Fact(value='PrivateCandidate', quote='PrivateCandidate'), contact_present=True), RESUME),
])
def test_substantial_source_rejects_sparse_models(data, text):
    assert completeness_problem(data, text)


@pytest.mark.parametrize('data,text', [
    (JobDescriptionData(responsibilities=[Fact(value='Greet visitors', quote='Greet visitors')]),
     'Receptionist opening. Duties: Greet visitors. Requirements: friendly manner. Join our team for a part-time position.'),
    (ResumeData(education=[] , skills=[Fact(value='Customer service', quote='Customer service')]),
     'Professional summary: Customer service volunteer seeking a first position. Skills: Customer service and greeting visitors.'),
    (complete(JobDescriptionData), JD), (complete(ResumeData), RESUME),
])
def test_supported_simple_and_substantial_documents_pass(data, text):
    assert completeness_problem(data, text) is None


@pytest.mark.parametrize('kind,schema,source', [('jd', JobDescriptionData, JD), ('resume', ResumeData, RESUME)])
@pytest.mark.parametrize('recover', [True, False])
def test_completeness_retry_and_cache(monkeypatch, caplog, kind, schema, source, recover):
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        assert not list(extraction_cache.CACHE_ROOT.rglob('*.json'))
        result = complete(schema) if recover and len(captured) == 2 else schema()
        return response(result.model_dump_json())
    constructor = httpx.Client
    monkeypatch.setattr(sarvam.httpx, 'Client', lambda **kwargs:
                        constructor(transport=httpx.MockTransport(handler), **kwargs))
    memory = {}
    if recover:
        assert pipeline.cached_extraction(kind, source, memory) == complete(schema)
        assert extraction_cache.ExtractionCache().load(kind, pipeline.extraction_hash(source)) == complete(schema)
        assert pipeline.cached_extraction(kind, source, {}) == complete(schema)
    else:
        with pytest.raises(extraction.ExtractionError, match='omitted substantial document information') as exc:
            pipeline.cached_extraction(kind, source, memory)
        assert not memory
        assert not list(extraction_cache.CACHE_ROOT.rglob('*.json'))
        assert 'Sarvam' not in str(exc.value)
    assert len(captured) == 2
    assert CORRECTIVE_INSTRUCTION not in captured[0]['messages'][0]['content']
    assert CORRECTIVE_INSTRUCTION in captured[1]['messages'][0]['content']
    assert captured[0]['messages'][1] == captured[1]['messages'][1]
    assert captured[0]['response_format'] == captured[1]['response_format']
    assert 'Sarvam completeness retry:' in caplog.text
    assert 'PrivateCandidate' not in caplog.text
    assert source not in caplog.text


def test_combined_transport_and_completeness_budget_is_six(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    sleep = Mock()
    monkeypatch.setattr(sarvam.time, 'sleep', sleep)
    handler = Mock(side_effect=[httpx.ReadTimeout('timeout'), httpx.ReadTimeout('timeout'), response('{}'),
                               httpx.ReadTimeout('timeout'), httpx.ReadTimeout('timeout'),
                               response(complete(ResumeData).model_dump_json())])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert extraction.extract(ResumeData, RESUME, client=client) == complete(ResumeData)
    assert handler.call_count == 6
    assert sleep.call_args_list == [call(2), call(4), call(2), call(4)]


def test_stale_sparse_cache_hit_never_reaches_matcher(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'sarvam')
    store = extraction_cache.ExtractionCache()
    store.save('jd', pipeline.extraction_hash(JD), JobDescriptionData())
    failing = Mock(side_effect=extraction.ExtractionError('omitted substantial document information'))
    monkeypatch.setattr(pipeline, 'extract_jd_structure', failing)
    matcher = Mock(side_effect=AssertionError('Sparse extraction must not reach matching'))
    monkeypatch.setattr(pipeline, 'match_documents', matcher)
    with pytest.raises(extraction.ExtractionError):
        pipeline.analyze_pair(JD, RESUME)
    failing.assert_called_once()
    matcher.assert_not_called()


def test_gemini_keeps_original_prompt_and_behavior(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'gemini')
    request = Mock(return_value=complete(JobDescriptionData))
    monkeypatch.setattr(extraction, 'structured_request', request)
    assert extraction.extract(JobDescriptionData, JD) == complete(JobDescriptionData)
    assert request.call_args.args[1] == extraction.SYSTEM_PROMPT
    request.assert_called_once()


def test_targeted_session_cleanup_preserves_other_providers_and_valid_results():
    version = '0fdfeadcd24886460d6d9e030607045a5f638fefd0d8bef6f54794cc97bcc236'
    identity = ('sarvam', 'sarvam-105b', version)
    gemini = ('gemini', 'other-model', version)
    bad = ResumeData()
    good = complete(ResumeData)
    jd = complete(JobDescriptionData)
    bad_key = ('local-matching-v1', BAD_JD, BAD_RESUME, identity, True, 'date')
    good_key = ('local-matching-v1', BAD_JD, 'other-resume', identity, True, 'date')
    gemini_key = ('local-matching-v1', BAD_JD, BAD_RESUME, gemini, True, 'date')
    bad_result = {'jd': jd, 'resume': bad}
    good_result = {'jd': jd, 'resume': good}
    gemini_result = {'jd': JobDescriptionData(), 'resume': ResumeData()}
    state = {'extraction_cache': {(*identity, 'resume', BAD_RESUME): bad,
                                 (*identity, 'jd', BAD_JD): jd,
                                 (*gemini, 'resume', BAD_RESUME): ResumeData()},
             'analysis_cache': {bad_key: bad_result, good_key: good_result, gemini_key: gemini_result},
             'results': [dict(bad_result), dict(good_result), dict(gemini_result)]}
    clear_bad_session_entries(state)
    assert len(state['extraction_cache']) == 2
    assert set(state['analysis_cache']) == {good_key, gemini_key}
    assert state['results'] == [good_result, gemini_result]
    # A repaired extraction for the same document survives subsequent reruns.
    state['extraction_cache'][(*identity, 'resume', BAD_RESUME)] = good
    clear_bad_session_entries(state)
    assert len(state['extraction_cache']) == 3
