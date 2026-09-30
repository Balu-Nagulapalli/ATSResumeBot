from datetime import date
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
import json
import pytest
import pymupdf
from docx import Document
from pydantic import ValidationError
from config import WEIGHTS
from models.schemas import Fact, Years, Role, Education, JobDescriptionData, ResumeData
from parsers.document_parser import (DocumentError, extract_text, normalize_text,
    reject_identical_documents, validate_jd_document, validate_resume_document)
from services.normalizer import normalize_skill, term_in_quote
from services.experience import candidate_experience
from services.extraction import ExtractionError, validate_source_evidence, structured_request
from services.matcher import match_documents
from services.scorer import score_match
from services.feedback import generate_feedback
from services.benchmark import evaluate_benchmarks

AS_OF = date(2026, 1, 1)


def facts(*values):
    return [Fact(value=v, quote=v) for v in values]


def jd_fixture():
    return JobDescriptionData(job_title=facts('Senior Full-Stack Developer')[0],
        required_skills=facts('JavaScript', 'React', 'Node.js', 'Python', 'AWS', 'Docker', 'PostgreSQL', 'MongoDB', 'Git', 'REST APIs'),
        preferred_skills=facts('TypeScript', 'Kubernetes', 'GraphQL', 'Redis', 'Microservices Architecture'),
        minimum_experience_years=Years(value=5.0, quote='5+ years'),
        education_requirements=[Education(level='bachelor', field='Computer Science', related_field_allowed=True,
            quote="Bachelor's degree in Computer Science or related field")],
        responsibilities=facts('Build scalable web applications', 'Lead code reviews', 'Mentor junior developers',
                               'Improve performance and security', 'Implement CI/CD pipelines', 'Cross-functional collaboration'))


def resume_fixture():
    return ResumeData(candidate_name=facts('Alex Example')[0], headline=facts('Senior Full-Stack Developer')[0],
        total_experience_years=Years(value=7.0, quote='7 years experience'),
        skills=facts('JavaScript', 'Python', 'TypeScript', 'React', 'Node.js', 'GraphQL', 'PostgreSQL', 'MongoDB',
                     'Redis', 'AWS', 'Docker', 'Kubernetes', 'CI/CD', 'Microservices', 'REST APIs'),
        education=[Education(level='bachelor', field='Computer Science', related_field_allowed=False,
                             quote='Bachelor of Science in Computer Science')],
        responsibilities=facts('Led development team', 'Microservices redesign', 'Implemented Kubernetes CI/CD',
                               'Mentored junior developers', 'Built React applications with Node.js', 'Managed AWS infrastructure'),
        achievements=facts('Reduced API response time by 40%'),
        roles=[Role(title='Senior Full-Stack Developer', start_date='2019', end_date='Present',
                    quote='Senior Full-Stack Developer 2019 - Present')], contact_present=True)


def resume_text(resume=None):
    resume = resume or resume_fixture()
    return '\n'.join(['Alex Example', 'alex@example.test', 'Skills', 'Experience', 'Education',
        '7 years experience', 'Bachelor of Science in Computer Science'] +
        [f.quote for f in resume.skills + resume.responsibilities + resume.achievements] + [r.quote for r in resume.roles])


def match(resume=None, jd=None):
    resume = resume or resume_fixture()
    return match_documents(jd or jd_fixture(), resume, resume_text(resume), as_of=AS_OF)


def test_text_and_alias_normalization():
    assert normalize_text('Skills\r\n\u2022  Python\n\n\nEducation') == 'Skills\n- Python\n\nEducation'
    for a, b in [('JS', 'javascript'), ('React.js', 'react'), ('Postgres', 'postgresql'),
                 ('Amazon Web Services', 'aws'), ('CI CD', 'ci/cd'), ('RESTful API', 'rest api')]:
        assert normalize_skill(a) == b
    assert normalize_skill('Flask') != normalize_skill('FastAPI')
    assert not term_in_quote('Java', 'JavaScript')
    assert not term_in_quote('React', 'Reactive programming')


def test_identical_documents_rejected():
    with pytest.raises(DocumentError, match='identical'):
        reject_identical_documents('Job: Senior Developer\nPython', 'JOB Senior Developer  Python')


def test_document_validation_uses_multiple_signals():
    assert validate_resume_document(resume_text()).valid
    assert validate_jd_document('Senior Developer\nRequirements: Python\nResponsibilities: Build applications.\nPreferred: Docker\n5+ years of professional experience.').valid
    assert not validate_resume_document('Experience ' + 'An unrelated story about the ocean. ' * 12).valid
    assert not validate_jd_document('A grocery shopping list of vegetables and fruit. ' * 8).valid


@pytest.mark.parametrize('name,data,message', [
    ('bad.pdf', b'not a pdf', 'corrupted'), ('empty.txt', b'', 'empty'),
    ('short.txt', b'hello', 'too short'), ('bad.exe', b'hello', 'Unsupported')])
def test_bad_documents(name, data, message):
    with pytest.raises(DocumentError, match=message):
        extract_text(BytesIO(data), name)


def test_scanned_pdf():
    with pymupdf.open() as doc:
        doc.new_page()
        data = doc.tobytes()
    with pytest.raises(DocumentError, match='no extractable text'):
        extract_text(BytesIO(data), 'scan.pdf')


def test_real_pdf_and_docx_extraction():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), resume_text())
        assert 'Alex Example' in extract_text(BytesIO(pdf.tobytes()), 'resume.pdf', resume=True)
    doc = Document()
    doc.add_paragraph('Alex Example alex@example.test Experience Education Skills')
    table = doc.add_table(rows=1, cols=1)
    table.cell(0, 0).text = 'Python and React development, seven years of professional experience building applications.'
    stream = BytesIO()
    doc.save(stream)
    assert 'Python and React' in extract_text(stream, 'resume.docx', resume=True)


def test_text_encoding_and_resume_extension():
    data = ('Requirements: Python\nResponsibilities: Build applications\nPreferred: Docker\nAn engineer\u2019s role with 5+ years experience.').encode('cp1252')
    assert 'engineer' in extract_text(BytesIO(data), 'job.txt')
    with pytest.raises(DocumentError):
        extract_text(BytesIO(data), 'resume.txt', resume=True)


def test_required_preferred_and_tools():
    result = match()
    assert result.criteria['required_skills'].score == 90
    assert result.criteria['required_skills'].missing_requirements == ['Git']
    assert result.criteria['preferred_skills'].score == 100
    assert not result.criteria['tools'].applicable
    jd = JobDescriptionData(required_tools=facts('AWS'), preferred_tools=facts('Docker'))
    result = match(ResumeData(skills=facts('AWS')), jd)
    assert result.criteria['required_skills'].score == 100
    assert result.criteria['tools'].score == 0  # Only optional Docker remains here.


def test_related_technologies_are_missing():
    result = match(ResumeData(skills=facts('FastAPI', 'Azure', 'Angular', 'MySQL')),
                   JobDescriptionData(required_skills=facts('Flask', 'AWS', 'React', 'PostgreSQL')))
    assert result.criteria['required_skills'].score == 0


def test_experience_comparison_and_unknown():
    assert match().criteria['experience'].score == 100
    candidate = resume_fixture()
    candidate.total_experience_years = Years(value=2.5, quote='2.5 years')
    assert match(candidate).criteria['experience'].score == 50
    candidate.total_experience_years = None
    candidate.roles = []
    result = match(candidate)
    assert result.candidate_years is None
    assert result.criteria['experience'].score == 0
    assert any('could not be reliably determined' in w for w in result.warnings)


def test_overlap_and_present_ranges():
    candidate = ResumeData(roles=[Role(title='Developer', start_date=None, end_date=None, quote=q) for q in [
        'Developer Jan 2020 - Dec 2022', 'Developer Jan 2022 - Dec 2023', 'Developer Jan 2024 - Present']])
    assert candidate_experience(candidate, AS_OF)[0] == 6.0
    candidate.roles = [Role(title='Developer', start_date=None, end_date=None, quote='Developer 2017 - 2021')]
    assert candidate_experience(candidate, AS_OF)[0] == 4.0


def test_education_level_and_field():
    assert match().criteria['education'].score == 100
    candidate = resume_fixture()
    candidate.education[0].field = 'Computer Engineering'
    assert match(candidate).criteria['education'].score == 100
    candidate.education[0].field = 'Literature'
    assert match(candidate).criteria['education'].score == 50
    candidate.education = []
    assert match(candidate).criteria['education'].score == 0


def test_nonapplicable_and_weight_normalization():
    result = match()
    scored = score_match(result)
    assert 'certifications' not in scored.effective_weights
    assert sum(scored.effective_weights.values()) == pytest.approx(1.0)
    assert scored.overall == round(sum(scored.contributions.values()), 2)
    sparse = score_match(match(jd=JobDescriptionData(required_skills=facts('Python'))))
    assert sparse.effective_weights['resume_quality'] <= .05
    assert score_match(match(jd=JobDescriptionData())).classification == 'Insufficient JD criteria'


def test_high_medium_unrelated_and_determinism():
    high = score_match(match()).overall
    medium_resume = ResumeData(headline=facts('Developer')[0], skills=facts('Python', 'JavaScript', 'React', 'AWS', 'Docker'),
                              total_experience_years=Years(value=3.0, quote='3 years'))
    medium = score_match(match(medium_resume)).overall
    unrelated = score_match(match(ResumeData(headline=facts('Pastry Chef')[0], skills=facts('Baking')))).overall
    assert 80 < high <= 100
    assert 25 < medium < high
    assert 0 <= unrelated < 20
    assert score_match(match()).model_dump() == score_match(match()).model_dump()


def test_score_rejects_bad_weights_and_nonfinite_schema():
    with pytest.raises(ValueError):
        score_match(match(), {**WEIGHTS, 'required_skills': 3.0})
    with pytest.raises(ValidationError):
        Years(value=float('nan'), quote='NaN years')
    with pytest.raises(ValidationError):
        ResumeData(total_experience_years='seven')
    with pytest.raises(ValidationError):
        ResumeData(invented=True)


def test_evidence_must_exist_and_support_technology():
    with pytest.raises(ExtractionError):
        validate_source_evidence(ResumeData(skills=[Fact(value='AWS', quote='AWS production')]), 'Azure production')
    with pytest.raises(ExtractionError):
        validate_source_evidence(ResumeData(skills=[Fact(value='AWS', quote='Azure production')]), 'Azure production')
    validate_source_evidence(ResumeData(skills=[Fact(value='AWS', quote='Amazon Web Services')]), 'Amazon Web Services')


def test_benchmark_labels_never_reach_predictor():
    received = []
    def predictor(jd, resume):
        received.append((jd, resume))
        return 90.0
    table, mae = evaluate_benchmarks([{'jd': 'job', 'resume': 'candidate', 'expected_score': 95.0},
                                     {'jd': 'job', 'resume': 'other', 'expected_score': 70.0}], predictor)
    assert received == [('job', 'candidate'), ('job', 'other')]
    assert mae == 12.5
    assert list(table['Absolute Error']) == [5, 20]
    with pytest.raises(ValueError):
        evaluate_benchmarks([], predictor)


def test_feedback_is_gap_based():
    feedback = generate_feedback(match())
    assert feedback['missing_required_skills'] == ['Git']
    assert feedback['learning_roadmap'][0]['topic'] == 'Git'
    assert any('If you have experience with Git' in s for s in feedback['improvements'])


def test_missing_key_safe_error(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    with pytest.raises(ExtractionError, match='Configure GEMINI_API_KEY'):
        structured_request(ResumeData, 'extract', 'document')


def test_sdk_request_contract_without_live_call():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(candidates=[SimpleNamespace(finish_reason='STOP')], text=ResumeData().model_dump_json())
    result = structured_request(ResumeData, 'extract', 'document', client=client)
    assert isinstance(result, ResumeData)
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs['config'].response_json_schema == ResumeData.model_json_schema()
    assert kwargs['config'].response_mime_type == 'application/json'
    assert client.models.generate_content.call_count == 1
    client.models.generate_content.return_value = SimpleNamespace(candidates=[], text=None)
    with pytest.raises(ExtractionError, match='declined'):
        structured_request(ResumeData, 'extract', 'document', client=client)


def test_semantic_match_cannot_substitute_distinct_technologies():
    from services.semantic import Decisions, Decision, refine_semantically
    jd = JobDescriptionData(required_skills=facts('AWS'))
    candidate = ResumeData(skills=facts('Azure'))
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(candidates=[SimpleNamespace(finish_reason='STOP')], text=Decisions(
        decisions=[Decision(requirement_id=0, candidate_id=0, relation='equivalent')]).model_dump_json())
    result = refine_semantically(match(candidate, jd), jd, candidate, client=client)
    assert result.criteria['required_skills'].score == 0
    client.models.generate_content.assert_not_called()


def test_extraction_cache_reuses_jd(monkeypatch):
    from services import pipeline
    jd_call, resume_call = Mock(return_value=jd_fixture()), Mock(return_value=resume_fixture())
    monkeypatch.setattr(pipeline, 'extract_jd_structure', jd_call)
    monkeypatch.setattr(pipeline, 'extract_resume_structure', resume_call)
    cache = {}
    pipeline.analyze_pair('job', 'resume A', cache=cache, semantic=False, as_of=AS_OF)
    pipeline.analyze_pair('job', 'resume B', cache=cache, semantic=False, as_of=AS_OF)
    assert jd_call.call_count == 1
    assert resume_call.call_count == 2


def test_required_tool_takes_precedence_over_duplicate_preferred_skill():
    jd = JobDescriptionData(required_tools=facts('AWS'), preferred_skills=facts('Amazon Web Services'))
    result = match(ResumeData(skills=facts('AWS')), jd)
    assert result.criteria['required_skills'].score == 100
    assert not result.criteria['tools'].applicable
    assert not result.criteria['preferred_skills'].applicable


def test_dates_are_not_contact_information():
    from parsers.document_parser import contact_evidence
    assert not contact_evidence('Developer 2019 - 2021')
    assert contact_evidence('Call +1 202 555 0199')


def test_preferred_tool_has_optional_learning_priority():
    result = match(ResumeData(), JobDescriptionData(preferred_tools=facts('Docker')))
    assert generate_feedback(result)['learning_roadmap'][0]['priority'] == 'Optional'


@pytest.mark.parametrize('span,years', [('Jan 2020 - Mar 2024', 4.25), ('2019 - Present', 7.0),
                                     ('2020-01 - 2024-03', 4.25), ('01/2020 - 03/2024', 4.25)])
def test_date_formats(span, years):
    candidate = ResumeData(roles=[Role(title='Developer', start_date=None, end_date=None, quote=span)])
    assert candidate_experience(candidate, AS_OF)[0] == years
