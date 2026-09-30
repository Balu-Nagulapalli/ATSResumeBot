"""Offline Streamlit rendering tests. No real documents or API calls."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_missing_key_ui_and_clear(monkeypatch):
    # A blank process value wins over local .env, isolating this test from real credentials.
    monkeypatch.setenv('GEMINI_API_KEY', '')
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == 'ATS Resume Analyzer'
    assert any('AI extraction is currently unavailable' in warning.value for warning in app.warning)
    assert next(b for b in app.button if b.label == '🚀 Analyze Candidates').disabled
    app.radio[0].set_value('Paste').run()
    app.text_area[0].set_value('A draft job description').run()
    next(b for b in app.button if b.label == 'Clear documents and results').click().run()
    assert not app.exception
    assert app.radio[0].value == 'Upload'


def test_comparison_and_evidence_rendering():
    script = '''
from app import show_results
from tests.test_core import match, jd_fixture, resume_fixture
from services.scorer import score_match
from services.feedback import generate_feedback
result = match()
first = dict(candidate='Alex Example', filename='first.pdf', score=score_match(result),
             match=result, jd=jd_fixture(), resume=resume_fixture(),
             feedback=generate_feedback(result), as_of='2026-01-01')
second = dict(first, candidate='Second Candidate', filename='second.pdf')
show_results([first, second])
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert len(app.dataframe) >= 3
    assert len(app.get('plotly_chart')) == 1
    assert any(m.label == 'ATS Match Score' for m in app.metric)
    assert any('Git' in item.value for item in app.markdown)


def test_upload_workflow_isolates_failure_and_reuses_results(monkeypatch):
    from io import BytesIO
    from unittest.mock import Mock
    import pymupdf
    import streamlit
    from services import pipeline
    from tests.test_core import jd_fixture, resume_fixture, resume_text
    monkeypatch.setenv('GEMINI_API_KEY', 'test-placeholder')
    jd_file = BytesIO(b'Senior Developer\nRequired skills: Python\nResponsibilities: Develop applications\nPreferred: Docker\nExperience: 5+ years professional development.')
    jd_file.name = 'job.txt'
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), resume_text())
        good = BytesIO(pdf.tobytes())
    good.name = 'candidate.pdf'
    bad = BytesIO(b'corrupt PDF')
    bad.name = 'bad.pdf'
    uploads = {'jd': jd_file, 'resumes': [good, bad]}
    monkeypatch.setattr(streamlit, 'file_uploader', lambda label, **kwargs:
                        uploads['resumes'] if kwargs.get('accept_multiple_files') else uploads['jd'])
    jd_extract = Mock(return_value=jd_fixture())
    resume_extract = Mock(return_value=resume_fixture())
    monkeypatch.setattr(pipeline, 'extract_jd_structure', jd_extract)
    monkeypatch.setattr(pipeline, 'extract_resume_structure', resume_extract)
    monkeypatch.setattr(pipeline, 'refine_semantically', lambda result, jd, resume: result)
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'), default_timeout=30).run()
    next(b for b in app.button if b.label == '🚀 Analyze Candidates').click().run()
    assert not app.exception
    assert any('Resume 2' in error.value for error in app.error)
    assert any(metric.label == 'ATS Match Score' for metric in app.metric)
    assert jd_extract.call_count == resume_extract.call_count == 1
    next(b for b in app.button if b.label == '🚀 Analyze Candidates').click().run()
    assert jd_extract.call_count == resume_extract.call_count == 1
    app.run()
    assert app.metric  # unchanged reruns retain the current analysis

    changed_jd = BytesIO(jd_file.getvalue() + b'\nAdditional requirements: SQL')
    changed_jd.name = jd_file.name  # same filename, different content
    replacement = BytesIO(b'replacement PDF')
    replacement.name = good.name
    renamed_jd = BytesIO(jd_file.getvalue())
    renamed_jd.name = 'replacement.txt'
    for field, changed in [
        ('jd', changed_jd), ('jd', renamed_jd), ('jd', None),
        ('resumes', [good]), ('resumes', [good, bad, replacement]),
        ('resumes', [replacement, bad]), ('resumes', []),
    ]:
        original = uploads[field]
        uploads[field] = changed
        app.run()
        assert not app.exception
        assert not app.metric
        assert not app.error
        assert not app.session_state['results']
        assert 'result_signature' not in app.session_state
        uploads[field] = original
        app.run()
        assert not app.metric  # restoring documents must not resurrect results
        assert jd_extract.call_count == resume_extract.call_count == 1
        next(b for b in app.button if b.label == '🚀 Analyze Candidates').click().run()
        assert app.metric
        assert jd_extract.call_count == resume_extract.call_count == 1

    app.radio[0].set_value('Paste').run()
    assert not app.metric  # previous scores must not be shown against changed inputs
    app.text_area[0].set_value(jd_file.getvalue().decode()).run()
    assert not app.metric
    next(b for b in app.button if b.label == '🚀 Analyze Candidates').click().run()
    assert app.metric
    app.text_area[0].set_value(changed_jd.getvalue().decode()).run()
    assert not app.metric
    app.text_area[0].set_value(jd_file.getvalue().decode()).run()
    assert not app.metric
    assert jd_extract.call_count == resume_extract.call_count == 1

    # A failed Analyze shows errors but does not establish an analyzed fingerprint.
    uploads['resumes'] = [bad]
    next(b for b in app.button if b.label == '🚀 Analyze Candidates').click().run()
    assert app.error
    assert not app.metric
    assert 'result_signature' not in app.session_state
    uploads['resumes'] = [good]
    app.run()
    assert not app.error
    next(b for b in app.button if b.label == '🚀 Analyze Candidates').click().run()
    assert app.metric

    # Start over clears session analysis state, leaving all persistent JSON intact.
    from services.extraction_cache import CACHE_ROOT
    cached = {path: path.read_bytes() for path in CACHE_ROOT.rglob('*.json')}
    assert cached
    next(b for b in app.button if b.label == 'Clear documents and results').click().run()
    assert not app.exception
    assert not app.metric
    assert not app.session_state['results']
    assert 'result_signature' not in app.session_state
    assert not app.session_state['analysis_cache']
    assert not app.session_state['extraction_cache']
    assert {path: path.read_bytes() for path in CACHE_ROOT.rglob('*.json')} == cached
    assert jd_extract.call_count == resume_extract.call_count == 1
