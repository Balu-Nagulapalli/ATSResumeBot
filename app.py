"""Run with: python -m streamlit run app.py"""
import hashlib
import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import streamlit as st
from config import MAX_RESUMES, SEMANTIC_MATCHING, get_api_key, ConfigurationError
from parsers.document_parser import DocumentError, check_text, extract_text, document_hash, validate_jd_document
from services.extraction import ExtractionError
from services.pipeline import analyze_pair, cached_extraction, extraction_hash
from services.extraction_cache import extraction_identity
from services.sarvam_cache_recovery import clear_bad_session_entries


def show_details(result: dict) -> None:
    score, match = result['score'], result['match']
    st.metric('ATS Match Score', f'{score.overall:.1f} / 100')
    st.caption(f'{score.classification} · Evidence confidence: {score.confidence:.0%}')
    columns = st.columns(3)
    for i, category in enumerate(['required_skills', 'preferred_skills', 'experience', 'role_relevance', 'responsibilities', 'education']):
        item = match.criteria[category]
        columns[i % 3].metric(category.replace('_', ' ').title(), f'{item.score:.0f}%' if item.applicable else 'N/A')
    evidence_tab, feedback_tab, roadmap_tab = st.tabs(['Scores & evidence', 'Improvements', 'Learning roadmap'])
    with evidence_tab:
        st.dataframe(pd.DataFrame([{'Criterion': name.replace('_', ' ').title(),
            'Score': round(item.score, 1) if item.applicable else None,
            'Confidence': item.confidence} for name, item in match.criteria.items()]), hide_index=True, width='stretch')
        for name, item in match.criteria.items():
            if not item.applicable:
                continue
            with st.expander(f'{name.replace("_", " ").title()} — {item.score:.1f}%'):
                st.write(item.explanation)
                if name == 'experience':
                    st.write(f'Candidate years: {match.candidate_years if match.candidate_years is not None else "Unknown"}')
                for entry in item.matches:
                    status = 'Matched' if entry.credit == 1 else 'Partial' if entry.credit else 'Not evidenced'
                    st.write(f'{entry.requirement} — {status} ({entry.credit:.0%})')
                    for quote in entry.evidence:
                        st.text(quote)
        warnings = [warning for warning in match.warnings
                    if not warning.startswith('Semantic relations are model judgments;')]
        if warnings:
            st.info('\n\n'.join(dict.fromkeys(warnings)))
    with feedback_tab:
        for key, title in [('strengths', 'Strengths'), ('missing_required_skills', 'Missing required skills'),
            ('missing_preferred_skills', 'Missing preferred skills'), ('tool_gaps', 'Tool gaps'),
            ('experience_gaps', 'Experience gaps'), ('education_gaps', 'Education gaps'),
            ('certification_gaps', 'Certification gaps'), ('responsibility_gaps', 'Responsibility gaps'),
            ('improvements', 'Resume improvements')]:
            items = result['feedback'][key]
            if items:
                st.subheader(title)
                for item in items:
                    st.write('• ' + item)
    with roadmap_tab:
        roadmap = result['feedback']['learning_roadmap']
        if not roadmap:
            st.success('No skill or tool gaps were identified in the supplied evidence.')
        for step in roadmap:
            st.subheader(f'{step["topic"]} · {step["priority"]}')
            st.write(step['why'])
            for i, topic in enumerate(step['learn'], 1):
                st.write(f'{i}. {topic}')


def show_results(results: list[dict]) -> None:
    if not results:
        return
    results = sorted(results, key=lambda r: -r['score'].overall)
    st.header('Results')
    if len(results) == 1:
        show_details(results[0])
        return
    rows = []
    for rank, result in enumerate(results, 1):
        criteria = result['match'].criteria
        rows.append({'Rank': rank, 'Candidate': result['candidate'], 'File': result['filename'],
            'ATS Score': result['score'].overall,
            'Required Skills Match': criteria['required_skills'].score if criteria['required_skills'].applicable else None,
            'Experience Match': criteria['experience'].score if criteria['experience'].applicable else None,
            'Role Relevance': criteria['role_relevance'].score if criteria['role_relevance'].applicable else None,
            'Missing Required Skills': ', '.join(criteria['required_skills'].missing_requirements),
            'Match Classification': result['score'].classification})
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, width='stretch')
    chart_frame = frame.assign(Label=frame['Rank'].astype(str) + '. ' + frame['Candidate'])
    chart = px.bar(chart_frame, x='ATS Score', y='Label', orientation='h', range_x=[0, 100],
                   text='ATS Score', color_discrete_sequence=['#2563eb'])
    chart.update_layout(yaxis={'autorange': 'reversed', 'title': None}, xaxis_title='ATS match score',
                        height=max(300, 55 * len(results)), margin=dict(l=0, r=20, t=20, b=0))
    st.plotly_chart(chart, width='stretch')
    for tab, result in zip(st.tabs([f'{i}. {r["candidate"]}' for i, r in enumerate(results, 1)]), results):
        with tab:
            show_details(result)


def main() -> None:
    st.set_page_config(page_title='ATS Resume Analyzer', page_icon='📄', layout='wide')
    st.title('ATS Resume Analyzer')
    st.write('AI-powered resume matching against job requirements.')
    if 'generation' not in st.session_state:
        st.session_state.generation = 0
    generation = st.session_state.generation
    st.session_state.setdefault('extraction_cache', {})
    st.session_state.setdefault('analysis_cache', {})
    clear_bad_session_entries(st.session_state)
    # Config imports are cached across reruns; recheck a locally added .env here.
    # Preserve process-environment precedence and never render the secret.
    load_dotenv(Path(__file__).resolve().parent / '.env', override=False)
    try:
        identity = extraction_identity()
        has_key = bool(get_api_key())
    except ConfigurationError as exc:
        st.error(str(exc))
        st.stop()
    if not has_key:
        st.warning('AI extraction is currently unavailable. Check local AI configuration.')
    st.header('1. Job Description')
    mode = st.radio('Provide the job description', ['Upload', 'Paste'], horizontal=True, key=f'mode{generation}')
    jd_file, pasted = None, ''
    if mode == 'Upload':
        jd_file = st.file_uploader('Job description · PDF, DOCX, TXT', type=['pdf', 'docx', 'txt'], key=f'jd{generation}')
    else:
        pasted = st.text_area('Paste the job description', height=220, max_chars=80_000, key=f'paste{generation}')
    st.header('2. Resume Upload')
    st.write('Upload up to 10 resumes')
    files = st.file_uploader('Select 1–10 resumes · PDF or DOCX · 10 MB per file', type=['pdf', 'docx'],
                             accept_multiple_files=True, key=f'resumes{generation}')
    semantic = SEMANTIC_MATCHING
    as_of = date.today()
    # Fingerprint uploads without parsing documents or invoking extraction.
    signature = hashlib.sha256(repr(('local-matching-v1', identity, mode, pasted,
        (jd_file.name, hashlib.sha256(jd_file.getvalue()).hexdigest()) if jd_file else None,
        [(f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files], semantic, str(as_of))).encode()).hexdigest()
    if st.session_state.get('current_document_signature') != signature:
        # Discard displayed state so restoring earlier documents cannot restore old results.
        # Keep extraction and analysis caches available for the next explicit Analyze.
        st.session_state['results'] = []
        st.session_state['errors'] = []
        st.session_state.pop('result_signature', None)
        st.session_state['current_document_signature'] = signature
    st.header('3. Analyze')
    if st.button('🚀 Analyze Candidates', type='primary', disabled=not has_key):
        st.session_state['results'] = []
        st.session_state['errors'] = []
        st.session_state.pop('result_signature', None)
        if not 1 <= len(files) <= MAX_RESUMES:
            st.error('Upload between 1 and 10 resumes.')
        else:
            try:
                jd_text = check_text(pasted) if mode == 'Paste' else extract_text(jd_file) if jd_file else ''
                validation = validate_jd_document(jd_text)
                if not validation.valid:
                    raise DocumentError('Provide a valid job description. ' + ' '.join(validation.reasons))
                with st.spinner('Extracting job requirements…'):
                    cached_extraction('jd', jd_text, st.session_state.extraction_cache)
                progress = st.progress(0.0)
                for i, file in enumerate(files):
                    try:
                        text = extract_text(file, resume=True)
                        cache_key = ('local-matching-v1', extraction_hash(jd_text), extraction_hash(text), identity, semantic, str(as_of))
                        with st.spinner(f'Analyzing resume {i + 1} of {len(files)}…'):
                            if cache_key not in st.session_state.analysis_cache:
                                if len(st.session_state.analysis_cache) >= 30:
                                    st.session_state.analysis_cache.pop(next(iter(st.session_state.analysis_cache)))
                                st.session_state.analysis_cache[cache_key] = analyze_pair(jd_text, text,
                                    cache=st.session_state.extraction_cache, semantic=semantic, as_of=as_of)
                            result = dict(st.session_state.analysis_cache[cache_key])
                            result['filename'] = file.name.replace('\\', '/').split('/')[-1]
                            st.session_state.results.append(result)
                    except (DocumentError, ExtractionError) as exc:
                        st.session_state.errors.append(f'Resume {i + 1}: {exc}')
                    except Exception:
                        st.session_state.errors.append(f'Resume {i + 1}: Analysis failed. Check the document and retry.')
                    progress.progress((i + 1) / len(files))
            except (DocumentError, ExtractionError) as exc:
                st.session_state.errors.append(str(exc))
            except Exception:
                st.session_state.errors.append('Analysis could not start. Check your configuration and try again.')
        if st.session_state.results:
            st.session_state['result_signature'] = signature
    # Failed attempts still show their errors, without marking documents as analyzed.
    for error in st.session_state.get('errors', []):
        st.error(error)
    if st.session_state.get('result_signature') == signature:
        show_results(st.session_state.get('results', []))
    with st.expander('Start over'):
        if st.button('Clear documents and results'):
            generation = st.session_state.generation + 1
            st.session_state.clear()
            st.session_state.generation = generation
            st.rerun()


if __name__ == '__main__':
    main()
