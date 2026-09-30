"""Synthetic, evidence-backed local matching regressions; no benchmark labels or APIs."""
from datetime import date
from models.schemas import Fact, Years, Role, Education, JobDescriptionData, ResumeData
from services.matcher import match_documents
from services.local_relevance import responsibility_matches, role_similarity
from services.scorer import score_match

TODAY = date(2026, 1, 1)


def facts(*values):
    return [Fact(value=value, quote=value) for value in values]


def job():
    return JobDescriptionData(job_title=facts('Senior Full-Stack Developer')[0],
        required_skills=facts('JavaScript', 'React', 'Python', 'PostgreSQL', 'AWS', 'Docker'),
        required_tools=facts('Git'), preferred_skills=facts('TypeScript'),
        minimum_experience_years=Years(value=5.0, quote='5+ years professional software development'),
        responsibilities=facts('Implement CI/CD pipelines', 'Optimize application performance and security',
            'Lead code reviews and mentor junior developers', 'Build web applications'),
        education_requirements=[Education(level='bachelor', field='Computer Science',
            related_field_allowed=True, quote='Bachelor in Computer Science or related field')],
        domain=facts('Education')[0])


def candidate(title, skills, years, start, duties, field='Computer Science', domain='Education'):
    return ResumeData(headline=facts(title)[0], candidate_name=facts('Synthetic Candidate')[0],
        skills=facts(*skills), total_experience_years=Years(value=float(years), quote=f'{years} years experience'),
        roles=[Role(title=title, start_date=str(start), end_date='2026', quote=f'{title} {start} - 2026')],
        responsibilities=facts(*duties), domains=facts(domain),
        achievements=facts('Delivered 12 successful projects'), contact_present=True,
        education=[Education(level='bachelor', field=field, related_field_allowed=False, quote=f'Bachelor in {field}')])


def evaluate(resume, jd=None):
    text = 'Synthetic Candidate\ncandidate@example.test\nSkills\nExperience\nEducation\n' + '\n'.join(
        f.quote for f in resume.skills + resume.responsibilities + resume.achievements)
    result = match_documents(jd or job(), resume, text, as_of=TODAY)
    return result, score_match(result)


def test_generic_candidate_strengths_are_ordered():
    strong = candidate('Senior Full-Stack Developer',
        ['JS', 'React.js', 'Python', 'Postgres', 'Amazon Web Services', 'Docker', 'Git', 'TypeScript'], 7, 2019,
        ['Implemented Kubernetes-based CI/CD pipeline', 'Reduced API response time by 40% through optimization',
         'Improved application security', 'Led code reviews', 'Mentored 3 junior developers', 'Built web applications'])
    partial = candidate('Frontend Developer', ['JavaScript', 'React', 'AWS', 'Git'], 3, 2023,
        ['Built web applications with React', 'Reduced application load time', 'Built CI/CD pipelines'])
    junior = candidate('Junior Robotics Engineer', ['Python', 'C++'], 1, 2025,
        ['Implemented robot motion controllers'], field='Mechanical Engineering', domain='Manufacturing')
    unrelated = candidate('Business Analyst', ['Excel', 'Tableau'], 8, 2018,
        ['Managed financial reporting', 'Analyzed quarterly budgets'], field='Business', domain='Banking')
    high_match, high = evaluate(strong)
    _, medium = evaluate(partial)
    _, low = evaluate(junior)
    unrelated_match, near_zero = evaluate(unrelated)
    assert high_match.criteria['responsibilities'].score >= 90
    assert high_match.criteria['experience'].score == 100
    assert high.overall >= 85
    assert 35 <= medium.overall < 80
    assert 0 < low.overall < 35
    assert near_zero.overall <= 5
    assert unrelated_match.criteria['resume_quality'].score >= 85
    assert unrelated_match.criteria['experience'].score == 0
    assert high.overall > medium.overall > low.overall > near_zero.overall


def test_pipeline_and_performance_equivalences():
    for requirement, evidence in [
        ('Implement CI/CD pipelines', 'Implemented Kubernetes-based CI/CD pipeline'),
        ('Optimize application performance', 'Reduced API response time by 40%'),
        ('Design and implement APIs', 'Designed and implemented APIs'),
    ]:
        result = responsibility_matches(facts(requirement), facts(evidence))[0]
        assert result.credit >= .8
        assert result.evidence == [evidence]


def test_compound_responsibilities_get_only_supported_credit():
    duties = facts('Lead code reviews and mentor junior developers')
    result = responsibility_matches(duties, facts('Mentored 3 junior developers', 'Led a development team'))[0]
    assert result.credit == .5
    assert result.evidence == ['Mentored 3 junior developers']
    partial = responsibility_matches(facts('Optimize application performance and security'),
                                     facts('Reduced API response time by 40%'))[0]
    assert partial.credit == .5


def test_action_alone_and_negation_do_not_satisfy_a_responsibility():
    for evidence in ['Implemented payroll policy', 'Never implemented CI/CD pipelines']:
        assert responsibility_matches(facts('Implement CI/CD pipelines'), facts(evidence))[0].credit == 0


def test_experience_bullets_and_projects_are_compared():
    resume = ResumeData(roles=[Role(title='Developer', start_date=None, end_date=None,
        quote='Developer: Implemented CI/CD pipelines')], projects=facts('Reduced API response time by 40%'))
    result, _ = evaluate(resume, JobDescriptionData(responsibilities=facts(
        'Implement CI/CD pipelines', 'Optimize application performance')))
    assert result.criteria['responsibilities'].score == 100


def test_business_analysis_years_are_not_software_years():
    resume = candidate('Business Analyst', ['Excel'], 5, 2021, ['Prepared financial forecasts'])
    result, _ = evaluate(resume)
    assert result.candidate_years == 5
    assert result.criteria['experience'].score == 0
    assert not result.criteria['experience'].matches[0].evidence


def test_mixed_roles_only_count_relevant_periods_and_merge_overlaps():
    resume = ResumeData(total_experience_years=Years(value=10.0, quote='10 years experience'), roles=[
        Role(title=title, start_date=None, end_date=None, quote=f'{title} {span}')
        for title, span in [('Business Analyst', '2016 - 2024'), ('Software Developer', '2022 - 2025'),
                            ('Software Engineer', '2023 - 2026')]])
    result, _ = evaluate(resume)
    assert result.criteria['experience'].score == 80  # Four distinct relevant years / five.
    assert all('Business Analyst' not in q for q in result.criteria['experience'].matches[0].evidence)


def test_new_skills_and_projects_do_not_relabel_old_unrelated_tenure():
    resume = candidate('Business Analyst', ['JS', 'React', 'Python', 'AWS', 'Postgres', 'Docker', 'Git'],
                       8, 2018, ['Prepared budgets'])
    resume.projects = facts('Implemented CI/CD pipelines', 'Built web applications')
    result, _ = evaluate(resume)
    assert result.criteria['experience'].score == 0
    assert result.criteria['required_skills'].score == 100


def test_unknown_durations_use_conservative_mixed_role_relevance():
    resume = ResumeData(total_experience_years=Years(value=8.0, quote='8 years experience'), roles=[
        Role(title=title, start_date=None, end_date=None, quote=title)
        for title in ['Software Developer', 'Business Analyst']])
    result, _ = evaluate(resume)
    assert result.criteria['experience'].score == 0
    assert 'conservatively' in result.criteria['experience'].explanation


def test_shared_generic_title_is_not_role_equivalence():
    assert role_similarity('Software Engineer', 'Civil Engineer') == 0
    assert role_similarity('Business Analyst', 'Laboratory Analyst') == 0
    assert role_similarity('Software Engineer', 'Software Developer') == 1


def test_safeguard_combines_evidence_instead_of_vetoing_missing_skills():
    resume = candidate('Senior Full-Stack Developer', [], 7, 2019,
        ['Implemented CI/CD pipelines', 'Built web applications', 'Led code reviews', 'Mentored junior developers'])
    result, score = evaluate(resume)
    assert result.criteria['required_skills'].score == 0
    assert score.overall > 25
    assert score.overall == round(sum(score.contributions.values()), 2)


def test_identity_does_not_affect_fit_and_scoring_is_repeatable():
    resume = candidate('Software Developer', ['Python', 'JS'], 4, 2022, ['Implemented CI/CD pipelines'])
    result, before = evaluate(resume)
    resume.candidate_name = facts('Different Synthetic Name')[0]
    _, after = evaluate(resume)
    assert before == after
    assert score_match(result) == score_match(result)
