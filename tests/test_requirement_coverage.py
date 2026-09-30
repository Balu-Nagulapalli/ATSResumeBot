"""Synthetic local coverage checks; no documents, benchmark scores, or APIs."""
from datetime import date
import pytest
from models.schemas import Fact, JobDescriptionData, ResumeData
from services.matcher import match_documents
from services.normalizer import normalize_skill
from services.local_relevance import responsibility_matches, responsibility_clauses


def facts(*values):
    return [Fact(value=v, quote=v) for v in values]


def match(jd, resume):
    return match_documents(jd, resume, '', as_of=date(2026, 1, 1)).criteria


def test_mandatory_tools_pool_all_fields_and_deduplicate():
    jd = JobDescriptionData(required_skills=facts('Python'), required_tools=facts('Python', 'SQL', 'Pandas'))
    result = match(jd, ResumeData(programming_languages=facts('Python'), databases=facts('SQL'), frameworks=facts('Pandas')))
    assert result['required_skills'].score == 100
    assert len(result['required_skills'].matches) == 3
    assert not result['tools'].applicable


@pytest.mark.parametrize('value,credit', [('TensorFlow', 1), ('PyTorch', 1), ('JAX', 0)])
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('preferred', [False, True])
def test_alternatives(value, credit, split, preferred):
    quote = 'TensorFlow or PyTorch'
    reqs = [Fact(value=v, quote=quote) for v in (['TensorFlow', 'PyTorch'] if split else [quote])]
    field = 'preferred_skills' if preferred else 'required_tools'
    result = match(JobDescriptionData(**{field: reqs}), ResumeData(frameworks=facts(value)))
    matches = result['preferred_skills' if preferred else 'required_skills'].matches
    assert len(matches) == 1
    assert matches[0].credit == credit
    assert matches[0].required is not preferred


def test_slash_alternative_preserves_technology_names():
    jd = JobDescriptionData(required_tools=facts('TensorFlow/PyTorch', 'CI/CD', 'TCP/IP'))
    result = match(jd, ResumeData(tools=facts('PyTorch', 'CI/CD', 'TCP')))
    assert [m.credit for m in result['required_skills'].matches] == [1, 1, 0]


def test_prose_quote_does_not_turn_a_skill_into_an_alternative():
    jd = JobDescriptionData(required_skills=[Fact(value='Python', quote='Python or equivalent experience required')])
    result = match(jd, ResumeData(skills=facts('Python')))
    assert result['required_skills'].matches[0].requirement == 'Python'


@pytest.mark.parametrize('left,right', [('ML','Machine Learning'), ('NLP','Natural Language Processing'),
    ('Natural Language Processing (NLP)','NLP'), ('Apache Spark','Spark'), ('R Programming','R'),
    ('Scikit learn','scikit-learn'), ('sklearn','scikit-learn')])
def test_aliases(left, right):
    assert normalize_skill(left) == normalize_skill(right)


@pytest.mark.parametrize('requirement,evidence,credit', [
    ('Machine Learning', 'Created ML models using scikit-learn', 1),
    ('Statistical Analysis', 'Python; won Kaggle competitions; published papers', 0),
    ('Data Visualization', 'Visualization: Tableau, Matplotlib', 1),
    ('Machine Learning', 'TensorFlow', 0),
    ('Data Science', 'Python', 0),
    ('Machine Learning', 'Never created ML models', 0),
    ('Java', 'JavaScript', 0),
])
def test_explicit_capabilities_only(requirement, evidence, credit):
    result = match(JobDescriptionData(required_skills=facts(requirement)), ResumeData(evidence=[evidence]))
    item = result['required_skills'].matches[0]
    assert item.credit == credit
    assert item.evidence == ([evidence] if credit else [])


def test_required_preferred_are_separate():
    result = match(JobDescriptionData(required_tools=facts('Python'), preferred_skills=facts('Python', 'SQL')),
                   ResumeData(skills=facts('SQL')))
    assert result['required_skills'].score == 0
    assert len(result['preferred_skills'].matches) == 1
    assert result['preferred_skills'].score == 100


def test_responsibility_actions_and_adjectives():
    assert responsibility_clauses('Analyze large and complex datasets') == ['Analyze large and complex datasets']
    assert responsibility_matches(facts('Analyze large and complex datasets'), facts('Analyzed datasets'))[0].credit == 1
    result = responsibility_matches(facts('Develop and train machine learning models'), facts('Created ML models using scikit-learn'))[0]
    assert result.credit >= .75
    assert result.credit < 1  # Model creation does not explicitly establish training.


def test_joint_model_deployment_evidence_and_unrelated_negative():
    requirement = facts('Deploy models to production')
    deployment = 'Deployed TensorFlow models serving predictions'
    production = 'Built computer vision pipeline in production'
    result = responsibility_matches(requirement, facts(deployment, production))[0]
    assert result.credit == 1
    assert result.evidence == [deployment, production]
    for unrelated in ['Managed factory production', 'Built payment pipeline in production', 'Never built ML pipeline in production']:
        assert responsibility_matches(requirement, facts(deployment, unrelated))[0].credit == .5


def test_generic_data_is_not_work_and_technology_quotes_can_prove_activity():
    requirement = facts('Create data visualizations and dashboards')
    assert responsibility_matches(requirement, facts('Senior Data Scientist'))[0].credit == 0
    resume = ResumeData(tools=[Fact(value='Tableau', quote='Created visualizations and dashboards using Tableau')])
    result = match(JobDescriptionData(responsibilities=requirement), resume)
    assert result['responsibilities'].score == 100
