"""Evidence-backed local matching with responsibility and employment relevance."""
import re
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.schemas import (CriterionResult, Fact, JobDescriptionData, MatchResult,
                            RequirementMatch, ResumeData)
from services.experience import candidate_experience, relevant_experience
from services.local_relevance import responsibility_matches, role_matches
from services.normalizer import clean, normalize_role, normalize_skill
from services.requirements import requirements, requirement_coverage, structured_evidence
from parsers.document_parser import contact_evidence


def criterion(matches: list[RequirementMatch], explanation: str, *, confidence: float = .9,
              weights: list[float] | None = None) -> CriterionResult:
    weights = weights or [1.0] * len(matches)
    score = 100 * sum(m.credit * w for m, w in zip(matches, weights)) / sum(weights) if matches else 0.0
    return CriterionResult(applicable=bool(matches), score=round(score, 4), matches=matches,
        missing_requirements=[m.requirement for m in matches if m.credit < 1],
        explanation=explanation if matches else 'Not specified in the JD; excluded from scoring.',
        confidence=confidence if matches else 1.0)


def unique(facts: list[Fact]) -> list[Fact]:
    distinct = {}
    for fact in facts:
        distinct.setdefault(normalize_skill(fact.value), fact)
    return list(distinct.values())


def candidate_technologies(resume: ResumeData) -> list[Fact]:
    return unique(resume.skills + resume.tools + resume.programming_languages +
                  resume.frameworks + resume.databases + resume.cloud_platforms)


def coverage(requirements: list[Fact], candidates: list[Fact], *, required: bool = True) -> list[RequirementMatch]:
    lookup = {normalize_skill(f.value): f for f in candidates}
    result = []
    for req in unique(requirements):
        found = lookup.get(normalize_skill(req.value))
        method = 'Exact match' if found and clean(req.value) == clean(found.value) else 'Equivalent alias'
        result.append(RequirementMatch(requirement=req.value, required=required, credit=1.0 if found else 0.0,
            evidence=[found.quote] if found else [], method=method if found else 'No evidence'))
    return result


def text_similarity(left: str, right: str) -> float:
    if normalize_role(left) == normalize_role(right):
        return 1.0
    try:
        vectors = TfidfVectorizer(ngram_range=(1, 2), stop_words='english').fit_transform([left, right])
        return float(cosine_similarity(vectors[0], vectors[1])[0, 0])
    except ValueError:
        return 0.0


def textual_matches(requirements: list[Fact], candidates: list[Fact]) -> list[RequirementMatch]:
    matches = []
    for req in unique(requirements):
        best = max(candidates, key=lambda c: text_similarity(req.value, c.value), default=None)
        similarity = text_similarity(req.value, best.value) if best else 0.0
        # Lexical similarity is conservative; it is never labeled semantic equivalence.
        credit = 1.0 if similarity >= .95 else .5 if similarity >= .65 else 0.0
        matches.append(RequirementMatch(requirement=req.value, credit=credit,
            evidence=[best.quote] if credit else [],
            method='Exact wording' if credit == 1 else 'Partial lexical overlap' if credit else 'No sufficient lexical evidence'))
    return matches


DEGREES = {'other': 0, 'high_school': 1, 'associate': 2, 'bachelor': 3, 'master': 4, 'doctorate': 5}
RELATED_COMPUTING = {'computer science', 'computer engineering', 'software engineering',
                     'information technology', 'computing', 'informatics'}


def education_match(jd: JobDescriptionData, resume: ResumeData) -> CriterionResult:
    if not jd.education_requirements:
        return criterion([], '')
    alternatives = []
    for req in jd.education_requirements:
        label = f'{req.level}: {req.field or "any field"}'
        best_credit, best_quote, best_note = 0.0, None, 'No degree evidence'
        for candidate in resume.education:
            level = candidate.level == req.level if req.level == 'other' else DEGREES[candidate.level] >= DEGREES[req.level]
            field = not req.field or (candidate.field and clean(req.field) == clean(candidate.field))
            related = bool(req.related_field_allowed and candidate.field and req.field
                           and clean(req.field) in RELATED_COMPUTING and clean(candidate.field) in RELATED_COMPUTING)
            credit = .5 * bool(level) + .5 * bool(field or related)
            if credit > best_credit:
                best_credit, best_quote = credit, candidate.quote
                best_note = f'Degree level: {"met" if level else "not evidenced"}; field: {"met" if field or related else "not evidenced"}'
        alternatives.append(RequirementMatch(requirement=label, credit=best_credit,
            evidence=[best_quote] if best_quote else [], method=best_note))
    best = max(alternatives, key=lambda item: item.credit)
    return criterion([best], 'Degree level and field each contribute half. Acceptable degree alternatives use the best match.')


def quality_match(resume: ResumeData, text: str) -> CriterionResult:
    checks = [
        ('Candidate identity', [resume.candidate_name.quote] if resume.candidate_name else []),
        ('Contact information', contact_evidence(text)),
        ('Headline or summary', [resume.headline.quote] if resume.headline else []),
        ('Skills', [f.quote for f in candidate_technologies(resume)][:1]),
        ('Employment or projects', [r.quote for r in resume.roles] or [p.quote for p in resume.projects]),
        ('Education', [e.quote for e in resume.education]),
        ('Measurable achievements', [f.quote for f in resume.achievements if re.search(r'\d', f.quote)]),
        ('Readable text and sections', [text[:150]] if len(text.splitlines()) >= 3 and len(text) >= 80 else []),
    ]
    return criterion([RequirementMatch(requirement=label, credit=float(bool(quotes)),
                      evidence=quotes, method='Source evidence' if quotes else 'Not evidenced')
                      for label, quotes in checks],
                     'Basic completeness only; this does not inspect visual layout or emulate a proprietary ATS.')


def match_documents(jd: JobDescriptionData, resume: ResumeData, resume_text: str,
                    *, as_of: date) -> MatchResult:
    tech = candidate_technologies(resume)
    evidence = structured_evidence(resume)
    required = requirements([('required_skills', jd.required_skills), ('required_tools', jd.required_tools)], required=True)
    used = {frozenset(g.alternatives) for g in required}
    preferred = requirements([('preferred_skills', jd.preferred_skills)], required=False, excluded=used)
    used |= {frozenset(g.alternatives) for g in preferred}
    tools_preferred = requirements([('preferred_tools', jd.preferred_tools)], required=False, excluded=used)
    tool_matches = requirement_coverage(tools_preferred, tech, evidence)
    years, years_evidence, warnings = candidate_experience(resume, as_of)
    relevant_years, relevant_quotes, experience_explanation = relevant_experience(jd, resume, as_of, years)
    exp_matches = []
    if jd.minimum_experience_years and jd.minimum_experience_years.value > 0:
        minimum = jd.minimum_experience_years.value
        exp_matches = [RequirementMatch(requirement=f'At least {minimum:g} years',
            credit=min(1.0, relevant_years / minimum) if relevant_years is not None else 0.0,
            evidence=relevant_quotes, method='Relevance-weighted experience; unrelated employment excluded')]
    if years is None:
        warnings.append('Experience could not be reliably determined.')
    if jd.maximum_experience_years:
        warnings.append('The JD maximum experience is shown in extracted data; exceeding it is not penalized.')
    role_facts = ([resume.headline] if resume.headline else []) + [Fact(value=r.title, quote=r.quote) for r in resume.roles]
    criteria = {
        'required_skills': criterion(requirement_coverage(required, tech, evidence), 'Equal coverage of mandatory skills and tools; alternatives count once.'),
        'preferred_skills': criterion(requirement_coverage(preferred, tech, evidence), 'Equal coverage of distinct preferred skills.'),
        'tools': criterion(tool_matches, 'Optional tools only; mandatory tools count in required coverage without duplication.'),
        'experience': criterion(exp_matches, experience_explanation, confidence=.75 if relevant_years is not None else .2),
        'role_relevance': criterion(role_matches(jd.job_title, role_facts), 'Local title and specialization relevance from cited roles/headline.', confidence=.75),
        'responsibilities': criterion(responsibility_matches(unique(jd.responsibilities), evidence), 'Each activity is compared with all cited work, achievements and projects; compound requirements receive only supported coverage.', confidence=.7),
        'education': education_match(jd, resume),
        'certifications': criterion(coverage(jd.certifications, resume.certifications), 'Only explicitly requested certifications count.'),
        'domain': criterion(textual_matches([jd.domain] if jd.domain else [], resume.domains), 'Explicit industry/domain evidence only.', confidence=.75),
        'resume_quality': quality_match(resume, resume_text),
    }
    return MatchResult(criteria=criteria, candidate_years=years,
                       warnings=warnings + jd.uncertainties + resume.uncertainties)
