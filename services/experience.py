"""Conservative calendar-month experience calculation with overlap merging."""
import calendar
import re
from datetime import date
from models.schemas import Fact, JobDescriptionData, ResumeData
from services.local_relevance import responsibility_matches, role_similarity
from services.normalizer import normalize_skill, term_in_quote

MONTHS = {name.lower(): i for i in range(1, 13)
          for name in (calendar.month_name[i], calendar.month_abbr[i])}
MONTH_PATTERN = '|'.join(sorted(MONTHS, key=len, reverse=True))
DATE = rf'(?:(?:{MONTH_PATTERN})\s+\d{{4}}|\d{{4}}[-/](?:0?[1-9]|1[0-2])|(?:0?[1-9]|1[0-2])/\d{{4}}|\d{{4}}|Present|Current|Now)'
RANGE = re.compile(rf'(?<!\d)({DATE})\s*(?:\s[-\u2013\u2014]\s|[-\u2013\u2014]|\bto\b)\s*({DATE})(?!\d)', re.I)


def month_index(value: str, as_of: date, *, end: bool = False) -> int | None:
    value = value.strip().lower()
    if value in {'present', 'current', 'now'}:
        return as_of.year * 12 + as_of.month - 1
    if re.fullmatch(r'\d{4}', value):
        # Year-only ranges: use Jan-Jan; do not invent twelve extra months.
        return int(value) * 12
    match = re.fullmatch(r'([a-z]+)\s+(\d{4})', value)
    if match and match[1] in MONTHS:
        return int(match[2]) * 12 + MONTHS[match[1]] - 1 + int(end)
    match = re.fullmatch(r'(\d{4})[-/](\d{1,2})', value)
    if match and 1 <= int(match[2]) <= 12:
        return int(match[1]) * 12 + int(match[2]) - 1 + int(end)
    match = re.fullmatch(r'(\d{1,2})/(\d{4})', value)
    if match and 1 <= int(match[1]) <= 12:
        return int(match[2]) * 12 + int(match[1]) - 1 + int(end)
    return None


def candidate_experience(resume: ResumeData, as_of: date) -> tuple[float | None, list[str], list[str]]:
    intervals, evidence, warnings = [], [], []
    now = as_of.year * 12 + as_of.month - 1
    for role in resume.roles:
        # Parse the quoted source, rather than trusting generated date values.
        ranges = list(RANGE.finditer(role.quote))
        if len(ranges) != 1:
            continue
        match = ranges[0]
        start = month_index(match[1], as_of)
        end = month_index(match[2], as_of, end=True)
        if start is None or end is None or start >= end or start > now or start < 1940 * 12:
            continue
        if end > now + 1:
            warnings.append('A future employment end date was capped at the analysis date.')
        intervals.append((start, min(end, now)))
        evidence.append(role.quote)
        if re.fullmatch(r'\d{4}', match[1]) or re.fullmatch(r'\d{4}', match[2]):
            warnings.append('Year-only employment dates are approximate (January boundaries).')
    calculated = None
    if intervals:
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        calculated = round(sum(end - start for start, end in merged) / 12, 2)
    stated = resume.total_experience_years
    if stated:
        if calculated is not None and abs(calculated - stated.value) > 1:
            warnings.append('Stated experience differs from dated employment; the stated total is used because the history may be incomplete.')
        return stated.value, [stated.quote], sorted(set(warnings))
    return calculated, evidence, sorted(set(warnings))


def relevant_experience(jd: JobDescriptionData, resume: ResumeData, as_of: date,
                        total_years: float | None) -> tuple[float | None, list[str], str]:
    """Estimate relevance-weighted years without assigning global skills to unrelated jobs."""
    if total_years is None:
        return None, [], 'Experience could not be reliably determined.'
    target = jd.job_title.value if jd.job_title else (
        jd.minimum_experience_years.quote if jd.minimum_experience_years else '')
    if jd.minimum_experience_years and re.search(
            r'\b(development|developer|engineering|engineer|analyst|analysis|nursing|teaching|accounting|sales|marketing)\b',
            jd.minimum_experience_years.quote, re.I):
        target = jd.minimum_experience_years.quote
    technologies = {normalize_skill(f.value) for f in jd.required_skills + jd.required_tools}

    def evidence_relevance(title: str, quote: str) -> float:
        title_fit = role_similarity(target, title)
        duties = responsibility_matches(jd.responsibilities, [Fact(value=quote, quote=quote)])
        responsibility_fit = sum(m.credit for m in duties) / len(duties) if duties else 0.0
        tech_fit = sum(term_in_quote(t, quote) for t in technologies) / len(technologies) if technologies else 0.0
        domain_fit = float(bool(jd.domain and term_in_quote(jd.domain.value, quote)))
        # Skills/domain alone cannot label years as relevant work. They corroborate
        # actual role or responsibility evidence in this employment quotation.
        work_fit = max(title_fit, responsibility_fit)
        corroboration = max(tech_fit, domain_fit)
        return max(work_fit, min(1.0, work_fit + (1 - work_fit) * corroboration)) if work_fit else 0.0

    now = as_of.year * 12 + as_of.month - 1
    intervals, quotes, role_fits = [], [], []
    dated_roles = 0
    for role in resume.roles:
        relevance = evidence_relevance(role.title, role.quote)
        role_fits.append(relevance)
        ranges = list(RANGE.finditer(role.quote))
        if len(ranges) != 1:
            continue
        dated_roles += 1
        start = month_index(ranges[0][1], as_of)
        end = month_index(ranges[0][2], as_of, end=True)
        if start is None or end is None or start < 1940 * 12 or start >= min(end, now):
            continue
        intervals.append((start, min(end, now), relevance))
        if relevance:
            quotes.append(role.quote)
    if intervals:
        # Sweep boundaries, taking the greatest supported relevance at each month.
        # Concurrent jobs never add more than one month of experience per month.
        boundaries = sorted({point for start, end, _ in intervals for point in (start, end)})
        months = sum((end - start) * max((fit for a, b, fit in intervals if a <= start < b), default=0.0)
                     for start, end in zip(boundaries, boundaries[1:]))
        years = min(total_years, round(months / 12, 2))
        return years, quotes, f'Relevance-weighted dated employment: {years:g} years (overlaps counted once; capped by stated total).'

    if dated_roles:
        return 0.0, [], 'No valid elapsed employment intervals support relevant experience.'

    if role_fits:
        # Without dates there is no defensible way to assign the stated total among
        # mixed occupations. The least-supported role bounds the whole total.
        factor = min(role_fits)
        quotes = [role.quote for role in resume.roles if factor]
        note = 'Undated employment: conservatively bounded by the least relevant listed role.'
    else:
        # A total explicitly scoped to the requested profession is usable. A generic
        # total needs both a related headline and corroborating work/technical evidence.
        stated = resume.total_experience_years
        scoped_fit = role_similarity(target, stated.quote) if stated else 0.0
        headline_fit = role_similarity(target, resume.headline.value) if resume.headline else 0.0
        work = resume.responsibilities + resume.achievements  # Projects alone do not prove professional tenure.
        duties = responsibility_matches(jd.responsibilities, work)
        duty_fit = sum(m.credit for m in duties) / len(duties) if duties else 0.0
        tech = resume.skills + resume.tools + resume.programming_languages + resume.frameworks + resume.databases + resume.cloud_platforms
        found = {normalize_skill(f.value) for f in tech}
        tech_fit = len(technologies & found) / len(technologies) if technologies else 0.0
        domain_fit = float(bool(jd.domain and any(normalize_skill(d.value) == normalize_skill(jd.domain.value) for d in resume.domains)))
        factor = max(scoped_fit, min(headline_fit, max(duty_fit, tech_fit, domain_fit)))
        quotes = ([resume.headline.quote] if resume.headline and factor else []) + [q for m in duties for q in m.evidence]
        if factor and tech_fit:
            quotes += [f.quote for f in tech if normalize_skill(f.value) in technologies]
        if factor and domain_fit:
            quotes += [d.quote for d in resume.domains]
        note = 'No dated employment: conservative relevance estimate from the stated summary and corroborating evidence.'
    if resume.total_experience_years and factor:
        quotes.append(resume.total_experience_years.quote)
    years = round(total_years * factor, 2)
    return years, list(dict.fromkeys(quotes)), f'{note} Relevance-weighted estimate: {years:g} years.'
