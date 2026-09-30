"""Local requirement groups and explicit evidence coverage; no extraction or inference."""
import re
from dataclasses import dataclass
from models.schemas import Fact, ResumeData, RequirementMatch
from services.normalizer import clean, normalize_skill, term_in_quote, LOOKUP


@dataclass(frozen=True)
class Requirement:
    alternatives: tuple[str, ...]
    source_field: str
    evidence: str
    required: bool
    label: str

    @property
    def canonical_name(self):
        return ' OR '.join(self.alternatives)


def alternatives(text):
    # Slash is ambiguous: accept only two known, separately named technologies.
    if re.search(r'\s+or\s+', text, re.I):
        parts = re.split(r'\s+or\s+', text, flags=re.I)
    elif '/' in text and clean(text) not in LOOKUP:
        parts = text.split('/')
        known = set(LOOKUP) | {'tensorflow', 'pytorch', 'python', 'sql', 'java', 'docker'}
        if not all(clean(p) in known for p in parts):
            return ()
    else:
        return ()
    return tuple(dict.fromkeys(normalize_skill(p) for p in parts if p.strip()))


def requirements(fields, *, required, excluded=()):
    result, seen = [], set(excluded)
    quoted_values = {}
    for _, facts in fields:
        for fact in facts:
            quoted_values.setdefault(fact.quote, set()).add(normalize_skill(fact.value))
    for field, facts in fields:
        for fact in facts:
            options = alternatives(fact.value)
            quoted = alternatives(fact.quote)
            # Reconstruct cached split facts only when the entire quote is a list
            # of alternatives containing this fact, not an arbitrary sentence.
            if (not options and len(quoted) > 1
                    and set(quoted) <= quoted_values[fact.quote]):
                options = quoted
            options = options or (normalize_skill(fact.value),)
            key = frozenset(options)
            if key in seen:
                continue
            seen.add(key)
            result.append(Requirement(options, field, fact.quote, required, " OR ".join(options) if len(options) > 1 else fact.value))
    return result


def structured_evidence(resume: ResumeData):
    fields = ('skills', 'tools', 'programming_languages', 'frameworks', 'databases',
              'cloud_platforms', 'roles', 'responsibilities', 'achievements', 'projects')
    quotes = [f.quote for field in fields for f in getattr(resume, field)] + resume.evidence
    return [Fact(value=q, quote=q) for q in dict.fromkeys(quotes)]


def positive(quote):
    return not re.search(r'\b(no|not|never|without|lack|lacks|lacking)\b', clean(quote))


def evidence_supports(term, quote):
    if not positive(quote):
        return False
    if term_in_quote(term, quote):
        return True
    return term == 'data visualization' and bool(re.search(
        r'\bvisualization\s*:\s*(?:tableau|matplotlib)\b', clean(quote)))


def requirement_coverage(groups, technologies, evidence):
    matches = []
    for group in groups:
        found = next((f for f in technologies if normalize_skill(f.value) in group.alternatives
                      and positive(f.quote)), None)
        if found is None:
            found = next((f for f in evidence if any(evidence_supports(t, f.quote)
                          for t in group.alternatives)), None)
        matches.append(RequirementMatch(requirement=group.label, required=group.required,
            credit=float(found is not None), evidence=[found.quote] if found else [],
            method='Explicit value or quoted evidence' if found else 'No evidence'))
    return matches
