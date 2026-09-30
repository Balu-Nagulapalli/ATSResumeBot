"""Local, auditable text relevance. Phrase equivalences never imply technology equivalence."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.schemas import Fact, RequirementMatch
from services.normalizer import clean
from services.requirements import positive

# Actions and work outcomes, not candidate-specific rules or a skill extractor.
PHRASES = {
    r'\bml\b|\bmachine learning\b': 'machine_learning',
    r'\bnlp\b|\bnatural language processing(?:\s*\(nlp\))?\b': 'natural_language_processing',
    r'\bscikit[ -]learn\b|\bsklearn\b': 'scikit_learn',
    r'\bcomputer vision\b': 'computer_vision',
    r'\bci\s*[/ -]\s*cd\b|\bcontinuous integration(?: and continuous (?:delivery|deployment))?\b': 'cicd',
    r'\bcode reviews?\b|\bpeer code reviews?\b': 'code_review',
    r'\bresponse times?\b|\blatency\b|\bthroughput\b|\bload times?\b': 'performance',
    r'\bcross[ -]functional\b': 'crossfunctional',
    r'\bfull[ -]?stack\b': 'fullstack',
    r'\bfront[ -]?end\b': 'frontend',
    r'\bback[ -]?end\b': 'backend',
}
WORD_GROUPS = {
    'train': 'train trains trained training',
    'communicate': 'communicate communicates communicated communicating present presented presentation',
    'implement': 'implement implements implemented implementing implementation introduce introduced establish established',
    'build': 'build builds building built develop develops developed developing create creates created creating deliver delivered',
    'improve': 'improve improves improved improving optimize optimizes optimized optimizing optimization optimise optimised optimisation reduce reduced reducing accelerate accelerated tune tuned',
    'mentor': 'mentor mentors mentored mentoring coach coached coaching',
    'lead': 'lead leads leading led leadership manage manages managed managing management supervise supervised',
    'collaborate': 'collaborate collaborates collaborated collaborating collaboration partner partnered partnering cooperate coordinated coordinate',
    'design': 'design designs designed designing architect architected architecture redesign redesigned',
    'maintain': 'maintain maintains maintained maintaining maintenance support supports supported supporting',
    'test': 'test tests tested testing validate validated validation',
    'deploy': 'deploy deploys deployed deploying deployment release released releasing',
    'analyze': 'analyze analyzes analyzed analyzing analyse analysed analysis',
    'secure': 'secure secures secured securing harden hardened hardening',
    'automate': 'automate automated automating automation',
}
WORDS = {word: canonical for canonical, group in WORD_GROUPS.items() for word in group.split()}
ACTIONS = set(WORD_GROUPS)
CONTEXT = {'application', 'app', 'based', 'using', 'work', 'experience', 'professional',
           'responsibility', 'responsible', 'year', 'data', 'large', 'complex'}


def tokens(text: str) -> list[str]:
    text = clean(text)
    for pattern, replacement in PHRASES.items():
        text = re.sub(pattern, replacement, text)
    words = re.findall(r'[a-z][a-z_+#.]*', text)
    result = []
    for word in words:
        if word in WORDS:
            word = WORDS[word]
        elif word.endswith('ed') and len(word) > 4:
            word = word[:-2]
        elif word.endswith('ing') and len(word) > 5:
            word = word[:-3]
        if word.endswith('s') and not word.endswith(('ss', 'is')) and len(word) > 3:
            word = word[:-1]
        if word not in ENGLISH_STOP_WORDS:
            result.append(word)
    return result


def _cosine(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    matrix = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b').fit_transform([' '.join(left), ' '.join(right)])
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def responsibility_similarity(requirement: str, evidence: str) -> float:
    """Require an object match; a shared generic action alone cannot earn points."""
    if not positive(evidence):
        return 0.0  # Conservative for negated claims; no inferred positive credit.
    wanted, observed = tokens(requirement), tokens(evidence)
    req, cand = set(wanted), set(observed)
    actions = req & ACTIONS
    objects = req - ACTIONS - CONTEXT
    if not objects:
        return 0.0
    overlap = objects & cand
    if not overlap:
        return 0.0
    object_coverage = len(overlap) / len(objects)
    action_coverage = len(actions & cand) / len(actions) if actions else 1.0
    # Matching the subject without the requested activity is only partial evidence.
    supported = object_coverage * (1.0 if action_coverage == 1 else .5 * (1 + action_coverage))
    lexical = _cosine(wanted, observed)
    # N-gram similarity can support a partial match, but cannot manufacture absent objects.
    return round(min(object_coverage, max(supported, lexical)), 4)


def responsibility_matches(requirements: list[Fact], candidates: list[Fact]) -> list[RequirementMatch]:
    matches = []
    for requirement in requirements:
        # Conjoined activities retain independent coverage: mentoring cannot imply code reviews.
        clauses = responsibility_clauses(requirement.value)
        for index, clause in enumerate(clauses[:-1]):
            clause_tokens = set(tokens(clause))
            if clause_tokens & ACTIONS and not clause_tokens - ACTIONS - CONTEXT:
                # "Design and implement APIs" shares the trailing object.
                objects = set(tokens(clauses[index + 1])) - ACTIONS - CONTEXT
                clauses[index] = clause + ' ' + ' '.join(sorted(objects))
        credits, evidence = [], []
        for clause in clauses:
            best = max(candidates, key=lambda fact: responsibility_similarity(clause, fact.quote), default=None)
            credit = responsibility_similarity(clause, best.quote) if best else 0.0
            joint_credit, joint_quotes = joint_deployment(clause, candidates)
            if joint_credit > credit:
                credit = joint_credit
                evidence.extend(q for q in joint_quotes if q not in evidence)
            elif credit and best.quote not in evidence:
                evidence.append(best.quote)
            credits.append(credit)
        credit = sum(credits) / len(credits) if credits else 0.0
        matches.append(RequirementMatch(requirement=requirement.value, credit=credit,
            evidence=evidence, method='Local action/object coverage and TF-IDF' if credit else 'No supported work evidence'))
    return matches


def responsibility_clauses(text: str) -> list[str]:
    clauses = [c.strip() for c in re.split(r'\s+(?:and|&)\s+|;', text) if c.strip()]
    # Adjectives do not define separate activities or objects.
    if len(clauses) == 2 and set(tokens(clauses[0])) & ACTIONS:
        if not set(tokens(clauses[0])) - ACTIONS - CONTEXT:
            if not set(tokens(clauses[1])) & ACTIONS:
                return [text]
    return clauses


def joint_deployment(clause, candidates):
    """Combine deployment and production only within explicit model-work context."""
    required = set(tokens(clause))
    if not {'deploy', 'model', 'production'} <= required:
        return 0.0, []
    contexts = {'machine_learning', 'natural_language_processing', 'computer_vision',
                'tensorflow', 'pytorch', 'scikit_learn'}
    deployments = [f for f in candidates if positive(f.quote)
                   and {'deploy', 'model'} <= set(tokens(f.quote))]
    production = [f for f in candidates if positive(f.quote)
                  and 'production' in tokens(f.quote)
                  and set(tokens(f.quote)) & ACTIONS
                  and ('model' in tokens(f.quote) or
                       ('pipeline' in tokens(f.quote) and set(tokens(f.quote)) & contexts))]
    for deployed in deployments:
        for produced in production:
            if deployed == produced:
                continue
            if not set(tokens(deployed.quote)) & contexts or not set(tokens(produced.quote)) & contexts:
                continue
            score = responsibility_similarity(clause, deployed.quote + ' ' + produced.quote)
            return score, [deployed.quote, produced.quote]
    return 0.0, []


TITLE_STOP = {'senior', 'junior', 'sr', 'jr', 'lead', 'principal', 'staff', 'years',
              'year', 'professional', 'experience', 'required', 'minimum'}
GENERIC_TITLES = {'engineer', 'developer', 'analyst', 'manager', 'consultant', 'specialist', 'director'}
SOFTWARE_AREAS = {'software', 'fullstack', 'frontend', 'backend', 'web', 'mobile', 'application'}


def _title_tokens(title: str) -> set[str]:
    title = clean(title)
    for pattern, replacement in PHRASES.items():
        title = re.sub(pattern, replacement, title)
    title = re.sub(r'\bdevelopment\b', 'developer', title)
    title = re.sub(r'\bengineering\b', 'engineer', title)
    result = set(re.findall(r'[a-z]+', title)) - TITLE_STOP - ENGLISH_STOP_WORDS
    if result & SOFTWARE_AREAS and result & {'engineer', 'developer'}:
        result.discard('engineer')
        result.add('developer')
        result.add('software')
    return result


def role_similarity(required: str, candidate: str) -> float:
    req, cand = _title_tokens(required), _title_tokens(candidate)
    if not req or not cand:
        return 0.0
    if req == cand:
        return 1.0
    shared = req & cand
    # Engineer/manager/analyst alone is not proof that two specific professions align.
    req_scope, cand_scope = req - GENERIC_TITLES, cand - GENERIC_TITLES
    if not shared or (req_scope and cand_scope and not (req_scope & cand_scope)):
        return 0.0
    if not req_scope or not cand_scope:
        return .5 if shared else 0.0
    # Additional candidate specialization does not diminish a broader requested role.
    return len(shared) / len(req)


def role_matches(required: Fact | None, candidates: list[Fact]) -> list[RequirementMatch]:
    if required is None:
        return []
    best = max(candidates, key=lambda fact: role_similarity(required.value, fact.value), default=None)
    credit = role_similarity(required.value, best.value) if best else 0.0
    return [RequirementMatch(requirement=required.value, credit=credit,
        evidence=[best.quote] if credit else [], method='Local role and specialization overlap' if credit else 'No related role evidenced')]
