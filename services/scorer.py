"""No API dependency: scores are deterministic weighted sums."""
import math
from config import SCORING_VERSION, WEIGHTS
from models.schemas import MatchResult, ScoreResult


def score_match(result: MatchResult, weights: dict[str, float] | None = None) -> ScoreResult:
    weights = WEIGHTS if weights is None else weights
    if set(weights) != set(result.criteria):
        raise ValueError('Weights must cover exactly the scoring criteria.')
    if any(not math.isfinite(w) or w < 0 for w in weights.values()) or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8):
        raise ValueError('Weights must be finite, nonnegative, and sum to 1.0.')
    active = {name: weight for name, weight in weights.items() if result.criteria[name].applicable and weight > 0}
    total = sum(active.values())
    effective = {name: weight / total for name, weight in active.items()} if total else {}
    # Prevent sparse JDs from turning the 4% presentation factor into a dominant score.
    if effective.get('resume_quality', 0) > .05 and len(effective) > 1:
        quality = effective.pop('resume_quality')
        effective = {name: w * .95 / (1 - quality) for name, w in effective.items()}
        effective['resume_quality'] = .05
    if set(effective) == {'resume_quality'}:
        effective = {}  # no actual job criterion: no meaningful job-match score
    contributions = {name: result.criteria[name].score * weight for name, weight in effective.items()}
    primary = {'required_skills', 'tools', 'role_relevance', 'responsibilities', 'experience', 'domain'}
    primary_weight = sum(weight for name, weight in effective.items() if name in primary)
    primary_fit = (sum(contributions[name] for name in contributions if name in primary)
                   / (100 * primary_weight)) if primary_weight else 0.0
    # Supporting qualifications cannot create job compatibility without substantive
    # fit. This uses the existing weights and all applicable primary criteria, not
    # a required-skills-only veto or a candidate-specific score adjustment.
    contributions = {name: value if name in primary else value * primary_fit
                     for name, value in contributions.items()}
    overall = round(min(100.0, max(0.0, sum(contributions.values()))), 2)
    classification = ('Insufficient JD criteria' if not effective else 'Strong match' if overall >= 85
                      else 'Moderate match' if overall >= 65 else 'Limited match' if overall >= 40 else 'Low match')
    confidence = sum(result.criteria[name].confidence * weight for name, weight in effective.items())
    return ScoreResult(overall=overall, effective_weights=effective, contributions=contributions,
                       classification=classification, confidence=confidence, scoring_version=SCORING_VERSION)
