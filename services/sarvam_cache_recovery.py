"""Targeted session cleanup for the two confirmed bad Sarvam extractions."""
import hashlib
import json

from services.extraction_completeness import JD_TECH_FIELDS, TECH_FIELDS

BAD_NAMESPACE = '4378bdaaddc83f738f7036d4dc9f35577e42daed2de0bb726ee2ac6718a91bc7'
BAD_JD = 'ab916d897bbaf60baefb0a923e378a86262addbaad98114bb183e9f4919c9330'
BAD_RESUME = '00466b7cbc748a69ba623025f63b0fc37494c22c602d29aad027652fef55f1fa'


def _affected(identity, kind, digest, data):
    if not isinstance(identity, tuple) or len(identity) != 3 or identity[0] != 'sarvam':
        return False
    if hashlib.sha256(json.dumps(list(identity[1:])).encode()).hexdigest() != BAD_NAMESPACE:
        return False
    if kind == 'jd' and digest == BAD_JD:
        return not any(getattr(data, field, None) for field in JD_TECH_FIELDS)
    if kind == 'resume' and digest == BAD_RESUME:
        fields = (*TECH_FIELDS, 'roles', 'education', 'responsibilities', 'projects',
                  'achievements', 'certifications', 'total_experience_years')
        return not any(getattr(data, field, None) for field in fields)
    return False


def clear_bad_session_entries(state):
    """Preserve unrelated caches and repaired extractions of these same documents."""
    bad_objects = set()
    memory = state.get('extraction_cache', {})
    for key, data in list(memory.items()):
        if isinstance(key, tuple) and len(key) == 5 and _affected(key[:3], key[3], key[4], data):
            bad_objects.add(id(data))
            del memory[key]
    analyses = state.get('analysis_cache', {})
    for key, result in list(analyses.items()):
        if not isinstance(key, tuple) or len(key) != 6:
            continue
        affected = [(kind, digest) for kind, digest in (('jd', key[1]), ('resume', key[2]))
                    if _affected(key[3], kind, digest, result.get(kind))]
        if affected:
            bad_objects.update(id(result.get(kind)) for kind, _ in affected)
            del analyses[key]
    results = state.get('results', [])
    retained = [result for result in results if not (
        id(result.get('jd')) in bad_objects or id(result.get('resume')) in bad_objects)]
    if len(retained) != len(results):
        state['results'] = retained
