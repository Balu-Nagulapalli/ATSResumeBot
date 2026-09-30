"""Aliases standardize extracted terms; they do not discover candidate skills."""
import re
import unicodedata

ALIASES = {
    'javascript': ['js', 'javascript'], 'react': ['react.js', 'reactjs', 'react'],
    'node.js': ['node', 'nodejs', 'node.js'], 'postgresql': ['postgres', 'postgresql'],
    'rest api': ['rest', 'restful api', 'rest apis', 'restful apis', 'rest api'],
    'aws': ['aws', 'amazon web services'], 'gcp': ['gcp', 'google cloud platform'],
    'machine learning': ['ml', 'machine learning'],
    'natural language processing': ['nlp', 'natural language processing', 'natural language processing (nlp)'],
    'spark': ['spark', 'apache spark'], 'r': ['r', 'r programming'],
    'scikit-learn': ['scikit-learn', 'scikit learn', 'sklearn'],
    'ci/cd': ['ci cd', 'ci/cd', 'ci-cd', 'continuous integration and continuous delivery',
              'continuous integration/continuous deployment'],
    'microservices': ['microservices architecture', 'microservice architecture', 'microservices'],
    'typescript': ['ts', 'typescript'], 'c#': ['c sharp', 'c#'],
    '.net': ['dotnet', '.net'], 'kubernetes': ['k8s', 'kubernetes'],
}


def clean(value: str) -> str:
    value = unicodedata.normalize('NFKC', value).casefold().strip()
    return re.sub(r'\s+', ' ', value).replace('\u2013', '-').replace('\u2014', '-')


LOOKUP = {clean(alias): canonical for canonical, aliases in ALIASES.items() for alias in aliases}


def normalize_skill(value: str) -> str:
    value = clean(value)
    return LOOKUP.get(value, value)


def term_in_quote(value: str, quote: str) -> bool:
    canonical = normalize_skill(value)
    return any(re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', clean(quote))
               for alias in ALIASES.get(canonical, [canonical]))


def normalize_role(value: str) -> str:
    return re.sub(r'[^\w]+', ' ', clean(value)).strip()
