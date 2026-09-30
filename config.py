"""Configuration. Initial weights are hypotheses, not benchmark-calibrated claims."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=False)
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 80_000
MIN_TEXT_CHARS = 80
MAX_PAGES = 100
MAX_RESUMES = 10
MODEL = os.getenv('GEMINI_MODEL', '').strip()
SEMANTIC_MATCHING = os.getenv('ATS_SEMANTIC_MATCHING', 'true').lower() == 'true'
API_TIMEOUT = 60.0


class ConfigurationError(ValueError):
    pass


def get_provider() -> str:
    provider = os.getenv('AI_PROVIDER', 'gemini').strip().lower()
    if provider not in {'gemini', 'sarvam'}:
        raise ConfigurationError('Unsupported AI_PROVIDER. Use gemini or sarvam.')
    return provider


def get_model(provider: str | None = None) -> str:
    provider = get_provider() if provider is None else provider
    return os.getenv(f'{provider.upper()}_MODEL', 'sarvam-105b' if provider == 'sarvam' else '').strip()


def get_api_key(provider: str | None = None) -> str:
    """Read only the selected provider's key; never display or persist it."""
    provider = get_provider() if provider is None else provider
    return os.getenv(f'{provider.upper()}_API_KEY', '').strip()


SCORING_VERSION = '2.0'
WEIGHTS = {
    'required_skills': .32, 'preferred_skills': .08, 'tools': .10,
    'experience': .15, 'role_relevance': .10, 'responsibilities': .10,
    'education': .06, 'certifications': .03, 'domain': .02, 'resume_quality': .04,
}
