"""Keep offline tests isolated from the user's persistent document cache."""
import pytest


@pytest.fixture(autouse=True)
def isolated_extraction_cache(tmp_path, monkeypatch):
    # Existing tests exercise Gemini regardless of the user's local selection.
    # Credentials are synthetic and never read from the user's .env in tests.
    monkeypatch.setenv('AI_PROVIDER', 'gemini')
    monkeypatch.setenv('GEMINI_API_KEY', 'offline-test-key')
    monkeypatch.setenv('SARVAM_API_KEY', 'offline-test-key')
    from services import extraction_cache
    monkeypatch.setattr(extraction_cache, 'CACHE_ROOT', extraction_cache.filesystem_path(tmp_path / '.cache'))
