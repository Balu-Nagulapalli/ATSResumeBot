"""Small local cache containing only schema-validated extraction JSON."""
import os
import re
import tempfile
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError
from config import ROOT, get_provider, get_model
from models.schemas import JobDescriptionData, ResumeData

def filesystem_path(path: Path) -> Path:
    """Keep long Windows cache paths usable, including directory enumeration."""
    if os.name != 'nt':
        return path
    absolute = str(path.absolute())
    if not absolute.startswith('\\\\?\\'):
        absolute = ('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\')
                    else '\\\\?\\' + absolute)
    return Path(absolute)


CACHE_ROOT = filesystem_path(ROOT / '.cache')
SCHEMAS = {'jd': JobDescriptionData, 'resume': ResumeData}
EXTRACTION_VERSION = 'extraction-v3'


def extraction_identity() -> tuple[str, str, str]:
    provider = get_provider()
    # Fingerprint the actual schemas as well as the manually versioned prompt.
    schemas = {kind: schema.model_json_schema() for kind, schema in SCHEMAS.items()}
    version = hashlib.sha256(json.dumps([EXTRACTION_VERSION, schemas], sort_keys=True).encode()).hexdigest()
    return provider, get_model(provider), version


class ExtractionCache:
    def __init__(self, root: Path | None = None, *, identity=None) -> None:
        self.root = filesystem_path(CACHE_ROOT if root is None else Path(root))
        self.identity = extraction_identity() if identity is None else identity

    def path(self, kind: str, digest: str) -> Path:
        if kind not in SCHEMAS or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('Invalid extraction cache key.')
        provider, model, version = self.identity
        if provider not in {'gemini', 'sarvam'}:
            raise ValueError('Invalid extraction cache provider.')
        namespace = hashlib.sha256(json.dumps([model, version]).encode()).hexdigest()
        return self.root / provider / namespace / ('jd' if kind == 'jd' else 'resumes') / f'{digest}.json'

    def load(self, kind: str, digest: str) -> JobDescriptionData | ResumeData | None:
        path = self.path(kind, digest)
        try:
            return SCHEMAS[kind].model_validate_json(path.read_text(encoding='utf-8'))
        except (ValidationError, UnicodeError):
            # A stale or interrupted entry must never prevent normal extraction.
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        except OSError:
            pass
        return None

    def save(self, kind: str, digest: str, result: JobDescriptionData | ResumeData) -> None:
        # Only existing extraction-schema fields are serializable. No source
        # document, credentials, scores, or other envelope metadata are added.
        validated = SCHEMAS[kind].model_validate(result.model_dump())
        payload = validated.model_dump_json()
        path = self.path(kind, digest)
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             prefix='.pending-', suffix='.json', delete=False) as file:
                temporary = Path(file.name)
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        except OSError:
            # Disk caching is best effort; a valid extraction remains usable in memory.
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
