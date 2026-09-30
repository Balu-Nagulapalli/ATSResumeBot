"""Sarvam structured extraction via the supported v1 chat completion API."""
import time
from types import SimpleNamespace

import httpx
from pydantic import ValidationError
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from config import API_TIMEOUT, get_api_key, get_model
from models.schemas import JobDescriptionData, ResumeData
from services.extraction import ExtractionError, T, _retry_delay
from services.sarvam_diagnostics import diagnostic as _diagnostic, safe_validation_errors

ENDPOINT = 'https://api.sarvam.ai/v1/chat/completions'
RESUME_TIMEOUT = 90.0


def strict_json_schema(schema):
    """Require explicit missing values in the wire schema, without altering models."""
    result = schema.model_json_schema()

    def visit(node):
        if isinstance(node, dict):
            node.pop('default', None)
            if node.get('type') == 'object':
                node['additionalProperties'] = False
                node['required'] = list(node.get('properties', {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(result)
    return result


def structured_request(schema: type[T], instructions: str, text: str, *, client=None,
                       max_attempts: int = 3, max_tokens: int = 14000) -> T:
    stage = 'JD extraction' if schema is JobDescriptionData else 'Resume extraction'
    timeout = RESUME_TIMEOUT if schema is ResumeData else API_TIMEOUT
    key = get_api_key('sarvam')
    model = get_model('sarvam')
    if not key or not model:
        raise ExtractionError('AI extraction is currently unavailable. Check local AI configuration.')
    if not 1 <= max_attempts <= 3:
        raise ValueError('Extraction allows one to three attempts.')
    payload = {
        'model': model,
        'messages': [{'role': 'system', 'content': instructions}, {'role': 'user', 'content': text}],
        'temperature': 0, 'max_tokens': max_tokens, 'stream': False,
        'reasoning_effort': None,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': schema.__name__, 'strict': True, 'schema': strict_json_schema(schema)}},
    }
    owned = client is None
    status = None
    response_type = None
    def capture_status(response):
        nonlocal status, response_type
        status = response.status_code
        response_type = type(response).__name__
    try:
        if owned:
            client = httpx.Client(timeout=timeout, follow_redirects=False)
        # Capture only metadata, including when SDK response decoding itself fails.
        client.event_hooks['response'].append(capture_status)
        sdk = SarvamAI(api_subscription_key=key, httpx_client=client, timeout=timeout)
        for attempt in range(max_attempts):
            status = None
            response_type = None
            parsing = False
            try:
                wrapped = sdk.chat.with_raw_response.completions(**payload, request_options={'max_retries': 0})
                status = wrapped.status_code
                response = wrapped.data
                parsing = True
                response_type = type(response).__name__
                choice = response.model_dump()['choices'][0]
                if choice.get('finish_reason') != 'stop' or choice['message'].get('refusal'):
                    raise ExtractionError('The model returned an incomplete response or declined extraction.')
                content = choice['message'].get('content')
                response_type = type(content).__name__
                if not isinstance(content, str) or not content.strip():
                    raise ExtractionError('The model returned an empty response or declined extraction.')
                return schema.model_validate_json(content)
            except ApiError as exc:
                status = exc.status_code
                headers = httpx.Headers(exc.headers or {})
                if status is not None and 200 <= status < 300:
                    label = 'JD' if schema is JobDescriptionData else 'Resume'
                    _diagnostic(exc, f'{label} structured parsing', status,
                                response_type=response_type,
                                json_parse_error='SDK could not decode the successful HTTP response.')
                else:
                    _diagnostic(exc, stage, status, headers.get('retry-after'))
                # Reuse the existing bounded backoff without inspecting untrusted bodies.
                delay = _retry_delay(SimpleNamespace(code=status,
                    response=SimpleNamespace(headers=headers)), attempt)
                if delay is not None and attempt + 1 < max_attempts:
                    time.sleep(delay)
                    continue
                messages = {401: 'AI service rejected the configured API key.',
                            403: 'AI service access was denied. Check local permissions.',
                            429: 'AI service quota or rate limit reached. Please retry later.',
                            504: 'AI service timed out. Please retry.'}
                raise ExtractionError(messages.get(status,
                    'AI service could not process the request. Check local AI configuration and retry.')) from None
            except ValidationError as exc:
                label = 'JD' if schema is JobDescriptionData else 'Resume'
                if not parsing or any(error['type'] == 'json_invalid' for error in exc.errors(include_input=False)):
                    _diagnostic(exc, f'{label} structured parsing', status,
                                response_type=response_type,
                                json_parse_error='Invalid JSON or response envelope in structured response.')
                else:
                    _diagnostic(exc, f'{label} schema validation', status,
                                validation_errors=safe_validation_errors(exc, schema))
                raise ExtractionError('The model response did not pass schema validation.') from None
            except ExtractionError as exc:
                _diagnostic(exc, stage, status, response_type=response_type)
                raise
            except httpx.ReadTimeout as exc:
                _diagnostic(exc, stage)
                if attempt + 1 < max_attempts:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise ExtractionError('AI service could not be reached or timed out. Please retry.') from None
            except httpx.TransportError as exc:
                _diagnostic(exc, stage)
                raise ExtractionError('AI service could not be reached or timed out. Please retry.') from None
            except Exception as exc:
                label = 'JD' if schema is JobDescriptionData else 'Resume'
                parsing = parsing or (status is not None and 200 <= status < 300)
                _diagnostic(exc, f'{label} structured parsing' if parsing else stage, status,
                            response_type=response_type,
                            json_parse_error='Unexpected structured response shape.' if parsing else None)
                raise ExtractionError('AI service returned an invalid structured response. Please retry.') from None
    except ExtractionError:
        raise
    except Exception as exc:
        _diagnostic(exc, stage)
        raise ExtractionError('AI extraction failed. Check local AI configuration and retry.') from None
    finally:
        if client is not None and capture_status in client.event_hooks['response']:
            client.event_hooks['response'].remove(capture_status)
        if owned and client is not None:
            try:
                client.close()
            except Exception as exc:
                _diagnostic(exc, stage)
    raise ExtractionError('No valid structured response was returned.')
