"""Safe Sarvam diagnostics, importable even when the Sarvam SDK is missing."""
import json
import logging
import re


def diagnostic(exc, stage, status=None, retry_after=None, *, response_type=None,
               json_parse_error=None, validation_errors=None):
    # Match fixed excerpts only. Provider bodies and exception text may echo PII.
    body = getattr(exc, 'body', None)
    raw = json.dumps(body).casefold() if isinstance(body, (dict, list, str)) else ''
    codes = ('invalid_api_key_error', 'invalid_request_error', 'authentication_error',
             'permission_denied', 'insufficient_quota', 'rate_limit_exceeded',
             'model_not_found', 'not_found_error', 'model_overloaded')
    code = next((value for value in codes if re.search(
        r'(?<!\w)' + re.escape(value) + r'(?!\w)', raw)), None)
    phrases = ('invalid or missing authentication credentials', 'invalid api key',
               'access denied', 'forbidden', 'insufficient credits', 'rate limit exceeded',
               'invalid json schema', 'unsupported schema', 'unsupported keyword',
               'invalid response format', 'response_format', 'json_schema',
               'additionalproperties', 'anyof', 'oneof', '$defs', '$ref',
               'maximum number of tokens', 'max_tokens', 'context length exceeded')
    excerpts = [value for value in phrases if value in raw]
    message = '; '.join(excerpts) or 'Exception details withheld for privacy.'
    if isinstance(exc, ModuleNotFoundError) and exc.name == 'sarvamai':
        message = 'Sarvam SDK is not installed in the Python interpreter running the app.'
    if not isinstance(retry_after, str) or not re.fullmatch(r'\d{1,9}(?:\.\d{1,3})?', retry_after):
        retry_after = None
    fields = {'provider': 'sarvam', 'stage': stage, 'exception_type': type(exc).__name__,
              'http_status': status, 'provider_error_code': code, 'message': message,
              'retry_after': retry_after}
    if response_type is not None:
        fields['response_type'] = response_type
    if json_parse_error is not None:
        fields['json_parse_error'] = json_parse_error
    if validation_errors is not None:
        fields['validation_errors'] = validation_errors
    logging.getLogger(__name__).warning('[SARVAM ERROR]\n%s', '\n'.join(
        f'{name}={value if isinstance(value, str) else json.dumps(value)}'
        for name, value in fields.items()))


def safe_validation_errors(exc, schema):
    """Only schema-defined field names and fixed messages; no inputs/context/URLs."""
    field_names = set()
    def collect(node):
        if isinstance(node, dict):
            field_names.update(node.get('properties', {}))
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)
    collect(schema.model_json_schema())
    messages = {
        'missing': 'Field required.', 'extra_forbidden': 'Extra inputs are not permitted.',
        'string_type': 'Input should be a valid string.',
        'bool_type': 'Input should be a valid boolean.',
        'float_type': 'Input should be a valid number.',
        'list_type': 'Input should be a valid list.',
        'model_type': 'Input should be a valid object.',
        'model_attributes_type': 'Input should be a valid object.',
        'literal_error': 'Input must match an allowed value.',
        'string_too_short': 'String is shorter than the allowed minimum.',
        'greater_than_equal': 'Number is below the allowed minimum.',
        'less_than_equal': 'Number exceeds the allowed maximum.',
        'finite_number': 'Input should be a finite number.',
        'value_error': 'Value failed schema validation.',
    }
    return [{'field': '.'.join(part if part in field_names else '<unknown>'
                              for part in error['loc'] if isinstance(part, str)) or '<root>',
             'message': messages.get(error['type'], 'Value does not match the field schema.')}
            for error in exc.errors(include_input=False, include_context=False, include_url=False)]
