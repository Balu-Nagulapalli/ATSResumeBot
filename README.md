# ATS Resume Analyzer V2

A fresh Python/Streamlit implementation: document extraction → validated structured facts → evidence matching → deterministic Python scoring. Supports one JD (PDF/DOCX/TXT or pasted text) and 1–10 resumes (PDF/DOCX).

## Run locally

Python 3.11+ is required. In PowerShell:

```powershell
cd D:\AI-ChatBot\ats-analyzer-v2
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env locally to select AI_PROVIDER and configure that provider's key/model.
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Do not overwrite an existing `.env`. No key is included, displayed, or logged. Process environment values take precedence over `.env`, which is loaded from the project directory regardless of the working directory. Missing credentials show a useful UI message and disable analysis. Restart Streamlit after changing `.env`.

Select structured extraction with `AI_PROVIDER=gemini` (default) or `AI_PROVIDER=sarvam`. Unsupported names fail clearly. Only the selected provider's key is required. Set `GEMINI_API_KEY` and `GEMINI_MODEL` for Gemini, or `SARVAM_API_KEY` and `SARVAM_MODEL=sarvam-105b` for Sarvam. Restart Streamlit after configuration changes.

Gemini retains the official `google-genai` SDK and [JSON Schema output](https://ai.google.dev/gemini-api/docs/generate-content/structured-output). Sarvam uses HTTPX with the supported [v1 chat completion API](https://docs.sarvam.ai/api-reference/chat/chat-completions-v1), `response_format=json_schema`, and strict output. Both use the same existing `JobDescriptionData` / `ResumeData` models, extraction prompt, and source-evidence validation. Neither provider performs matching or scoring; those stages remain local.

Requests use temperature zero, a 60-second timeout, and at most three total attempts for HTTP 429/500/502/503/504. Backoff is 2s then 4s; valid Retry-After values are honored up to 60s. Longer cooldowns stop retries. Gemini also detects exhausted daily quotas. Authentication, bad-request, schema, and source-evidence failures are not retried. Diagnostics omit arbitrary provider messages, document text, and credentials.

Validated extraction JSON is stored under `.cache/<provider>/<model-and-version-hash>/jd/` or `resumes/`, keyed by the normalized document hash. Both memory and disk identities include provider, model, extraction version, and a fingerprint of the actual schemas. Increment `EXTRACTION_VERSION` when extraction instructions change. Legacy unversioned caches are left untouched and are not reused because their model/version provenance cannot be established. The first Sarvam extraction is a miss even if Gemini already cached the document; subsequent unchanged extractions can hit across restarts. Cache files contain structured evidence, which may include personal information, but never full raw documents, secrets, expected scores, or final scores.

## What is implemented

- In-memory PyMuPDF and python-docx extraction, including DOCX tables and header/footer paragraphs. UTF-8 TXT has a Windows-1252 fallback. Limits: 10 MB/file, 100 PDF pages, 80,000 extracted characters, 40 MB expanded DOCX. OCR and password-protected PDFs are not supported.
- Multi-signal document-type validation and normalized-hash JD/resume equality rejection. Validation is heuristic; unusual documents may need clearer section headings.
- Strict Pydantic schemas with source quotations and unknown/null fields. Quotes must occur in the document; technology labels must appear in their quotes directly or through a known alias. Missing or invalid evidence fails extraction rather than receiving credit.
- One successful extraction per distinct JD/resume per session, keyed by SHA-256 of normalized text (preserving punctuation), document kind, and model. Failed extractions are not cached. Terminal logs report only JD/Resume HIT/MISS. Matching makes no API calls. Refusals and incomplete outputs return safe errors.
- Exact and alias technology matching, plus local TF-IDF/cosine comparison for responsibilities, roles and domains. Related technologies do not satisfy an exact requirement. The legacy semantic flag cannot enable cloud matching; the compatibility wrapper preserves locally computed evidence.
- Explicit years or merged employment date ranges. Overlapping jobs are counted once. An explicit total takes precedence over dated history, with a warning if they disagree. `Present` uses the visible analysis date; year-only dates use conservative January boundaries. Total experience and role relevance are separate criteria.
- Degree level and field scored separately. Explicitly allowed related computing fields are supported. Multiple listed degrees are acceptable alternatives. Ambiguous or compound degree requirements should be reviewed in structured extraction.
- Basic resume completeness, gap-derived feedback, prioritized learning steps, per-candidate evidence, ranking and a horizontal comparison chart.
- Each resume can fail independently. Completed results and structured extractions are cached so rerenders and repeated analysis do not repeat requests. Changed inputs hide stale results.

## Scoring and uncertainty

The LLM never returns the ATS score. `services/scorer.py` calculates the weighted sum of criterion scores. Initial weights in `config.py` sum to 1:

| Criterion | Weight |
|---|---:|
| Required skills | 32% |
| Preferred skills | 8% |
| Tools | 10% |
| Experience | 15% |
| Role relevance | 10% |
| Responsibilities | 10% |
| Education | 6% |
| Certifications | 3% |
| Domain | 2% |
| Resume quality | 4% |

Non-applicable categories receive no contribution. Remaining weights are normalized; resume quality is capped at 5% of the effective total. If the JD provides no job criteria, the result is labeled insufficient and receives no meaningful match score. Skill duplicates are excluded across categories. Required tools have twice the within-category weight of preferred tools.

Exact/equivalent evidence receives full credit. Partial responsibility/title/domain evidence receives half; related technology experience receives no skill credit. Degree level and field each contribute half. Experience credit is `min(candidate_years / minimum_years, 1)`; unknown experience receives zero with an uncertainty notice. Exceeding a stated maximum is not penalized.

Scores are deterministic for the same structured facts, weights, and analysis date. LLM extraction may vary across new sessions, even with temperature zero. Confidence is a heuristic evidence-quality signal, not a calibrated probability. Lexical fallback uses conservative TF-IDF overlap and may undercount paraphrases. A score is a document-evidence comparison, not a hiring decision, an assessment of personal worth, or a claim to reproduce a proprietary ATS.

## Benchmarks and calibration

`benchmarks/example.json` contains the supplied illustrative 95% case in text form. It is not a calibration dataset or a guaranteed output. In particular, Git, code-review, security and cross-functional evidence are not all explicit in that example; the system should report those gaps rather than manufacture a 95.

```powershell
.\.venv\Scripts\python.exe -m services.benchmark benchmarks/example.json --as-of 2026-01-01
# Optional table export (contains case numbers/scores only):
.\.venv\Scripts\python.exe -m services.benchmark benchmarks/example.json --output benchmark-results.csv
```

Input is a JSON list of `{ "jd": "raw text", "resume": "raw text", "expected_score": 95 }`. Prediction receives only JD/resume text. Expected scores are read for error calculation after prediction. Output includes Expected, Predicted, Absolute Error and MAE. Live evaluation requires your key and incurs API usage. Tests inject predictors to verify label isolation without an API call.

Weights are initial design choices and **have not been calibrated against a labeled dataset**. For calibration, collect diverse role families and match strengths, freeze extraction results/model/date, split cases by JD or role family into training and held-out sets, adjust a small set of weights on training MAE, and report held-out MAE against the original weights. Do not optimize to one example. No candidate-name or filename rules exist. Automatic weight fitting is deliberately omitted for the single illustrative case.

## Privacy and operational limits

The app sends extracted text (including personal information present in resumes) to the configured provider when Analyze is selected and no validated extraction is cached. It does not write uploaded files or full document logs. Structured extractions are retained in the local persistent cache described above and in the owning session. Clear documents and results removes current-session data; persistent cache entries remain. Provider-side data handling depends on the selected provider and your account terms. The app does not promise zero provider-side retention.

The app is intended for local use. There is no database, authentication, Docker, external backend, OCR, vector database, or separate frontend. Visual layout and multi-column reading-order fidelity cannot be fully assessed from extracted text. Quotation validation checks provenance, not perfect semantic entailment; review disputed evidence. Text in documents is treated as untrusted in prompts.

## Tests

```powershell
.\.venv\Scripts\python.exe -m compileall .
.\.venv\Scripts\python.exe -m pytest -q
```

Tests exercise real PDF/DOCX parsing, malformed/empty/scanned documents, normalization, evidence checks, required/preferred coverage, experience overlap and uncertainty, education, non-applicable weights, high/medium/unrelated candidates, deterministic score limits, label isolation, mocked provider behavior, provider/model/version cache isolation, and Streamlit rendering. They do not contact either provider or require real keys.

## Module map

`parsers/` reads and validates documents; `models/` defines schemas; `services/extraction.py` owns the API boundary; JD/resume extractors validate document types; `normalizer.py`, `experience.py` and `matcher.py` implement deterministic matching; `semantic.py` provides a local-only compatibility wrapper; `scorer.py` calculates scores; `feedback.py` derives suggestions; `pipeline.py` coordinates caching; `benchmark.py` evaluates labeled cases; `app.py` renders the UI.

## Extraction call budget

For a shared Streamlit session (or an explicitly reused `cache` mapping in Python), the first JD/resume pair costs two extraction requests. A new resume against the cached JD costs one; repeating the same pair costs zero. One JD and N unique resumes cost N+1 successful extraction requests. Transient failures can add retry attempts; this is not an absolute HTTP-request cap. A new/reset session needs fresh extraction. No live connectivity probes or semantic comparison calls run during matching. Quota exhaustion is not repaired by caching: new documents still need available provider quota.
