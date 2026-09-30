"""Evaluate unseen labels only after prediction. No automatic fitting to a tiny benchmark."""
import argparse
import json
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
import pandas as pd
from pydantic import Field
from models.schemas import StrictModel
from services.pipeline import analyze_pair


class BenchmarkCase(StrictModel):
    jd: str = Field(min_length=1)
    resume: str = Field(min_length=1)
    expected_score: float = Field(ge=0, le=100)


def evaluate_benchmarks(cases: Iterable[BenchmarkCase | dict],
                        predictor: Callable[[str, str], float] | None = None,
                        *, as_of: date | None = None) -> tuple[pd.DataFrame, float]:
    cache = {}
    if predictor is None:
        predictor = lambda jd, resume: analyze_pair(jd, resume, cache=cache, as_of=as_of)['score'].overall
    rows = []
    for index, raw in enumerate(cases, 1):
        case = BenchmarkCase.model_validate(raw)
        # Expected score is never passed to prediction, extraction, or semantic review.
        predicted = float(predictor(case.jd, case.resume))
        if not 0 <= predicted <= 100:
            raise ValueError('Prediction must be a finite score between 0 and 100.')
        rows.append({'Case': index, 'Expected': case.expected_score, 'Predicted': predicted,
                     'Absolute Error': abs(case.expected_score - predicted)})
    frame = pd.DataFrame(rows, columns=['Case', 'Expected', 'Predicted', 'Absolute Error'])
    if frame.empty:
        raise ValueError('Provide at least one benchmark case.')
    return frame, float(frame['Absolute Error'].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate ATS benchmark JSON without label leakage.')
    parser.add_argument('file', type=Path)
    parser.add_argument('--as-of', type=date.fromisoformat, default=date.today())
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        cases = json.loads(args.file.read_text(encoding='utf-8'))
        table, mae = evaluate_benchmarks(cases, as_of=args.as_of)
    except Exception as exc:
        # Do not print arbitrary SDK exception bodies or document-containing validation errors.
        from services.extraction import ExtractionError
        message = str(exc) if isinstance(exc, ExtractionError) else 'Benchmark evaluation failed. Check file format and document validity.'
        parser.exit(1, message + '\n')
    print(table.to_string(index=False))
    print(f'MAE: {mae:.3f}')
    if args.output:
        table.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
