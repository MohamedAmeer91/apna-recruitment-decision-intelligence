# Apna Recruitment: interactive portfolio layer

A FastAPI backend and a no-build frontend (plain HTML, CSS, ES modules, inline SVG charts) that presents
the capstone results and lets you explore the Part 2 model. It only reads `outputs/` and imports
`common.py`; graded scripts, `data/` and `outputs/` are never modified. All data is synthetic.

## Run (Windows, from the repository root)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r web/requirements.txt
uvicorn web.app:app --port 8000
```

Linux or macOS: `source .venv/bin/activate` instead of the activate line above.
Then open `localhost:8000` in a browser. Interactive API docs are at `/docs`.

## Pages

Overview, Part 1, Part 2 Model Lab (threshold slider, confusion matrix, ROC, INR cost, KNN selector),
Part 3 Segments (segment explorer), Part 4 Ensembles (split comparison, voting lab, AdaBoost, sizing),
Score a record, Memo, Limitations.

## API

`/api/health`, `/api/overview`, `/api/part1` to `/api/part4`, `/api/memo`, `/api/limitations`,
`/api/threshold?tau=`, `/api/curve`, `/api/knn?k=`, `/api/segments`, `/api/segments/points`,
`/api/ensemble?tau=`, `POST /api/score`. Errors share one shape: `{"error": {"code", "message", "details"}}`.
Schema details are in `docs/DESIGN.md`.

## Tests

```
.venv\Scripts\python.exe -m unittest discover -s web/tests -t .
```

Tests check every endpoint, cross-check numbers against `outputs/`, input validation and static files.

## Docker

Build from the repository root (the image needs `common.py`, `data/` and `outputs/`):

```
docker build -f web/Dockerfile -t apna-recruitment-web .
docker run -p 8000:8000 apna-recruitment-web
```

## Notes

Design records are in `docs/` (PLAN, DESIGN, DECISIONS). The ROC curve, AUC and segment scatter are derived
display views, not graded numbers.
