# Web layer plan

## Purpose
An interactive portfolio view of the Apna Recruitment capstone. It presents the committed results
(outputs/) and lets a visitor explore the Part 2 model: threshold, cost, KNN K, segments, voting and
a single-record scorer. It is a read-only layer: it never writes to outputs/ or data/ and never edits
the graded scripts.

## Scope
In: FastAPI JSON API, static single-page frontend (plain HTML, CSS, ES modules, inline SVG charts),
tests, Dockerfile, run docs.
Out: authentication, databases, retraining, any change to graded files, network calls at runtime.

## Pages (hash routes of one static page)
| Route | Content |
|---|---|
| #/overview | Problem, three questions, headline numbers, key findings |
| #/part1 | Mean test, sample size, regression baseline, residual spread, multiple regression |
| #/part2 | Logistic results, odds ratios, INTERACTIVE threshold lab (confusion matrix, metrics, ROC, cost), KNN K selector |
| #/part3 | Segment explorer (table, scatter by segment, cluster picker), elbow, PCA, hierarchical |
| #/part4 | Split comparison, ensemble voting lab, AdaBoost by hand, network sizing |
| #/score | Score a candidate record with the trained logistic weights |
| #/memo | Model-choice memo |
| #/limits | Limitations |

## API endpoints (all GET unless noted, JSON)
/api/health, /api/overview, /api/part1 .. /api/part4, /api/memo, /api/limitations,
/api/threshold?tau=, /api/curve, /api/knn?k=, /api/segments, /api/segments/points,
/api/ensemble?tau=, POST /api/score.

## Data flow
outputs/*.txt|csv --(outputs_reader)--> parsed dicts --> endpoints (static facts).
data/records.csv --(common.fit_split_artifacts, same 560/240 split)--> cached Model
(probabilities, KNN, stump) --> interactive endpoints. Cache is built once at startup.

## Rules
- Reuse common.py only; importing it has no side effects.
- Derived values (ROC curve, AUC, threshold grid) are labelled as derived, not graded numbers.
- Cost rule from the brief: INR 50 per false positive, INR 400 per false negative (common.cost).
- Synthetic data banner on every page; no protected attributes are used or derived.

## Phases
Plan, Design, Implement, Test, Deploy (local). One local commit per phase on branch dev.
