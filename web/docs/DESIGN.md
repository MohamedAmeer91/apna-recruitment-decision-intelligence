# Web layer design

## Module layout
```
web/
  app.py                 FastAPI app, routes, error handlers, static mount
  outputs_reader.py      parse outputs/*.txt (key = value) and outputs/*.csv
  model_service.py       cached model: probabilities, curves, KNN, stump, segments, scorer
  schemas.py             pydantic request/response models
  content.py             memo and limitations text (numbers filled from outputs)
  requirements.txt  Dockerfile  README.md  .gitignore
  static/
    index.html           shell, nav, theme tokens link
    css/style.css        palette tokens, light/dark, layout, cards, tables
    js/main.js           hash router, page registry
    js/api.js            fetch wrapper with error handling
    js/charts.js         inline SVG: line, bar, scatter
    js/ui.js             helpers (el, table, kpi, formatters)
    js/pages/*.js        overview, part1..part4, score, memo, limits
  tests/                 unittest + TestClient
  docs/                  PLAN, DESIGN, DECISIONS
```
`uvicorn web.app:app` is run from the repository root; app.py puts the root on sys.path so
`import common` works.

## API schema
Errors always: `{"error": {"code": str, "message": str, "details": [..]?}}` with 404/422/503.

- GET /api/health -> `{status:"ok", rows:800, train_rows:560, test_rows:240}`
- GET /api/overview -> `{title, questions:[str], kpis:[{label,value,note}], findings:[{part,text}], synthetic_notice}`
- GET /api/part1..4 -> `{results: {key: number|string}, tables: {name: [row objects]}, notes:[str]}`
- GET /api/threshold?tau=0..1 -> `{tau,tp,fp,tn,fn,accuracy,precision,recall,f1,specificity,fpr,cost_inr,cost_flag_nobody_inr,saving_vs_nobody_inr,positives,negatives}`
- GET /api/curve -> `{auc, points:[{tau,fpr,tpr,precision,f1,cost_inr}] (101 points), cost_tau_train, default_tau}`
- GET /api/knn?k= (odd 1..21) -> `{k,is_best,best_k,tp,fp,tn,fn,accuracy,precision,recall,f1, cv:[{K,mean_cv_f1}]}`
- GET /api/segments -> `{segments:[{segment,size,mean_X1..mean_X5,outcome_rate,mean_value,label}], elbow:[{K,wcss}], pca:{explained,loadings}}`
- GET /api/segments/points -> `{points:[{x1,x2,x3,x4,x5,segment}]}` (nearest segment centroid in standardised space)
- GET /api/ensemble?tau= -> `{tau, models:[{model,tp,fp,tn,fn,precision,recall,f1,cost_inr}]}`
- POST /api/score body `{experience_months, interview_reschedules_90d, salary_gap_ratio, expected_salary_inr_thousands, profile_completeness_score, competing_offer_flag(0|1), referred_flag(0|1), tau?}` ->
  `{probability, logit, flagged, tau, contributions:[{feature,value,z,weight,contribution}], bias, segment, note}`
  Ranges: months 0-600, reschedules 0-30, gap 0-1, salary 1-1000, completeness 0-100; unknown fields rejected.
- GET /api/memo, /api/limitations -> `{title, paragraphs:[str]}`

Column to X mapping: X1 experience_months, X2 interview_reschedules_90d, X3 salary_gap_ratio,
X4 expected_salary_inr_thousands, X5 profile_completeness_score, X6 competing_offer_flag, X7 referred_flag.

## Computation
Model service calls `common.fit_split_artifacts()` once (stratified 560/240 split, standardise on train,
logistic 0.1 x 1000 steps). Test probabilities p_te; thresholding via `common.confusion/metrics/cost`.
KNN via `common.knn_cv/knn_predict/knn_proba`. Stump = X1 <= 24 with train rates (as Part 4).
Segments: centroids from part3_segments.csv means, nearest in z-space of all 800 rows (X1..X5).

## Frontend structure
Single page, hash router, each page module exports `render(root)`. State is local per page; sliders
call the API with a debounce and redraw SVG. Errors render an inline alert, never a blank page.

## Visual design (shared across the four portfolio projects)
- Neutral background, one accent (indigo 4f46e5 light, a8a4ff dark), cards with 1px border, 12px radius.
- Type: system UI stack, 16px base, tabular numerals in tables, max width 1080px, 16px gutters.
- Tokens on :root; dark via prefers-color-scheme; text/background contrast >= 4.5:1.
- Charts: inline SVG using tokens (accent for the main series, muted neutral for reference, semantic
  red/green only for cost/good), axis labels, role=img with aria-label; sliders labelled.
- Responsive: grid collapses to one column under 760px; nav scrolls horizontally.
- Accessibility: landmarks, focus rings, keyboard slider, aria-live for live results.
