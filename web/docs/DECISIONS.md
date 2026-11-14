# Decisions

1. Branch: all work on local `dev`; `main` untouched; nothing pushed.
2. Read-only layer: only outputs/ files are read and only `common.py` is imported (no part scripts), so
   graded code, data and outputs stay byte-identical.
3. Segment explorer: part3.py is not imported. Segment centroids are the means in part3_segments.csv;
   records are assigned to the nearest centroid in the standardised space of all 800 rows. K-means has
   converged, so this reproduces the committed sizes 322, 255, 223 (checked in tests).
4. Ensemble lab mirrors part4.py: KNN uses its own majority vote, logistic and stump use the slider
   threshold, soft vote averages probabilities. Default 0.3 reproduces part4_voting.csv.
5. ROC curve, AUC and cost curve use a 0.01 threshold grid on test probabilities. They are derived
   display values and labelled as such.
6. Threshold slider evaluates on test rows (as the brief reports). The cost-chosen 0.15 (selected on
   training rows) is a preset button, read from part2_results.txt.
7. Score form uses raw feature units, standardised with the training mean and SD, and validates ranges
   with pydantic (unknown fields rejected). Default alert threshold is the cost-chosen one.
8. No JS libraries: charts are small inline SVG helpers, so there is nothing to vendor.
9. Memo and limitations text are rewritten in content.py with numbers read from outputs/; the root README
   is not read.
10. Errors share one JSON shape; unknown /api paths return 404 JSON; missing outputs return 503.
11. The root .gitignore ignores any `tests/` folder, so web/.gitignore re-includes `tests/`
    (the root .gitignore is not edited).
12. KNN K restricted to odd values 1 to 21 (the cross-validated set).
13. Pinned versions: fastapi 0.115.6, uvicorn 0.34.0, httpx 0.28.1 (needed by TestClient).
