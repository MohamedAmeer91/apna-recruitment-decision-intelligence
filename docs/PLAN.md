# PLAN - Apna Recruitment Decision Intelligence (Capstone, 100 marks)

Phase: PLAN only. No design or implementation here. Source: `docs/REQUIREMENTS.md`.

## 1. Summary of the brief

One public GitHub repo, branch `main`, graded by reading text/CSV files (graders do not run code). A synthetic dataset (800 rows, 10 columns, 170 positives) feeds four connected Parts, all in Python 3.10+ with NumPy only for graded numbers:

| Part | Marks | Question answered | Core output |
|---|---|---|---|
| 1 | 25 | Is the 8.0-day planning mean right, how large a sample is needed, regression baseline | `part1_results.txt` |
| 2 | 30 | Who will decline an offer; pick alert threshold using INR cost | `part2_results.txt`, `part2_thresholds.csv`, `part2_knn_cv.csv` |
| 3 | 25 | Candidate segments (K-means, hierarchical, PCA) | `part3_results.txt`, `part3_segments.csv`, `part3_hierarchical.csv` |
| 4 | 20 | Split purity, voting, AdaBoost by hand, NN sizing, memo | `part4_results.txt`, 4 CSVs |

Later Parts reuse earlier data, the 560/240 split and the models. The README carries every written answer.

## 2. Deliverable / rubric to task map

Per-item mark weights inside a Part are not published; only Part totals. Each results key, CSV and README answer is scored independently, so completeness of every key matters.

| Deliverable | Marks bucket | Task(s) |
|---|---|---|
| `generate_data.py` verbatim, `data/records.csv`, hash `0537bc7f...` | gate for everything | T0 |
| `requirements.txt` (`numpy==<ver>`) | compliance | T0 |
| Part 1 keys: n, mean, sd, se, CI, z_stat, reject_h0, n_required, closed-form regression (slope, intercept, tss/ess/rss, r2, rmse, mae), GD (J0/1/10/100/500, slope, intercept, gap), residual funnel and log-log keys, multiple regression betas 0-7, mr_r2, mr_adj_r2, loglog_adj_r2 | P1 (25) | 1.1-1.7 |
| README R1.1-R1.5 | P1 | 1.1-1.7 write-up |
| Part 2 keys: split counts, 1000-step logistic losses, weights, bias, test_bce, LR effect, odds ratios, two-threshold metrics, blind_accuracy, thresholds CSV (9 rows), KNN CV CSV (11 rows), knn_* keys, cost keys | P2 (30) | 2.1-2.8 |
| README R2.1-R2.5 | P2 | 2.x write-up |
| Part 3 keys: wcss_k2..k6, elbow_k, segment sizes, segments CSV (3 rows), hierarchical CSV (15 rows), cut cluster counts, PCA eigen/ratio/loadings/projections/mean_pc1 | P3 (25) | 3.1-3.5 |
| README R3.1-R3.4 | P3 | 3.x write-up |
| Part 4 keys: root entropy/gini, splits CSV (4), best splits, stump rates, voting CSV (5), AdaBoost keys + CSV (16), oob_fraction, architecture CSV (7), mlp/cnn totals, fc/conv layer params, rnn_total | P4 (20) | 4.1-4.5 |
| README R4.1-R4.4 + `## Model-choice memo` (<=300 words) | P4 | 4.x, 4.6 |
| README structure, `## How to run`, `## Limitations` (>=3) | compliance | T-README |

## 3. Constraints and rules from the brief

Repo and submission
- Single public repo, only branch `main`; submit link to `main` (root URL only if default branch is `main`, or `/tree/main`). Verify default branch.
- Exact layout: `README.md`, `requirements.txt`, `generate_data.py`, `data/records.csv`, `part1.py`..`part4.py` (shared helper file allowed, import must not re-run a Part), `outputs/` with the 4 results files and 8 CSVs listed.
- README first line exactly `Domain: Recruitment`. Section order: How to run, Part 1-4 findings, Model-choice memo, Limitations.
- Scripts must run top to bottom on a clean clone, in README order, and regenerate the committed `outputs/` exactly.

Forbidden
- No images, screenshots, PDFs, slides, video, audio committed.
- No external links inside repo/README (no notebooks, drives, apps, second repo).
- No paid/API/downloaded data. Only stdlib + NumPy for graded numbers; no ready-made regression, KNN, k-means, PCA, metric routines (`polyfit`, `cov`, `corrcoef`, `lstsq` forbidden). Allowed: `numpy.linalg.solve` (1.7), `numpy.linalg.eigh` (3.5), array arithmetic.
- Never type a reference/checkpoint value; compute it. Do not edit generator or CSV.
- Never use or derive protected attributes (age, gender, etc.). Name segments by pipeline behaviour only.
- Money as `INR 400` / `400 Rupees`, never `$`.
- Original work; I must be able to explain every line.
- Test rows only evaluate; never fit, standardise, choose K or tau.

Number conventions
- Sample SD (n-1) everywhere unless stated; ln natural; positive = prob >= tau; stable argsort for ties; format: floats 4 dp, ints plain, yes/no, `key = value`.
- Tolerances: max(0.005, 0.0005*|ref|); confusion cells +/-1; `cost_tau` exact; `gd_slope_gap` < 0.001; `n_required` +/-1 half marks; money +/-INR 450.
- Pinned split (first floor(7n/10) of each class to train), K-fold = train position mod 5, K-means start rows 0, floor(800/K)*i, no iteration cap.

Dates
- Window 6 Nov - 16 Nov 2026 (11 days). Today is 14 Nov 2026, so only about 3 days remain (14, 15, 16). Submit before the platform closing time on 16 Nov. Suggested pace (P1 d1-3, P2 d4-6, P3 d7-8, P4+README d9-11) must be compressed (section 6).

## 4. Assumptions and open questions

Assumptions
- A1. Python >= 3.10 and a NumPy version installable locally; I pin whichever I use (brief says 1.26.4 and 2.2.6 give identical numbers).
- A2. Repo `main` already exists locally (git initialised); a GitHub remote is added later by the user, not by me now.
- A3. Mark weights within a Part are equal-ish per key; treat every key as worth getting right.
- A4. "Checkpoint" values (n=800, mean 8.9382; 560/119/240/51; loss 0.6931; wcss_k2 2262.6753; eigenvalue_sum 5; root_entropy 0.7462; round1_error 0.25) are used only as assertions comparing computed output, never as inputs.
- A5. Part 4.2 stump is the split with best information gain; K for KNN is `knn_best_k` from 2.7.
- A6. Hierarchical "no ties" so merge order is deterministic; cut rule as stated (largest gap).
- A7. Rounding done only at print time.

Open questions (for the user)
1. Which GitHub username/repo name, and will the user create and push (I must not push)? Is `main` the default branch there?
2. Which Python and NumPy versions are installed locally (to pin)?
3. Exact submission closing time on 16 Nov 2026 (the brief defers to the platform)?
4. Is the user fine with a possible extra stdlib-only helper module (e.g. `common.py`)? Brief allows it.
5. Does the user want the README memo to recommend logistic at tau from 2.8 by default, pending actual numbers? (Decision deferred until Part 4 outputs exist.)
6. The brief never defines `tau_0.3` rows' blind_accuracy beyond test set; assumed computed on the 240 test rows.
7. Ambiguity: Task 4.2 KNN "probability" ties/thresholds at 0.3 for soft vote use neighbour share; assumed as written.

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Only ~3 days left | Incomplete Parts | Compress schedule; do Parts 1-3 first (80 marks), Part 4 and README last; write README per Part immediately |
| Forbidden-function use (`polyfit`, `cov`, `lstsq`, sklearn) | Zero on those keys | Own formulas only; grep for banned calls before commit |
| Convention slips (n-1 SD, stable sort, fold index, tie-breaks, K-means start rows) | Numbers outside tolerance | Implement conventions literally; unit-assert checkpoints in a scratch test, not in graded files |
| Generator hash mismatch | All data wrong | Verify printed `800 10 170 0537bc7f...` first; commit file verbatim (watch CRLF line endings on Windows: write with newline handling that matches hash) |
| Windows line endings / `.gitattributes` altering `records.csv` or scripts | Hash differs on clean clone | Generator writes with `newline=""`; verify md5 after clone-like regeneration; add `.gitattributes` `* -text` if needed |
| K-means non-optimum and long pass counts (K=6 27 passes) | Confusion in README | Use pinned starts; quote pass counts as sanity check; state non-global-optimum in R3.1 |
| Overfit/unstable KNN CV, GD precision | Off-band | Full float64; no early stop |
| README quotes a number not from outputs | Zero for that answer | Generate README numbers by copying from committed files at the end; cross-check script |
| Accidental extras (images, links, notebooks, `__pycache__`) | Penalty/ungraded | `.gitignore` for caches; scan for URLs and binaries before final commit |
| Protected-attribute language in segment names | Violates brief | Name by experience / pay gap / responsiveness only |
| Memo over 300 words or cites uncomputed model cost | Lost marks | Word-count check; cite only own computed costs |

## 6. Work breakdown and milestones (Plan -> Design -> Implement -> Test -> Deploy)

Compressed to the 3 remaining days (14-16 Nov); the suggested 11-day pace is mapped to these slots.

### M1 Plan (14 Nov, this document)
- Deliverable: `docs/PLAN.md`.
- Exit: every rubric item mapped; open questions raised; plan committed.

### M2 Design (14 Nov)
- Define module layout: `common.py` helper (data load, split, standardise, metrics, logistic, KNN, formatting `fmt`, results writer); `part1.py`..`part4.py` import it without side effects.
- Fix key lists and CSV headers per Part as constants; define the output writer producing `key = value` and exact headers.
- Draft README skeleton with required headings in order.
- Exit: key/header checklist per Part complete; function signatures listed; no ambiguity left in conventions.

### M3 Implement (14-15 Nov)
Order by marks and dependency:
1. T0: add `generate_data.py` verbatim, run, confirm hash/print; `requirements.txt`, `.gitignore`.
2. Part 1 (1.1-1.7): stats, closed-form OLS, GD, funnel check, log-log, normal equation. Write README Part 1.
3. Common split/standardise (2.1) then Part 2 (2.2-2.8). Write README Part 2.
4. Part 3: K-means, elbow, segments, hierarchical, PCA. Write README Part 3.
5. Part 4: splits, stump/voting, AdaBoost, OOB, architecture counts, memo. Write README Part 4 and memo.
- Exit per Part: script runs cleanly end to end; all keys present; checkpoints match; README section drafted from output values.

### M4 Test (15-16 Nov)
- Checkpoint assertions (scratch, not graded): 800/8.9382, 560/119/240/51, 0.6931, 2262.6753, 5.0000, 0.7462, 0.2500.
- Consistency checks: tss=ess+rss; rates match cells; test rows never used in fit; segment sizes sum to 800; row counts 9/11/3/15/4/5/16/7; headers exact; cost_tau multiple of 0.05; gd_slope_gap < 0.001.
- Static checks: grep forbidden routines and `$`/URLs/image files; memo word count <= 300; README first line exact.
- Clean-clone test: clone to a temp folder inside the project area only, `pip install -r requirements.txt`, run scripts in README order, diff regenerated `outputs/` against committed (must be identical).
- Exit: zero diffs; all checks pass.

### M5 Deploy / Submit (16 Nov, before closing time)
- Final commit on `main`; user adds remote and pushes (I do not push); confirm default branch `main` on GitHub; confirm the repo view shows all files.
- Submit the root or `/tree/main` link on the course platform before closing.
- Exit: link resolves to `main` with complete layout; submission confirmation received.

## 7. Proposed tech stack

- Python 3.10+ (stdlib: `csv`, `math`, `hashlib`, `os`), NumPy (version pinned in `requirements.txt`).
- Plain scripts, no frameworks, no notebooks, no plotting libs (no images allowed).
- Git on `main` only; GitHub for hosting (user pushes).
- Tooling for checks only (not for graded numbers): ad hoc asserts, `diff`, `grep`, `wc -w`.

## 8. Dependencies and data needed

- Data: only `data/records.csv` from the provided generator (seed 2508); the 6x6 dissimilarity table (3.4) and the 8-point AdaBoost table (4.3) are printed in the brief and hard-coded as inputs.
- Printed constants allowed: 8.0 days, E=0.5, 1.96, tau grids, INR 50/INR 400, split thresholds (X6=1, X7=1, X3>0.5, X1<=24), network dimensions.
- Cross-Part dependencies: P2 split/standardisation -> P2 logistic -> P2 cost; P2 logistic and KNN and 2.1 split -> P4.2 voting; P3.3 segments -> P3.5 `mean_pc1_segment_*`; P1 fitted values -> 1.6 funnel.
- Environment: Python + `pip install numpy`; no network needed after install; no accounts, keys or downloads.
- External: a GitHub account/repo (user-supplied) for the final push only.
