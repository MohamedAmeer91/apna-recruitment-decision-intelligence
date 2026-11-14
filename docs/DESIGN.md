# DESIGN - Apna Recruitment Decision Intelligence

Phase: DESIGN only. Source: `docs/REQUIREMENTS.md` (brief), `docs/PLAN.md`. Signatures and pseudocode only; no implementation.

## 1. Repository layout (exactly per the brief, plus helpers)

```
apna-recruitment/
├── README.md            # line 1: Domain: Recruitment
├── requirements.txt     # numpy==<installed version in .venv>   (only line unless extra pkgs)
├── generate_data.py     # brief script, verbatim (never edited)
├── data/records.csv     # generator output, committed
├── common.py            # shared helpers (no side effects on import)
├── part1.py part2.py part3.py part4.py
├── outputs/
│   ├── part1_results.txt part2_results.txt part3_results.txt part4_results.txt
│   ├── part2_thresholds.csv part2_knn_cv.csv
│   ├── part3_segments.csv part3_hierarchical.csv
│   └── part4_splits.csv part4_voting.csv part4_adaboost.csv part4_architecture.csv
├── docs/ (REQUIREMENTS.md PLAN.md DESIGN.md)   # allowed; no links, no images
├── tests/test_checks.py  # scratch-quality checks, stdlib unittest; not graded
├── .gitattributes .gitignore
```
No images/PDFs/notebooks. `README.md` and docs contain no URLs. `.venv/` and `__pycache__/` are ignored.

## 2. Data generator and hash stability

- `generate_data.py` is pasted verbatim. Deterministic: `random.Random(2508)`, Box-Muller, `N=800`; no NumPy involved. Columns/dtypes in the file: `record_id` str (`R0001`..`R0800`), `experience_months` float (1 dp), `interview_reschedules_90d` int, `salary_gap_ratio` float (3 dp), `expected_salary_inr_thousands` float (1 dp), `profile_completeness_score` float (1 dp), `competing_offer_flag` int, `referred_flag` int, `offer_declined` int, `days_to_hire` float (2 dp). Expected print: `800 10 170 0537bc7fb4a044719831c43166e79d3c`.
- **Line-ending finding (needs verification at implementation).** `csv.writer` defaults to `lineterminator="\r\n"` and the script opens with `newline=""`, so the file written on any OS contains CRLF row endings, and the MD5 is over those bytes. The current `.gitattributes` forces `eol=lf` on `*.csv`; on checkout that would rewrite `records.csv` to LF and change the hash if the reference hash is of the CRLF file. Rule: run the generator, compare MD5 with the brief. If it matches with CRLF bytes, change `.gitattributes` to `data/records.csv -text` and `generate_data.py -text` (keep LF for other text), then `git add --renormalize`. Verify the committed blob with `git cat-file blob HEAD:data/records.csv | md5sum`.
- Other `outputs/*.csv` and `*.txt` are written by my code with `newline="\n"` (explicit `open(..., "w", newline="\n")`), so they are LF on every OS and regenerate byte-identically.
- Python 3.10 float formatting (`round`, `math`) is platform-independent for this generator, so no further hash risk. Data is read in `common.load_data()` with `csv.reader` into a float matrix (`np.float64`); `record_id` dropped.

## 3. Module and function layout

`common.py` (importing runs nothing):

```python
ROOT: Path; DATA_PATH; OUT_DIR
def load_data() -> dict            # X (800x7), y (800,) int, value (800,), names
def fmt(v) -> str                  # float -> f"{v:.4f}", int -> str, bool -> yes/no, str as-is
def write_results(path, pairs: list[tuple[str, object]]) -> None   # "key = value" lines, LF
def write_csv(path, header: list[str], rows: list[list]) -> None   # exact header, floats 4dp
def sample_sd(a) -> float; def sample_sd_cols(A) -> array         # ddof 1 by hand
def ols_closed_form(x, y) -> dict  # slope, intercept, fitted
def stratified_split(y) -> (train_idx, test_idx)                   # floor(7*cnt/10), file order
def standardise(Xtr, Xte) -> (Ztr, Zte, mean, sd)                  # train stats only
def sigmoid(z); def bce(y, p)                                      # clip not applied; p never exactly 0/1 here
def train_logistic(Z, y, eta, steps, record=()) -> (w, b, {epoch: loss})
def predict_proba(Z, w, b)
def confusion(y, pred) -> (tp, fp, tn, fn)
def metrics(tp, fp, tn, fn) -> dict   # accuracy, precision, recall, f1, specificity, fpr (0 if denom 0)
def knn_proba(Zref, yref, Zq, k) -> array    # share of positive among k; stable argsort on (dist, index)
def knn_predict(Zref, yref, Zq, k) -> array  # share*k >= (k+1)/2, integer compare: pos_count >= (k+1)//2
def fit_split_artifacts() -> dict            # cached split, standardised sets, logistic w,b (deterministic recompute; used by part2 and part4)
def cost(fp, fn) -> int                      # 50*fp + 400*fn
```
Each `partN.py` has `def main()` guarded by `if __name__ == "__main__"`, so `part4` can import from `common` (and nothing from `part2` except via `common`). Part 4 recomputes Part 2's model by calling `common.fit_split_artifacts()` (same code, same seed-free math, identical numbers); Part 3 `mean_pc1_segment_*` uses its own Task 3.3 segments (same script).

Forbidden calls (static grep in tests): `polyfit`, `np.cov`, `corrcoef`, `lstsq`, `sklearn`, `pandas`, `scipy`, `np.std`/`np.var` with ddof shortcuts are avoided too (SD by explicit formula). No pandas/sklearn/scipy at all; `requirements.txt` has only the `numpy==` line.

## 4. Part designs

Notation: n=800, X4 = col index 3, value = days_to_hire. All writes at 4 dp through `fmt`.

### Part 1 (`outputs/part1_results.txt`, all 800 rows)
Key order as written:
1. `n, mean_value, sd_value, se_value, ci95_low, ci95_high`: mean = sum/n; sd = sqrt(sum((y-m)^2)/(n-1)); se = sd/sqrt(n); ci = mean -/+ 1.96*se. Assert (scratch) n=800, mean 8.9382.
2. `z_stat = (mean-8.0)/se`, `reject_h0 = abs(z)>1.96` -> yes/no.
3. `n_required = ceil((1.96*sd/0.5)^2)` using `math.ceil` on the float (smallest integer n >= value).
4. Closed-form: `reg_slope = Sxy/Sxx` with divisor n-1 both (equivalently sum products / sum squares), `reg_intercept`, `tss, ess, rss, tss_equals_ess_plus_rss` (|tss-ess-rss| < 1e-6*tss), `r2 = 1-rss/tss, rmse = sqrt(rss/n), mae`.
5. GD: z = (X4-mean)/sd; m=c=0; loop 500 updates, eta 0.1, full batch; cost J=mean(e^2) recorded at 0,1,10,100,500 -> `gd_J0, gd_J1, gd_J10, gd_J100, gd_J500`; `gd_slope = m/sd_x4`, `gd_intercept = c - m*mean/sd`, `gd_slope_gap = abs(gd_slope - reg_slope)` (< 0.001 check).
6. Funnel: `order = np.argsort(fitted, kind="stable")`; lower = order[:400], upper = order[400:]; residual sample SDs -> `resid_sd_lower_raw, resid_sd_upper_raw, resid_sd_ratio_raw`. Log-log: `ln y` on `ln X4` via same closed form -> `loglog_intercept, loglog_slope, loglog_r2, loglog_rmse`, then `resid_sd_lower_log, resid_sd_upper_log, resid_sd_ratio_log` (stable sort on its own fitted values).
7. Normal equation: design = [1, lnX4, X1, X2, X3, X5, X6, X7] (columns: indices 3->ln, 0,1,2,4,5,6); `beta = np.linalg.solve(A.T@A, A.T@lny)`; keys `mr_beta_0..mr_beta_7, mr_r2, mr_adj_r2` (p=7), `loglog_adj_r2` (p=1). R2 = 1 - rss/tss on ln scale.

README R1.1-R1.5 filled by copying values from the file (a `tests/` cross-check parses README numbers against outputs).

### Part 2 (`part2_results.txt`, `part2_thresholds.csv`, `part2_knn_cv.csv`)
- 2.1 `stratified_split`: idx0 = where y==0, idx1 = where y==1; train = first (7*len)//10 of each; merge and sort ascending (file order). Keys `train_rows, train_positives, test_rows, test_positives` (560/119/240/51). Standardise X1..X7 with train mean and sample SD; apply to test.
- 2.2 `train_logistic(Ztr, ytr, 0.1, 1000, record=(0,1,10,100,1000))`: w=0,b=0; each step p=sigmoid(Z@w+b); gw=Z.T@(p-y)/n, gb=mean(p-y). Keys `loss_epoch_0,1,10,100,1000`, `weight_X1..X7`, `bias`, `test_bce` (BCE on test rows, final params). Loss at epoch 0 = ln2 assertion.
- 2.3 three restarts, 200 steps: `loss_epoch_200_eta_0.01`, `..._0.1`, `..._1.0`.
- 2.4 `odds_ratio_Xj = exp(weight_Xj)`.
- 2.5 for tau 0.5 and 0.3: predict = p >= tau; keys `tau_{t}_tp,fp,tn,fn,accuracy,precision,recall,f1,specificity` (tau labelled `0.5`, `0.3`); `blind_accuracy = (240-51)/240` (test negatives / test rows).
- 2.6 tau = 0.1..0.9 (use `k/10` for k=1..9 to avoid float drift); CSV header `tau,tp,fp,tn,fn,tpr,fpr`, 9 rows; tau printed 4 dp (e.g. 0.1000).
- 2.7 KNN: Euclidean distance to reference rows (vectorised squared distance; ranking by squared distance is monotone-equivalent); stable argsort gives (distance, file position) tie-break. CV: fold = train position % 5; for each fold held-out = fold rows, reference = other four folds in file order; features from 2.1 (no refit); K in 1,3,...,21; per fold F1 (0 if undefined), mean over 5 -> `part2_knn_cv.csv` header `K,mean_cv_f1`, 11 rows; `knn_best_k = first K with max mean` (strict `>` loop, so ties go small). Test with all 560 train as reference: `knn_tp,fp,tn,fn,accuracy,precision,recall,f1`.
- 2.8 For k=1..19, tau=k/20 on training probabilities, cost 50*fp+400*fn; keep strictly lower cost only (ties -> smaller tau). Compare tau as `k/20` integer-index to guarantee `cost_tau` exact multiple of 0.05. Keys `cost_tau, cost_train_at_tau`, then test: `cost_test_at_tau, cost_test_at_0.5, cost_test_flag_nobody = 400*test_positives`. Money printed as integers (INR amounts are integers).

README R2.1-R2.5 as listed in the brief; INR formatting in prose.

### Part 3 (`part3_results.txt`, `part3_segments.csv`, `part3_hierarchical.csv`)
- 3.1 Z = standardise X1..X5 over all 800 (sample SD). `kmeans(Z, K)`: init centroids = Z[[i*(800//K) for i in range(K)]]; loop: d2 = squared distances (800xK); assign = `argmin` (first index on ties = lower cluster); stop on first pass with unchanged labels (no cap; count passes for a scratch check: K=4 21, K=5 15, K=6 27); else centroid = mean of members, empty cluster keeps its old centroid. `wcss = sum(d2 to final centroids)` where final centroids are those used in the final (no-change) assignment pass. Keys `wcss_k2..wcss_k6` (assert k2 = 2262.6753).
- 3.2 `elbow_k = min K in {3,4,5} with W(K)-W(K+1) < 0.5*(W(K-1)-W(K))`; if none, the brief does not define a value: write the text rule result literally (fallback documented: `none` is not expected to occur on the reference data; the implementation asserts a value is found, and if not found, the open question goes to the user). Proceed with K=3 regardless.
- 3.3 Take the K=3 run; cluster means of original X3 sorted ascending -> segments A,B,C. Keys `size_segment_A/B/C`; CSV header `segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value`, rows A,B,C (means in original units, outcome rate = mean of outcome, mean value of days).
- 3.4 `agglomerate(D, linkage)`: clusters as sorted tuples of 0-based ids; each step compute linkage distance between each cluster pair from the original matrix (single=min, complete=max, average=mean of all member pairs); merge smallest (loop order by cluster min-member ensures deterministic choice; the brief says no ties). Name `E{i+1}` joined with `+`; `cluster_a` = cluster with smaller min member. Rows: single, complete, average x merges 1..5; CSV header `linkage,merge_no,cluster_a,cluster_b,height`. Cut: gaps = diffs of 5 heights, `j = argmax(gaps)` (first on ties); merges below the cut = j+1; clusters = 6-(j+1). Keys `single_clusters_at_cut, complete_clusters_at_cut, average_clusters_at_cut`. The 6x6 table is hard-coded in `part3.py`.
- 3.5 PCA: mean (zeros, asserted ~1e-12), centre, `S = Xc.T@Xc/(n-1)`, `eigh`, reorder descending, sign flip so largest-|entry| is positive (first index on tie), W = top 2 vectors, `Zp = Xc@W`. Keys in the order: `eigenvalue_sum, eigenvalue_1..5, explained_ratio_1..5, cumulative_ratio_1..5, components_for_80pct, pc1_loading_X1..5, pc2_loading_X1..5, row1_pc1,row1_pc2,row2_pc1,row2_pc2,row3_pc1,row3_pc2, mean_pc1_segment_A/B/C` (assert sum = 5.0000).

### Part 4 (`part4_results.txt` + 4 CSVs)
- 4.1 On 560 training rows (original units): `entropy(P,n)`, `gini(P,n)` with 0 for pure/empty node; splits S1 X6==1, S2 X7==1, S3 X3>0.5, S4 X1<=24; info_gain = H_root - sum(share*H_child); gini_gain likewise. Keys `root_entropy, root_gini`, CSV header `split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain` (4 rows), `best_split_by_info_gain`, `best_split_by_gini_gain` (argmax, first on tie). Assert root_entropy = 0.7462.
- 4.2 Stump = best info-gain split; `stump_rate_yes/no` = training positive share on each side. Test predictions: logistic p>=0.3; knn proba = share, predicted positive by (K+1)/2 rule (the brief's KNN prediction); stump p>=0.3; hard_vote = at least 2 of the 3 binary predictions positive; soft_vote = mean of three probabilities >= 0.3 (logistic p, KNN share, stump p). CSV `model,tp,fp,tn,fn,precision,recall,f1`, rows in order logistic, knn, stump, hard_vote, soft_vote. Floating compare for soft vote uses `>= 0.3 - 1e-12` to avoid representation flips (documented; the brief tolerates +/-1 cell).
- 4.3 Hard-coded y, h1, h2. Pseudocode: w=1/8; for round (h): eps = sum(w[h!=y]); alpha = 0.5*ln((1-eps)/eps); u = w*exp(-alpha*y*h); w_new = u/sum(u). Rows `round,point,weight_before,weight_unnormalised,weight_after` (16). Keys `round1_error, round1_alpha, round2_error, round2_alpha, ensemble_accuracy = mean(sign(a1*h1+a2*h2)==y)` (sign 0 counted wrong if it occurs; flagged by assert). Assert round1_error = 0.25. R4.3 needs total after-weight of the two misclassified points (read from the CSV).
- 4.4 `oob_fraction = (1-1/560)**560`.
- 4.5 Counts (parameters include biases): mlp_hidden1 = 784*16+16; hidden2 = 16*16+16; output = 16*10+10; cnn_conv1 = (5*5*1+1)*6; conv2 = (5*5*6+1)*16; spatial: 28 -> conv1 (28-5+4)/1+1 = 28 -> pool 14 -> conv2 (14-5)/1+1 = 10 -> pool 5, flatten 5*5*16 = 400; cnn_dense1 = 400*32+32; cnn_dense2 = 32*10+10. Computed via a `conv_out(size, k, pad, stride)` helper and parameter functions `dense_params(i,o)`, `conv_params(kh,kw,cin,cout)`; shapes recorded for README R4.4 table (shapes: 28x28x6, 14x14x6, 10x10x16, 5x5x16, 400, 32, 10). `fc_layer_params = (128*128*3)*1024+1024`, `conv_layer_params = (3*3*3+1)*64`. RNN: `rnn_total = 128*64 + 128*128 + 16*128 + 128 + 16` (W_xh 128x64, W_hh 128x128, W_hy 16x128, b_h, b_y). Keys: `mlp_total, cnn_total, fc_layer_params, conv_layer_params, rnn_total`. Everything computed from the dimension constants, not typed totals. CSV header `layer,parameters`, 7 rows in the brief's order.

## 5. Output formatting spec
`write_results` emits `key = value` per line, LF, trailing newline. Floats `f"{x:.4f}"` (negative zero normalised to `0.0000`), ints plain (counts, `n`, `n_required`, `K`, money), yes/no booleans, text as-is (`S1`..). Key order follows the task text order. CSV via `write_csv`: header exact, rows floats 4 dp, ints plain, text as-is.

## 6. Model-choice memo structure (<= 300 words, `## Model-choice memo`)
1. To/From/Subject line (1 line).
2. Recommendation: one Task 4.2 model (decided from `part4_voting.csv` f1/recall and the cost table; default expectation: logistic at the Task 2.8 tau) and the threshold.
3. Evidence: >= 3 numbers from `outputs/` (e.g. recall and f1 at the chosen tau, `cost_test_at_tau`, `cost_test_at_0.5`, `cost_test_flag_nobody`) in INR.
4. Cost: logistic cost from Task 2.8 as comparator always; any other model's cost only if computed with `50*fp+400*fn` from its voting-CSV cells (computed in a scratch check from committed cells), otherwise state "not computed".
5. One limitation (51 test positives, single split, synthetic data).
6. One sentence: Model B (resume text, 16 job families, main use case) vs Model A (digit images, second use case) is worth building first after structured model is in production; justify by number of parameters/use case.
Word count verified by script (`len(text.split()) <= 300`, counted excluding the heading).

## 7. README outline (in the required order)
1. `Domain: Recruitment` (exact first line, nothing before it).
2. Short one-paragraph description (synthetic-data disclaimer, no protected attributes).
3. `## How to run`: `python -m venv .venv`; activate; `pip install -r requirements.txt`; `python generate_data.py` (expect `800 10 170 0537bc7fb4a044719831c43166e79d3c`); `python part1.py`; `python part2.py`; `python part3.py`; `python part4.py`.
4. `## Part 1 findings` R1.1-R1.5.
5. `## Part 2 findings` R2.1-R2.5.
6. `## Part 3 findings` R3.1-R3.4 (note PCA mean vector is zero after standardising).
7. `## Part 4 findings` R4.1-R4.4 (layer table, W shapes, activations).
8. `## Model-choice memo`.
9. `## Limitations` (>= 3: synthetic data, single pinned split and 51 test positives, K-means non-global start, normal-approximation CI, no calibration/fairness check).
Rules: INR spelling only, no `$`, no URLs, every quoted number copied from the committed `outputs/` (script `tests/check_readme.py` scans README numbers vs outputs).

## 8. Numerical and reproducibility considerations
- Python 3.10, NumPy 1.26.4 in `.venv`; `requirements.txt` = `numpy==1.26.4` (confirm `pip freeze` in the venv; pin whatever is installed). Brief says results are identical on 1.26.4 and 2.2.6.
- All arrays `float64`; no random numbers anywhere in Parts 1-4 (no seeds beyond the generator's `random.Random(2508)`).
- Sample SD is hand-coded with divisor n-1; no `np.std`/`np.var`/`np.cov` shortcuts.
- Stable sorts for all tie-prone orderings (`kind="stable"` in 1.6 and KNN); `argmin` for K-means ties (first index = lowest cluster).
- Round only at print time. Threshold comparisons `p >= tau` use `tau = k/20` or `k/10` computed as division of integers; the soft-vote 0.3 comparison tolerance documented above.
- BCE: no clipping needed (weights stay moderate; asserted p in (0,1)); eta=1.0 run still finite over 200 steps (assert finite).
- Run time is negligible (KNN CV 11 K values x 5 folds on 560 rows; K-means to convergence; no iteration cap but a debug guard raising after 10,000 passes to detect infinite loops).
- `.gitattributes` / CRLF rule from section 2 decides byte-identical regeneration; scripts run from the repo root with paths built from `Path(__file__).parent`.

## 9. Test strategy
Layers (stdlib `unittest` in `tests/`, plus one `scripts/check_all` style runner within `tests/`):
1. Unit tests: `fmt`, `confusion/metrics` (zero denominators), `stratified_split` on a toy vector, `sigmoid`/`bce` against hand values, `knn_predict` tie behaviour (stable order), `kmeans` on a 2-cluster toy set, `agglomerate` on a 4-point toy matrix, `entropy/gini` of known nodes, `conv_out`, `dense_params`, `conv_params`.
2. Checkpoint assertions (compute-then-compare, never typed into code paths): n=800 and mean 8.9382; 560/119/240/51; loss_epoch_0 0.6931; wcss_k2 2262.6753; eigenvalue_sum 5.0000; root_entropy 0.7462; round1_error 0.2500; K-means pass counts 21/15/27 for K=4/5/6; generator print line.
3. Per-requirement checks (tied to brief):
   - File presence: all 4 results files and 8 CSVs; no extra files in `outputs/`; no images/PDF in repo.
   - Header exactness and row counts: thresholds 9, knn_cv 11, segments 3, hierarchical 15, splits 4, voting 5, adaboost 16, architecture 7.
   - Keys: every key named in Tasks 1.1-1.7, 2.1-2.8, 3.1-3.5, 4.1-4.5 present exactly once in order (key lists are constants in `tests/expected_keys.py`, derived from the brief text, kept separate from the code under test).
   - Format: floats 4 dp, yes/no, `key = value` regex per line.
   - Invariants: tss = ess + rss; rates match cell counts; `gd_slope_gap < 0.001`; `cost_tau*20` integer; segment sizes sum to 800; adaboost weight_after sums to 1 per round; cumulative_ratio_5 = 1; test rows excluded from every fit (assert statistics equal those recomputed on train only).
   - Static: grep for forbidden routines (`polyfit`, `np.cov`, `corrcoef`, `lstsq`, `sklearn`, `pandas`, `scipy`), `$`, `http`, `.png/.jpg/.pdf`; memo word count; README first line and heading order; protected-attribute words absent in segment names.
   - README numbers: each number cited in R-answers found in the corresponding outputs file.
4. Determinism: run each Part twice and byte-compare outputs; clean-clone run (see section 10) byte-compared with committed files.

Final rubric checklist (tick before submit): generator hash line; requirements pin; layout; Part 1 keys (1.1-1.7) and R1.1-R1.5; Part 2 keys, 2 CSVs and R2.1-R2.5; Part 3 keys, 2 CSVs and R3.1-R3.4; Part 4 keys, 4 CSVs, R4.1-R4.4 and memo (<=300 words, >=3 numbers, tau and INR cost, logistic comparator, limitation, Model A/B sentence); README order and Limitations (>=3); no forbidden content; branch `main` only.

## 10. Deploy checklist (what "deploy" means here)
There is no server; deploy = submission-ready public repo that reproduces on a clean clone.
1. Working tree clean; `git branch` shows only `main`; `.gitattributes` per section 2; `git ls-files` equals the layout in section 1 (no `.venv`, no caches).
2. Clean-clone test inside the project folder only: `git clone . ./_cleanclone_tmp` (deleted afterwards and ignored), create a fresh venv, `pip install -r requirements.txt`, run `generate_data.py` (check printed hash) then `part1.py`..`part4.py` in README order, then `diff -r outputs _cleanclone_tmp/outputs` and `cmp data/records.csv` must show no differences.
3. Run the full test/check suite; resolve every failure.
4. README commands copy-pasted verbatim in the clean clone and working as written.
5. Final commit on `main`. The user (not me) creates the public GitHub repo, adds the remote, pushes `main`, sets default branch to `main`, and confirms in the browser that all files and the `main` branch show.
6. Submit root URL (default branch `main`) or `/tree/main` URL before the 16 Nov 2026 closing time; keep the confirmation.

## 11. Decisions needing the user's input
1. CRLF vs LF: `csv.writer` emits CRLF; current `.gitattributes` forces LF on `*.csv`. Implementation will check the hash and, if needed, mark `data/records.csv` (and `generate_data.py`) `-text`. Confirm that is acceptable.
2. NumPy version to pin (default: the version installed in `.venv`, expected 1.26.4).
3. `elbow_k` fallback if no K in {3,4,5} satisfies the rule (brief silent): proposal is to assert it exists on the reference data and ask before inventing a value.
4. Memo recommendation default (logistic at Task 2.8 tau) pending Part 4 numbers.
