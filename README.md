Domain: Recruitment

This repository answers three talent-analytics questions for a job platform: whether the planning mean for days to an offer decision still holds, which candidates will decline an offer, and what groups the pipeline contains. All data is synthetic, produced by `generate_data.py`; it is not real company data. No protected attribute (age, gender, religion, caste, disability or similar) is in the data or derived from it, and nothing here may be used to rank people on such attributes. Every number below is copied from the committed files in `outputs/`.

## How to run

From the repository root, in this order:

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py         # must print: 800 10 170 0537bc7fb4a044719831c43166e79d3c
python part1.py
python part2.py
python part3.py
python part4.py
```

`common.py` holds shared helpers and runs nothing on import. `part3.py` also prints the K-means pass counts to the console (not to a file). Each script rewrites its files in `outputs/` byte for byte.

## Part 1 findings

**R1.1** H0 (mean 8.0 days) is rejected: `z_stat` = 2.8930, beyond the critical value 1.96 (`reject_h0` = yes). A Type I error here would be concluding that the true average time to an offer decision differs from 8.0 days when it actually equals 8.0, and re-planning hiring for no reason. The gap `mean_value` − 8.0 is 8.9382 − 8.0 = 0.9382 days (95% interval 8.3026 to 9.5738). It is statistically clear, but under one day on a plan of 8; it matters only if capacity is planned tightly, and the standard deviation of 9.1723 days shows individual cases vary far more than this gap.

**R1.2** `n_required` = 1293 records for a margin of 0.5 day at 95% confidence, against 800 rows available. The current sample gives a margin of about 0.64 day (1.96 × 0.3243), so the planning team would need roughly 493 more records to reach half a day.

**R1.3** `gd_slope_gap` = 0.0000 (below 0.001). The cost is a single convex bowl in the slope and intercept, so with a small enough step size and enough updates (cost flat at 40.3698 from update 100 to 500) gradient descent ends at the same minimum the closed form solves for exactly.

**R1.4** `resid_sd_ratio_raw` = 3.7525 and `resid_sd_ratio_log` = 1.0396. The raw model shows a strong funnel: the residual spread in the upper half of fitted values (8.6891) is far larger than in the lower half (2.3155). The log-log model removes it (0.5369 against 0.5165).

**R1.5** Predictors with |coefficient| above 0.05 (constant excluded): ln(X4) expected salary (1.1576), X2 interview reschedules (0.0940), X3 salary gap ratio (0.1067) and X6 competing offer flag (0.1518). X1, X5 and X7 fall below the mechanical 0.05 line (−0.0074, 0.0028, 0.0172); this is not a ranking of importance. `mr_adj_r2` = 0.7403 against `loglog_adj_r2` = 0.6561, so the extra predictors earn their place: adjusted R² rises by about 0.08 even after the penalty for more terms.

## Part 2 findings

**R2.1** From lowest to highest after 200 updates: eta 1.0 (0.3703), eta 0.1 (0.3776), eta 0.01 (0.5091). A rate that is too small (0.01) has barely progressed after 200 updates, while the large rate (1.0) was still stable here and converged fastest, though a large rate can overshoot on other problems.

**R2.2** The three largest absolute weights are X3 salary gap ratio (0.9013), X1 experience months (−0.5554) and X2 interview reschedules (0.5505). The largest odds ratio is `odds_ratio_X3` = 2.4629: each rise of one training SD in the salary gap ratio multiplies the odds that a candidate declines the offer by about 2.46, other predictors fixed.

**R2.3** `blind_accuracy` = 0.7875 (predict nobody declines) next to accuracy 0.8042 at τ = 0.5: the model beats doing nothing by only 0.0167, so accuracy rewards ignoring the minority of declines. Lowering τ from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 and lowered precision from 0.5455 to 0.4634.

**R2.4** At τ = 0.3 the logistic model has `f1` = 0.5714 and recall 0.7451; the KNN model (K = 5) has `knn_f1` = 0.3696 and `knn_recall` = 0.3333. Logistic is clearly better on both.

**R2.5** Deploy the logistic model at the money-chosen τ = 0.15 (`cost_tau` = 0.1500). On the test rows it costs INR 6050 (`cost_test_at_tau`), against INR 11800 at τ = 0.5 and INR 20400 for flagging nobody. The threshold was chosen on training rows so the test rows remain an untouched, honest estimate of cost on new candidates.

## Part 3 findings

**R3.1** The WCSS drops are WCSS(2) − WCSS(3) = 659.8282, WCSS(3) − WCSS(4) = 108.0739, WCSS(4) − WCSS(5) = 171.0575 and WCSS(5) − WCSS(6) = 45.5466. The drops do not shrink steadily (the K = 4 to 5 drop is bigger than the K = 3 to 4 drop), because the pinned starting rows give one reproducible solution, not necessarily the global optimum. The elbow rule gives `elbow_k` = 3: the drop 108.0739 is below half of 659.8282 (329.9141), so K = 3 is supported.

**R3.2** Segments from `part3_segments.csv`:
- A, "Experienced, small pay gap, quick decision" (322 records; mean experience 47.5 months, pay gap 0.2250, outcome rate 0.0186, 3.5976 days to decision): keep to the standard fast track with minimal follow-up.
- B, "Mid-experience, reschedule-heavy, slow decision" (255 records; 6.0706 reschedules, pay gap 0.4534, outcome rate 0.2353, 17.1248 days): fix interview slots early and confirm scheduling, since delays are the pattern here.
- C, "Early-career, wide pay gap" (223 records; mean experience 14.4 months, pay gap 0.7118, outcome rate 0.4664, 7.2883 days): this is the segment with the highest `outcome_rate`, 0.4664; open a pay conversation and follow up before the offer lapses.

**R3.3** Single linkage cuts into 4 clusters: {E1}, {E4}, {E5} and {E2, E3, E6}. Complete linkage cuts into 2 clusters: {E1, E4, E5} and {E2, E3, E6}. Single linkage merges on the one closest pair, so it chains E5 to E6 at 3.4 and its largest height gap comes early; complete linkage needs every pair close, so the bridging link only appears at 7.9 (E1 to E2), which produces a huge final gap and a cut at 2 clusters.

**R3.4** The mean vector is all zeros after standardising. PC1 is driven by experience (X1 +0.5317) and profile completeness (X5 +0.4418) against salary gap (X3 −0.5187), reschedules (X2 −0.4100) and expected salary (X4 −0.2916): a contrast between experienced, complete-profile candidates and wide-pay-gap, reschedule-prone ones. 3 components keep at least 80% of the variance (cumulative 0.7593 for two, 0.8616 for three). `mean_pc1_segment_A/B/C` = 1.7773, −0.7831 and −1.6709, so PC1 separates segment A from C, with B near C on the negative side.

## Part 4 findings

**R4.1** S4 (X1 ≤ 24) is the best split by both information gain (0.1252) and Gini gain (0.0606) because it leaves the purest children: 86 of 188 on the yes side are positive (45.7%) against 33 of 372 on the no side (8.9%). A tree grown until every leaf is pure memorises noise in the 560 training rows and will not carry over to new candidates. A maximum depth or a minimum number of records per leaf stops the tree from growing too deep.

**R4.2** I would ship the logistic model: at τ = 0.3 it has `f1` 0.5714 and recall 0.7451 (`part4_voting.csv`), against KNN 0.3696 and 0.3333, stump 0.5532 and 0.7647, hard vote 0.5344 and 0.6863, soft vote 0.5507 and 0.7451. Combining models did not help: neither vote beat the logistic model alone.

**R4.3** After round 1 normalisation, the two misclassified points (3 and 7) carry 0.2500 + 0.2500 = 0.5000 of the total weight. Round 2 trusts h2 more because its weighted error (0.1667) is lower than h1's round-1 error (0.2500), giving α2 = 0.8047 against α1 = 0.5493.

**R4.4** Model A option 2 (CNN):

| Layer | Output shape (H × W × D) | Parameters |
|---|---|---|
| conv 5×5, 6 filters, pad 2 (+ ReLU) | 28 × 28 × 6 | 156 |
| max-pool 2×2, stride 2 | 14 × 14 × 6 | 0 |
| conv 5×5, 16 filters, pad 0 (+ ReLU) | 10 × 10 × 16 | 2416 |
| max-pool 2×2, stride 2 | 5 × 5 × 16 | 0 |
| flatten | 400 | 0 |
| dense 32 (+ ReLU) | 32 | 12832 |
| dense 10 | 10 | 330 |

Totals: `cnn_total` = 15734, `mlp_total` = 13002; `fc_layer_params` = 50332672 against `conv_layer_params` = 1792 shows why convolution scales to large inputs.

Model B (RNN): W_xh is 128 × 64, W_hh is 128 × 128 and W_hy is 16 × 128, with `rnn_total` = 26768. It does not grow with longer sequences because the same weights are reused at every time step; only the number of steps changes.

For 10 classes I would use ReLU in the hidden layers and softmax at the output with categorical cross-entropy loss. In a very deep plain network, gradients are multiplied through many layers and shrink toward zero, so early layers barely update. A skip connection adds the input of a block to its output, giving the gradient a direct path back to early layers.

## Model-choice memo

To: Head of Talent-Acquisition Operations. Subject: which model to run in production.

Recommendation: deploy the logistic-regression model with an alert threshold of 0.15, chosen on training rows by minimising cost at INR 50 per unneeded follow-up and INR 400 per missed decline. On 240 held-out test records this costs INR 6050 (`cost_test_at_tau`). Using 0.5 would cost INR 11800, and flagging nobody INR 20400, so the alert cuts expected cost by about 70 percent against doing nothing. At threshold 0.3 the logistic model catches 38 of 51 declines (recall 0.7451, F1 0.5714), while the best K-nearest-neighbour model catches only 17 (F1 0.3696). Combining models by voting did not beat the logistic model alone. The costs of the other Task 4.2 models at 0.15 were not computed.

Risk: the evidence rests on one split with only 51 positive test records and synthetic data, so the cost figures are rough and the threshold should be re-checked on live data before wide rollout.

Next build: Model B (resume text into 16 job families, 26768 parameters) is worth building before Model A, since it is the main use case, but only once the structured model is in production.

## Limitations

- The data is synthetic and generated from stated formulas; real candidate behaviour, drift and data quality problems are absent, so the numbers do not describe any real platform.
- A single pinned 70/30 split with only 51 positive test records means the test metrics and the chosen threshold are noisy; the cost of INR 6050 could change materially with a different split.
- K-means uses pinned starting rows, so its solution is reproducible but not necessarily the global optimum (the WCSS drops do not shrink steadily), and the six-cohort hierarchical clustering uses a supplied table, not the candidate data.
- The confidence interval and hypothesis test rely on a normal approximation with the sample SD, although the days-to-decision variable is strongly skewed.
- No calibration check or fairness review was done on the logistic model; probabilities are used only through a cost-chosen threshold and the model should not be used to rank people on protected attributes.
