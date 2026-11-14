"""Narrative text for the portfolio pages. Numbers are filled from outputs/ at request time."""
from web.outputs_reader import read_results

SYNTHETIC = ("All data is synthetic, produced by the project's generator script. No protected attribute "
             "is used or derived, and nothing here may be used to rank people on such attributes.")

QUESTIONS = [
    "Is the planning assumption about days to an offer decision still right, and how far can a simple model of it be trusted?",
    "Which candidates will decline an offer, and what alert threshold minimises cost in INR?",
    "What groups does the candidate pipeline contain, and how large would later text and image models need to be?",
]

SEGMENT_LABELS = {
    "A": "Experienced, small pay gap, quick decision",
    "B": "Mid-experience, reschedule-heavy, slow decision",
    "C": "Early-career, wide pay gap",
}


def overview():
    p1 = read_results("part1_results.txt")
    p2 = read_results("part2_results.txt")
    p3 = read_results("part3_results.txt")
    p4 = read_results("part4_results.txt")
    kpis = [
        {"label": "Mean days to decision", "value": f"{p1['mean_value']:.2f}", "note": "planning mean is 8.0"},
        {"label": "Records needed", "value": str(p1["n_required"]), "note": "for a 0.5 day margin, 800 available"},
        {"label": "Regression R2", "value": f"{p1['r2']:.3f}", "note": "raw baseline; log-log fits better"},
        {"label": "Test recall at 0.3", "value": f"{p2['tau_0.3_recall']:.3f}", "note": "logistic, 51 declines in test"},
        {"label": "Best threshold", "value": f"{p2['cost_tau']:.2f}", "note": "chosen on training rows by cost"},
        {"label": "Test cost at that threshold", "value": f"INR {p2['cost_test_at_tau']}",
         "note": f"vs INR {p2['cost_test_flag_nobody']} flagging nobody"},
        {"label": "Segments", "value": str(p3["elbow_k"]),
         "note": f"sizes {p3['size_segment_A']}, {p3['size_segment_B']}, {p3['size_segment_C']}"},
        {"label": "CNN parameters", "value": f"{p4['cnn_total']:,}", "note": f"MLP {p4['mlp_total']:,}"},
    ]
    findings = [
        {"part": 1, "text": f"The planning mean of 8.0 days is rejected (z = {p1['z_stat']:.3f}), but the gap is under one day "
                            f"({p1['mean_value'] - 8.0:.2f}). A 0.5 day margin needs {p1['n_required']} records."},
        {"part": 1, "text": f"The raw regression is funnel-shaped (residual spread ratio {p1['resid_sd_ratio_raw']:.2f}); "
                            f"the log-log model fixes it ({p1['resid_sd_ratio_log']:.2f})."},
        {"part": 2, "text": f"Accuracy misleads: blind accuracy is {p2['blind_accuracy']:.4f} against {p2['tau_0.5_accuracy']:.4f} "
                            "for the model at 0.5. The salary gap ratio is the strongest driver."},
        {"part": 2, "text": f"Alerting at {p2['cost_tau']:.2f} costs INR {p2['cost_test_at_tau']} on test, against "
                            f"INR {p2['cost_test_at_0.5']} at 0.5 and INR {p2['cost_test_flag_nobody']} for no alert."},
        {"part": 3, "text": f"Three segments emerge; the early-career wide-pay-gap segment C declines most "
                            "(outcome rate 0.4664 in the segments table)."},
        {"part": 4, "text": "Voting ensembles did not beat the logistic model alone; the logistic model is the recommendation."},
    ]
    return {"title": "Apna Recruitment Decision Intelligence", "questions": QUESTIONS, "kpis": kpis,
            "findings": findings, "synthetic_notice": SYNTHETIC}


def memo():
    p2 = read_results("part2_results.txt")
    paras = [
        "To: Head of Talent-Acquisition Operations. Subject: which model to run in production.",
        f"Recommendation: deploy the logistic-regression model with an alert threshold of {p2['cost_tau']:.2f}, chosen on "
        "training rows by minimising cost at INR 50 per unneeded follow-up and INR 400 per missed decline. "
        f"On 240 held-out test records this costs INR {p2['cost_test_at_tau']}. Using 0.5 would cost INR {p2['cost_test_at_0.5']}, "
        f"and flagging nobody INR {p2['cost_test_flag_nobody']}. At threshold 0.3 the logistic model catches "
        f"{p2['tau_0.3_tp']} of {p2['tau_0.3_tp'] + p2['tau_0.3_fn']} declines (recall {p2['tau_0.3_recall']:.4f}, "
        f"F1 {p2['tau_0.3_f1']:.4f}), while the best K-nearest-neighbour model catches only {p2['knn_tp']} (F1 {p2['knn_f1']:.4f}). "
        "Combining models by voting did not beat the logistic model alone.",
        "Risk: the evidence rests on one split with only 51 positive test records and synthetic data, so the cost "
        "figures are rough and the threshold should be re-checked on live data before wide rollout.",
        "Next build: the resume text model (16 job families, 26768 parameters) is worth building before the image "
        "model, once the structured model is in production.",
    ]
    return {"title": "Model-choice memo", "paragraphs": paras}


def limitations():
    paras = [
        "The data is synthetic and generated from stated formulas; real candidate behaviour, drift and data quality problems are absent.",
        "A single pinned 70/30 split with only 51 positive test records makes the test metrics and the chosen threshold noisy.",
        "K-means uses pinned starting rows, so its solution is reproducible but not necessarily the global optimum; the hierarchical clustering uses a supplied table.",
        "The confidence interval and test rely on a normal approximation although days to decision is strongly skewed.",
        "No calibration check or fairness review was done on the logistic model; it must not be used to rank people on protected attributes.",
        "Web layer: the ROC curve, AUC and segment scatter are derived views computed for display; they are not graded project numbers.",
    ]
    return {"title": "Limitations", "paragraphs": paras}
