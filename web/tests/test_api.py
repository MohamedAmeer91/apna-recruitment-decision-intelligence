"""Run from the repository root:  .venv\\Scripts\\python.exe -m unittest discover -s web/tests -t ."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from web.app import app  # noqa: E402
from web.outputs_reader import read_results, read_table  # noqa: E402

GOOD = dict(experience_months=20, interview_reschedules_90d=4, salary_gap_ratio=0.6,
            expected_salary_inr_thousands=15, profile_completeness_score=40,
            competing_offer_flag=1, referred_flag=0)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TestClient(app)
        cls.c = cls._ctx.__enter__()
        cls.p2 = read_results("part2_results.txt")

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def get(self, url, status=200):
        r = self.c.get(url)
        self.assertEqual(r.status_code, status, r.text)
        return r.json()

    # ---- status and schema
    def test_health(self):
        d = self.get("/api/health")
        self.assertEqual((d["status"], d["rows"], d["train_rows"], d["test_rows"]), ("ok", 800, 560, 240))

    def test_overview_and_text(self):
        d = self.get("/api/overview")
        self.assertEqual(len(d["kpis"]), 8)
        self.assertTrue(d["findings"] and d["questions"] and d["synthetic_notice"])
        for u in ("/api/memo", "/api/limitations"):
            d = self.get(u)
            self.assertTrue(d["title"] and d["paragraphs"])
        self.assertIn("6050", " ".join(self.get("/api/memo")["paragraphs"]))

    def test_part_endpoints_match_files(self):
        for n, f in (("part1", "part1_results.txt"), ("part2", "part2_results.txt"),
                     ("part3", "part3_results.txt"), ("part4", "part4_results.txt")):
            d = self.get("/api/" + n)
            self.assertEqual(d["results"], read_results(f))
            self.assertIn("tables", d)
        self.assertEqual(self.get("/api/part2")["tables"]["thresholds"], read_table("part2_thresholds.csv"))
        self.assertEqual(self.get("/api/part4")["tables"]["voting"], read_table("part4_voting.csv"))
        self.assertEqual(len(self.get("/api/part4")["tables"]["adaboost"]), 16)

    # ---- numbers against outputs
    def test_threshold_matches_part2(self):
        for tag, tau in (("0.5", 0.5), ("0.3", 0.3)):
            d = self.get(f"/api/threshold?tau={tau}")
            for k in ("tp", "fp", "tn", "fn"):
                self.assertEqual(d[k], self.p2[f"tau_{tag}_{k}"])
            for k in ("accuracy", "precision", "recall", "f1", "specificity"):
                self.assertAlmostEqual(d[k], self.p2[f"tau_{tag}_{k}"], places=4)
        d = self.get("/api/threshold?tau=0.5")
        self.assertEqual(d["cost_inr"], self.p2["cost_test_at_0.5"])
        self.assertEqual(d["cost_flag_nobody_inr"], self.p2["cost_test_flag_nobody"])

    def test_cost_tau_cost(self):
        d = self.get(f"/api/threshold?tau={self.p2['cost_tau']}")
        self.assertEqual(d["cost_inr"], self.p2["cost_test_at_tau"])

    def test_curve_matches_threshold_csv(self):
        d = self.get("/api/curve")
        self.assertEqual(len(d["points"]), 101)
        self.assertTrue(0.5 < d["auc"] <= 1)
        for row in read_table("part2_thresholds.csv"):
            p = d["points"][round(row["tau"] * 100)]
            self.assertAlmostEqual(p["tpr"], row["tpr"], places=4)
            self.assertAlmostEqual(p["fpr"], row["fpr"], places=4)

    def test_knn(self):
        d = self.get("/api/knn?k=5")
        self.assertEqual(d["best_k"], self.p2["knn_best_k"])
        self.assertTrue(d["is_best"])
        for k in ("tp", "fp", "tn", "fn"):
            self.assertEqual(d[k], self.p2["knn_" + k])
        self.assertAlmostEqual(d["f1"], self.p2["knn_f1"], places=4)
        cv = {r["K"]: r["mean_cv_f1"] for r in read_table("part2_knn_cv.csv")}
        for r in d["cv"]:
            self.assertAlmostEqual(r["mean_cv_f1"], cv[r["K"]], places=4)
        self.assertFalse(self.get("/api/knn?k=1")["is_best"])

    def test_segments(self):
        d = self.get("/api/segments")
        self.assertEqual([s["size"] for s in d["segments"]], [322, 255, 223])
        self.assertEqual(d["elbow_k"], 3)
        pts = self.get("/api/segments/points")["points"]
        self.assertEqual(len(pts), 800)
        counts = {s: sum(1 for p in pts if p["segment"] == s) for s in "ABC"}
        self.assertEqual(counts, {"A": 322, "B": 255, "C": 223})

    def test_ensemble_matches_voting_csv(self):
        d = self.get("/api/ensemble?tau=0.3")
        csvrows = {r["model"]: r for r in read_table("part4_voting.csv")}
        self.assertEqual({m["model"] for m in d["models"]}, set(csvrows))
        for m in d["models"]:
            for k in ("tp", "fp", "tn", "fn"):
                self.assertEqual(m[k], csvrows[m["model"]][k], m["model"])
            self.assertAlmostEqual(m["f1"], csvrows[m["model"]]["f1"], places=4)

    # ---- scoring
    def test_score(self):
        r = self.c.post("/api/score", json=GOOD)
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertTrue(0 < d["probability"] < 1)
        self.assertEqual(len(d["contributions"]), 7)
        self.assertAlmostEqual(d["logit"], d["bias"] + sum(c["contribution"] for c in d["contributions"]), places=9)
        self.assertEqual(d["tau"], self.p2["cost_tau"])
        self.assertIn(d["segment"], "ABC")
        self.assertAlmostEqual(d["contributions"][2]["weight"], self.p2["weight_X3"], places=4)
        low = self.c.post("/api/score", json={**GOOD, "salary_gap_ratio": 0.1}).json()["probability"]
        self.assertLess(low, d["probability"])  # wider pay gap raises decline probability

    def test_score_validation(self):
        bad_cases = [{**GOOD, "salary_gap_ratio": 3}, {**GOOD, "competing_offer_flag": 2},
                     {**GOOD, "experience_months": "abc"}, {**GOOD, "extra": 1}, {**GOOD, "tau": 5},
                     {k: v for k, v in GOOD.items() if k != "referred_flag"}, {}]
        for body in bad_cases:
            r = self.c.post("/api/score", json=body)
            self.assertEqual(r.status_code, 422, body)
            e = r.json()["error"]
            self.assertEqual(e["code"], "validation_error")
            self.assertTrue(e["details"])

    def test_query_validation_and_404(self):
        for u in ("/api/threshold?tau=1.5", "/api/threshold?tau=x", "/api/knn?k=4", "/api/knn?k=99", "/api/ensemble?tau=-1"):
            self.get(u, 422)
        d = self.get("/api/does-not-exist", 404)
        self.assertEqual(d["error"]["code"], "not_found")

    # ---- static
    def test_static_files(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("Apna Recruitment", r.text)
        for p in ("css/style.css", "js/main.js", "js/api.js", "js/charts.js", "js/ui.js", "js/pages/overview.js",
                  "js/pages/part1.js", "js/pages/part2.js", "js/pages/part3.js", "js/pages/part4.js",
                  "js/pages/score.js", "js/pages/text.js"):
            self.assertEqual(self.c.get("/static/" + p).status_code, 200, p)
        self.assertEqual(self.c.get("/static/missing.js").status_code, 404)

    def test_no_external_links_in_frontend(self):
        root = ROOT / "web" / "static"
        for f in root.rglob("*"):
            if f.is_file():
                text = f.read_text().replace("http://www.w3.org/2000/svg", "")
                self.assertNotIn("http://", text, f.name)
                self.assertNotIn("https://", text, f.name)


if __name__ == "__main__":
    unittest.main()
