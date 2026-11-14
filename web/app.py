"""FastAPI portfolio layer. Run from the repository root: uvicorn web.app:app"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Query, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from web import content  # noqa: E402
from web.model_service import get_model  # noqa: E402
from web.outputs_reader import OutputsMissing, read_results, read_table  # noqa: E402
from web.schemas import ScoreRequest, ScoreResponse  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
ODD_K = list(range(1, 22, 2))


@asynccontextmanager
async def lifespan(app):
    get_model()  # warm the cache at startup
    yield


app = FastAPI(title="Apna Recruitment portfolio", version="1.0.0", lifespan=lifespan)


def err(status, code, message, details=None):
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(RequestValidationError)
async def on_validation(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
    return err(422, "validation_error", "Invalid input", details)


@app.exception_handler(StarletteHTTPException)
async def on_http(request: Request, exc: StarletteHTTPException):
    code = "not_found" if exc.status_code == 404 else "http_error"
    return err(exc.status_code, code, str(exc.detail))


@app.exception_handler(OutputsMissing)
async def on_missing(request: Request, exc: OutputsMissing):
    return err(503, "outputs_missing", f"Required output file not found: {exc}. Run the part scripts first.")


def part_payload(name, results_file, tables, notes):
    return {"results": read_results(results_file),
            "tables": {t: read_table(f) for t, f in tables.items()}, "notes": notes}


@app.get("/api/health")
def health():
    m = get_model()
    return {"status": "ok", "rows": len(m.a["data"]["y"]), "train_rows": len(m.ytr), "test_rows": len(m.yte)}


@app.get("/api/overview")
def overview():
    return content.overview()


@app.get("/api/part1")
def part1():
    return part_payload("part1", "part1_results.txt", {}, [
        "Planning mean is 8.0 days; the value column is days to hire.",
        "gd_J0..gd_J500 are gradient-descent cost values at update 0, 1, 10, 100 and 500."])


@app.get("/api/part2")
def part2():
    return part_payload("part2", "part2_results.txt",
                        {"thresholds": "part2_thresholds.csv", "knn_cv": "part2_knn_cv.csv"},
                        ["Cost rule: INR 50 per false positive, INR 400 per false negative."])


@app.get("/api/part3")
def part3():
    p = part_payload("part3", "part3_results.txt",
                     {"segments": "part3_segments.csv", "hierarchical": "part3_hierarchical.csv"},
                     ["Hierarchical clustering uses a supplied 6-cohort distance table, not candidate data."])
    for r in p["tables"]["segments"]:
        r["label"] = content.SEGMENT_LABELS.get(r["segment"], "")
    return p


@app.get("/api/part4")
def part4():
    return part_payload("part4", "part4_results.txt",
                        {"splits": "part4_splits.csv", "voting": "part4_voting.csv",
                         "adaboost": "part4_adaboost.csv", "architecture": "part4_architecture.csv"},
                        ["Network sizes are parameter counts only; no network is trained."])


@app.get("/api/memo")
def memo():
    return content.memo()


@app.get("/api/limitations")
def limitations():
    return content.limitations()


@app.get("/api/threshold")
def threshold(tau: float = Query(0.5, ge=0, le=1)):
    return get_model().threshold(tau)


@app.get("/api/curve")
def curve():
    return get_model().curve()


@app.get("/api/knn")
def knn(k: int = Query(5, ge=1, le=21)):
    if k not in ODD_K:
        return err(422, "validation_error", "k must be an odd number from 1 to 21",
                   [{"field": "k", "message": "must be odd"}])
    return get_model().knn(k)


@app.get("/api/segments")
def segments():
    r = read_results("part3_results.txt")
    rows = read_table("part3_segments.csv")
    for row in rows:
        row["label"] = content.SEGMENT_LABELS.get(row["segment"], "")
    return {"segments": rows,
            "elbow": [{"K": k, "wcss": r[f"wcss_k{k}"]} for k in range(2, 7)],
            "elbow_k": r["elbow_k"],
            "pca": {"explained": [r[f"explained_ratio_{i}"] for i in range(1, 6)],
                    "loadings": {"pc1": [r[f"pc1_loading_X{j}"] for j in range(1, 6)],
                                 "pc2": [r[f"pc2_loading_X{j}"] for j in range(1, 6)]}}}


@app.get("/api/segments/points")
def segment_points():
    return {"points": get_model().segment_points()}


@app.get("/api/ensemble")
def ensemble(tau: float = Query(0.3, ge=0, le=1)):
    return get_model().ensemble(tau)


@app.post("/api/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    return get_model().score(req)


@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
def api_not_found(rest: str):
    return err(404, "not_found", f"No such endpoint: /api/{rest}")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
