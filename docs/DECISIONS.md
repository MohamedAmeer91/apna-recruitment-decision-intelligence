# Decisions on ambiguity

1. The brief's layout lists required files and says shared helper files are fine; it does not forbid extra files. `common.py`, `docs/` and `tests/` stay tracked.
2. `docs/REQUIREMENTS.md` held two GitHub address templates with a URL scheme. The brief says no other link may appear in the submission, so the scheme was removed (text now reads `github.com/<user>/<repo>`). Nothing else in the copy was changed.
3. Tests use `np.polyfit`, `lstsq` and `np.cov` only as independent cross-checks inside `tests/`; the forbidden-routine grep covers the graded code (`common.py`, `part1-4.py`, `generate_data.py`), not tests.
4. README number check: every backticked `key` = value in the README must equal the committed output; every 4 dp number in the README must appear in some output file, apart from a small allow-list of derived values (gap from 8.0, WCSS drops, blind-accuracy margin, half of the first WCSS drop).
5. The K-means pass counts go to stderr, and the tests read them from there.
6. Tests that regenerate outputs delete and rewrite files in `outputs/` and require a byte match with the committed copy.
