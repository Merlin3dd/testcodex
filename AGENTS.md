# AGENTS.md

<!--
  This file is read by Codex agents. Write instructions as if to a new team
  member, limiting context only to what is required to build, test and ship
  code correctly.
-->

## Project overview

* **Purpose**: Compute line‑of‑sight (viewshed) areas for radio transmitters and merge individual footprints into interactive coverage maps.
* **Key components**

| Path                | Role                                                                                                                       |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `radio_server.py`   | FastAPI micro‑service exposing **POST /radio\_viewshed** that accepts transmitter parameters and returns GeoJSON coverage. |
| `coverage_async.py` | Async CLI helper that samples elevation tiles, runs viewshed algorithm and writes raster or vector output.                 |
| `worker.py`         | Background job runner invoked by the API (simple `asyncio` queue; no external broker).                                     |
| `main.py`           | Command‑line aggregator that combines multiple coverage rasters into a single composite.                                   |
| `index.html`        | Stand‑alone Leaflet map to preview layers produced by the server.                                                          |

### Invariants

1. Do **not** hard‑code absolute paths; all IO paths must be provided via CLI or environment variables.
2. Do **not** commit files generated under `.coverage_cache/` or `.dem_cache/`.

## Directory layout

```text
.
├── data/               # DEM cache written at runtime
├── results/            # GeoTIFF and GeoJSON outputs
├── app/                # (optional) future FastAPI package home
└── tests/              # pytest unit and e2e tests
```

## Code style

* **Black** (line length 88).
* **Ruff**: see `pyproject.toml`.
* Docstrings follow Google style.
* Variables: `snake_case`, classes: `PascalCase`, constants: `UPPER_SNAKE`.

## Dependencies

Managed via **Poetry**. Typical flow:

```bash
poetry add <package>
poetry export -f requirements.txt --without-hashes > requirements.txt
```

All runtime deps must be pinned in `requirements.txt` for Docker builds.

## Tests & CI

* Run locally with `pytest -q`.
* GitHub Actions workflow `.github/workflows/ci.yml` runs on every pull request.
* Coverage threshold: 90% . Failing PRs are blocked.

### Local helpers

```bash
make env            # set up virtualenv
make dev            # start FastAPI with auto‑reload
make seed-dem       # download SRTM tiles for sample area
make test           # run full test suite
```

## Git workflow

1. Create branches `feat/<issue>-<slug>` or `fix/<issue>-<slug>`.
2. Follow commit convention:

```text
type(scope): short summary

Details:
 - What was done
 - Why
Fixes: #<issue>
```

3. PR description must use `.github/PULL_REQUEST_TEMPLATE.md`.

4. Tasks labelled **high‑risk** require a draft PR and human review before merge.

## Limitations

* The project works **offline**; do not call external APIs in tests.
* Files in `docs/diagrams/` are generated manually; do not edit them.

---

> Update this file whenever rules or environment change; Codex will apply
> the new instructions at the next run.
