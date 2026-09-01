# APPLY — P0 species column for SharkScarAnnotator

Three patches against **`SharkScarAnnotator` @ `2e82cd9`** ("Update team size in
project description", the tip of `main` in the local clone at
`/home/user/sharkscarannotator`).

```
0001-shark_catalog-record-species-defaulting-to-the-value.patch   schema + CRUD + admin routes
0002-dwc-fall-back-to-shark_catalog.species-for-dwc-scien.patch   Darwin Core mapping
0003-tests-schema-docs-for-shark_catalog.species.patch            tests + CLAUDE.md/AGENTS.md
```

They are ordered and each leaves the tree green; apply all three.

## Apply

```sh
cd /path/to/SharkScarAnnotator
git checkout -b p0-species main          # or whatever base you want
git am /home/user/SanBox/phase1b/p0-species-migration/*.patch
git log --oneline -3
```

`git am` was verified against a fresh clone of `main` — all three apply with no
fuzz. If you are applying onto a moved `main`, the touched regions are:
`annotation/database.py` (`init_catalog_dwc_columns`, `create_shark`,
`update_shark`), `annotation/dwc_adapter.py` (`_organism_map` and the three
`Occurrence` yield sites), `app.py` (`api_admin_catalog_create` /
`api_admin_catalog_patch`), plus one new test file and the two agent-guide schema
blocks.

## No migration script to run

There is deliberately no `scripts/migrate_schema_v*.py` in this change. Per the
repo's own convention (`annotation/database.py` comments around
`init_track_tables`, and `app.py`'s startup block), the running app calls the
`init_*_tables()` functions against the mounted volume and **never** executes the
versioned migration chain — that chain runs at Docker *build* time against a
throwaway DB. The column and its backfill therefore live in
`init_catalog_dwc_columns()`, which is already called from both places that
matter:

* `app.py:258` — every app start, against the persistent volume;
* `scripts/init_db.py:249` — dev/CI database creation.

So: **deploying is the migration.** Nothing extra to run.

## Run the tests

The suite needs Flask; `app.py` also imports `cv2` and `google-auth`, so
`tests/test_app_wiring.py` and `tests/test_route_binding.py` will not even
collect without them. From a bare checkout:

```sh
pip install -r requirements.txt -r requirements-dev.txt
mkdir -p database && python scripts/init_db.py    # database/ is .gitignored; app.py
                                                  # runs init_* at import and needs it
pytest tests/
```

### The tests that prove this change

```sh
pytest tests/test_shark_catalog_species.py tests/test_shark_catalog_dwc.py tests/test_dwc_adapter.py
```

Expected:

```
..........................................................               [100%]
58 passed in 1.45s
```

(20 new in `test_shark_catalog_species.py`, plus the 11 pre-existing catalog-DwC
identity tests in `test_shark_catalog_dwc.py` and the 27 pre-existing adapter
tests in `test_dwc_adapter.py`, all unchanged and still passing. No existing test
file was edited. Counts measured per file with
`pytest tests/<file>.py`.)

The last of the 20 is a route test that drives `POST`/`PATCH`/`GET
/api/admin/catalog` through the Flask test client. It imports `app` inside its
fixture, guarded by `importorskip`, so on an interpreter without
Flask/`cv2`/`google-auth` it skips (`19 passed, 1 skipped`) instead of taking the
whole file down with a collection error.

### Whole suite

```sh
pytest tests/
```

Expected:

```
1 failed, 994 passed, 16 skipped in ~22s
FAILED tests/test_signals_review_contract.py::test_every_review_route_exists_on_the_server
```

## The one failure is pre-existing — baseline first

Measured on the **unpatched** tree (`main` @ `2e82cd9`), same interpreter, same
environment:

```
1 failed, 974 passed, 16 skipped in 24.23s
FAILED tests/test_signals_review_contract.py::test_every_review_route_exists_on_the_server
```

Same single failure, before and after. 974 → 994 passed is exactly the 20 tests
this change adds; nothing regressed.

**Why it fails, and why it is unrelated.** `app.py:1239 _register_signals()`
registers the `/api/signals/*` blueprint only when `signals.enabled` is true in
`config.yaml`, which defaults OFF. A checkout ships `config.yaml.example` but no
`config.yaml`, so the blueprint is never registered and the contract test —
which asserts every route `static/js/signals_review.js` calls exists on the
server — reports six missing routes. Confirmed by construction:

```sh
printf 'signals:\n  enabled: true\n' > config.yaml
pytest tests/test_signals_review_contract.py     # 26 passed
rm config.yaml
```

It touches no catalog, DwC or species code path.

## Environment notes (this sandbox, not the patch)

Getting the suite to run here required, in order: `pip install --ignore-installed
blinker flask pyjwt cffi cryptography opencv-python-headless google-auth
google-auth-oauthlib google-api-python-client flask-cors` (the Debian `blinker`
and `cryptography` packages shadow working wheels — `cryptography` was
importable but panicked on a missing `_cffi_backend`), then `mkdir -p database &&
python scripts/init_db.py`. None of that is part of the patch; it is what a bare
container needs before `pytest tests/` collects at all.
