# DESIGN — `shark_catalog.species` (Phase 1B, P0)

## The problem

SharkScarAnnotator has no species field anywhere. It was written for one study
animal, so every catalog row it has ever produced means *Carcharodon carcharias*
by convention alone — a fact that exists only in the heads of the people who
built it. Phase 1B needs a sevengill (*Notorynchus cepedianus*) catalog living in
the same platform beside the white-shark one. Before anything else can be built,
the platform has to be able to say which animal a row is about.

The repo had already asked itself this question and answered it. From
`plans/11-system-manifest.md:520`:

> is this a single-species catalogue (white sharks?) or genuinely mixed, in which
> case per-individual taxon on `shark_catalog` is the right answer

This change implements that answer.

## What was added

**`shark_catalog.species TEXT` (nullable)**, added in
`annotation/database.py:init_catalog_dwc_columns()`.

That placement is not incidental. This codebase has two schema mechanisms and
only one of them runs in production: the `scripts/migrate_schema_v*.py` chain
executes at Docker *build* time against a throwaway layer and never touches the
mounted volume, while the running app calls the `init_*_tables()` functions at
import. `database.py` says so at length (the `init_track_tables` docstring;
`init_catalog_dwc_columns`'s own comment records that an earlier early-return in
that function meant production silently had *no catalog table at all*). A
migration script here would have been a file that never runs on the machine that
matters. So the column is a `CREATE TABLE IF NOT EXISTS` + `_column_exists()` +
`ALTER` inside the init function, which is already called from `app.py:258` and
`scripts/init_db.py:249`. Deploying is the migration.

## The backfill, and why it runs exactly once

Existing rows are backfilled to `DEFAULT_SPECIES = "Carcharodon carcharias"`.

Backfilling to NULL was the obvious alternative and is wrong: it would erase a
meaning the data has carried since the first row. These rows *are* white sharks.
The platform simply had no column in which to say so, and "we do not know" is a
strictly worse description of them than "white shark".

The subtler decision is that the backfill is **fused to the ALTER** — it lives
inside the `if not _column_exists(...)` branch — rather than running on every
init the way the `organism_id` heal three lines below it does. Those two are not
symmetric, and treating them the same would be a live corruption bug:

* `organism_id` has no legitimate NULL. A row missing one is always a defect, so
  healing it on every boot is always right.
* `species` has a legitimate NULL: *nobody has said*. Two paths produce it.
  `db_reid.decide_reid_individual` inserts a bare `shark_catalog` row on a re-ID
  acceptance (`INSERT OR IGNORE INTO shark_catalog (display_name, created_at)`)
  with no species, because at that moment nobody has determined one. And an
  operator can clear the field deliberately.

A repeating backfill would stamp `Carcharodon carcharias` onto exactly those rows
on the next app restart — silently converting every sevengill the re-ID pipeline
catalogued into a white shark. That is the precise failure this column exists to
prevent, so re-running init is a strict no-op and
`test_reinit_does_not_re_stamp_a_deliberately_cleared_species` pins it.

## Three states, not two

`create_shark` distinguishes:

| caller passes | stored | meaning |
|---|---|---|
| nothing (`species=None`) | `DEFAULT_SPECIES` | "I have no opinion" — the app's historical position, so behaviour is unchanged for every existing client |
| `""` | `NULL` | "I explicitly decline to assert a species" |
| `"Notorynchus cepedianus"` | verbatim | this is a sevengill |

The default lives in the Python app layer (`create_shark`), **not** as a SQLite
`DEFAULT` clause. A schema default would also apply to `db_reid`'s raw INSERT,
re-creating the corruption above through a different door.

`update_shark` follows the file's existing idiom exactly — `None` means "not part
of this patch", `""` means "clear it" — and deliberately does *not* re-default on
clear. An update is a human saying what they know; re-asserting white shark over
that would be the app overruling them.

`list_sharks` / `get_shark` use `SELECT *`, so the field surfaces on reads for
free. `POST` and `PATCH /api/admin/catalog` thread the field through, unchanged
in every other respect.

## Darwin Core

The catalog already had `scientific_name` / `scientific_name_id`, and
`dwc_adapter.py` already mapped them to `dwc:scientificName` /
`dwc:scientificNameID`. So there are now two columns that can name a taxon, and
the mapping has to pick.

`annotation/dwc_adapter.py::_taxon()` resolves it: **`scientific_name` wins when
set; `species` is the fallback.** The reason is that `scientific_name` and
`scientific_name_id` are one curated pair — the name travels with a WoRMS AphiaID
URI. Taking the name from `species` while publishing the identifier from
`scientific_name_id` would emit a record whose name and ID disagree, which no
consumer can untangle. Keeping the pair atomic is worth more than letting the
newer column win. In practice this rarely bites: `scientific_name` is unset
across the current catalog, so a sevengill entered with only `species` exports
correctly, which is the case Phase 1B actually needs.

**The identifier only ever accompanies the name it identifies.** Choosing the
name is not enough, because nothing constrains the two curated columns to be
written together — the admin `PATCH` route accepts either alone, and the schema
has no check — so a row can carry a `scientific_name_id` with `scientific_name`
NULL. Publishing that AphiaID unconditionally, as the first draft of `_taxon`
did, reintroduces exactly the disagreement the fallback rule exists to prevent:
`species` supplies the name, the orphaned ID rides along beside it, and the
record names two different taxa at once — worse than no identifier, because a
consumer cannot tell which one it is about. So `_taxon` returns the curated pair
together or the `species` name with `scientific_name_id: None`; the same rule
leaves an ID-only row fully untaxonomised, since there is no name for it to
identify. A curator who wants the ID published fills in the name it belongs to.
`test_an_aphia_id_is_never_published_beside_a_species_fallback` pins both rows.

**The guard the adapter's comment protects is untouched.** That comment —

> a wrong species applied to a whole dataset is far more damaging than a missing
> one

— is about `default_scientific_name`, the *dataset-level* setting that fills a
taxon for every uncatalogued occurrence in one stroke. It is still `None` by
default, still the only bulk mechanism, and `_taxon` never consults it.
`_taxon` reads only what a human recorded against one individual, and an
individual with neither column set still exports untaxonomised rather than
guessed. `test_a_per_individual_species_beats_the_dataset_default`,
`test_the_dataset_default_still_fills_an_untaxonomised_individual` and
`test_an_individual_with_no_species_stays_untaxonomised` pin all three halves:
the default fills blanks and only blanks, and `species` is no longer blank. (The
defaults are applied by the writers through `DatasetMeta.apply_defaults`, not by
`occurrences()`, so those two tests go through `dataset_meta()` — reading the raw
occurrence stream would have pinned nothing.)

The per-individual backfill is a different act from that dataset-level default,
and the distinction is the whole reason it is defensible: it writes a value these
specific rows have always asserted, once, at the moment the column appears —
rather than broadcasting a guess across records nobody has identified.

One defensive detail: `_organism_map` selects `species` only when `PRAGMA
table_info` shows it. This adapter is read-only and gets pointed at backups and
older volumes; an additive column that has not reached a given database must not
fail the entire export with `no such column`. It also means the existing
`tests/test_dwc_adapter.py` fixture — whose schema predates this column — keeps
passing without being edited.

## Deliberately left out

* **Per-annotation / per-track species.** The annotation grain would need
  species on `annotations`, `tracks` and `scar_objects`, plus a consensus rule
  for raters who disagree about the animal. Species belongs to the *individual*
  (`shark_catalog` is the `dwc:Organism`), and identity flows to occurrences
  through the existing `encounter_priority.shark_catalog_id` link, so the whole
  DwC export already picks it up from one column. Follow-up only if a real
  disagreement case appears.
* **A UI dropdown on the admin catalog panel.** The API accepts and returns the
  field; the admin panel does not yet show it. Wiring it needs a controlled
  vocabulary decision (free text? WoRMS lookup? a fixed two-species list?) that
  is a product question, not a schema one, and shipping the storage first
  unblocks everything downstream.
* **Config-driven default.** A sevengill-first deployment currently still gets
  `Carcharodon carcharias` on a create that names no species — correct for the
  operating La Jolla programme, wrong for a fresh sevengill install. The fix is a
  `catalog.default_species` key in `config.yaml`, and `DEFAULT_SPECIES` is a
  single named constant precisely so that becomes a one-line change. Held back
  because the task's requirement was that behaviour be unchanged until someone
  opts in, and a config key is a second way to be wrong before anyone has asked
  for it.
* **Threading species into `db_reid.decide_reid_individual`.** Left writing NULL
  on purpose. That path creates a placeholder individual from an accepted match
  and genuinely does not know the species; NULL is the honest record. (It also
  already omits `organism_id`, which `init_catalog_dwc_columns` heals on the next
  boot — a pre-existing gap noted in `plans/11-system-manifest.md:129`, out of
  scope here.)
* **Backfilling `scientific_name` from `species`.** Would look tidy and would
  publish names beside AphiaIDs nobody verified. `_taxon` does the join at read
  time instead, where it is reversible.
* **`dwc:kingdom` / higher taxonomy per row.** `dataset_meta` already sets
  `default_kingdom="Animalia"`; nothing else is knowable from a species string
  without a WoRMS lookup, which is a network dependency this adapter does not
  have.

## Tests

`tests/test_shark_catalog_species.py`, 20 behavioural tests in the existing
suite's style (real SQLite via `monkeypatch.setattr(db, "DB_PATH", ...)`, no
mocks). Grouped by the failure they prevent:

* **migration** — column lands on a legacy table and on a bare DB; backfill hits
  pre-existing rows; init twice changes nothing; a deliberately cleared species
  survives a later boot.
* **CRUD round-trip** — omitted/explicit/blank on create; set and clear on
  update; a patch that omits species leaves it alone; the field surfaces in
  `list_sharks`; a mixed white-shark/sevengill/unknown catalog round-trips with
  neither species rewriting the other.
* **DwC** — species reaches `dwc:scientificName`; the curated pair wins over it;
  an AphiaID is never published beside a `species` fallback (nor alone); neither
  column set stays untaxonomised; the dataset default does not overwrite a
  per-individual value but does still fill an untaxonomised one; an export
  against a DB without the column still runs.
* **admin routes** — species round-trips through `POST`/`PATCH`/`GET
  /api/admin/catalog` with the Flask test client, and a `POST` that omits the key
  still records `DEFAULT_SPECIES`, which is the back-compat contract for the
  existing admin UI. These endpoints hang off `app.py`'s module-level app rather
  than a blueprint taking injected decorators (the pattern
  `tests/test_dataset_routes.py` uses), so the auth stub goes in at
  `app._get_auth`, the single lookup `require_login` performs — the real
  `require_admin` still runs, as does `app.csrf_check`, which is why the client
  sends `X-Requested-With`. The `app` import is fixture-local and
  `importorskip`-guarded, so the file still collects without `cv2`/`google-auth`.

Also verified outside the suite, against a copy of the real dev `catalog.db`
rewound to its pre-P0 shape: a legacy row backfills to `Carcharodon carcharias`,
clearing it and re-running init leaves it NULL, and a new row takes
`Notorynchus cepedianus` verbatim.

## Result

* Baseline, unpatched `main` @ `2e82cd9`: **1 failed, 974 passed, 16 skipped**
* With all three patches: **1 failed, 994 passed, 16 skipped**

Same single failure both times —
`test_signals_review_contract.py::test_every_review_route_exists_on_the_server`,
which fails because a checkout has no `config.yaml`, so `signals.enabled` is off,
so the `/api/signals/*` blueprint is never registered. Adding
`signals: {enabled: true}` makes that file pass 26/26. Unrelated to this change;
detail in `APPLY.md`.
