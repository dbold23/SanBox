# SHARK RAG v1 assessment (September 2026)

Independent code review of `knowledge-tools/shark-rag` in the lab archive, produced while planning LabRAG. Line numbers refer to that snapshot.

# shark-rag assessment (synthesis)

**Scope:** `/home/user/ocean-predator-ecology-lab/knowledge-tools/shark-rag`, 6,156 Python LOC across 42 files, single commit (`22963ce "Remove MBARI work; add a contributor guide"`). Five subsystem maps and a new-member journey were merged; disputed and `verified=false` items were re-checked in code. No tests/, no scripts/, no lockfile.

## Verdict

This is a personal research tool snapshotted into the lab repo, not a lab tool. About half the code is an off-domain literature crawler and dead retrieval code; the half that matters has structural bugs that make it unusable by anyone who is not its author. The valuable core — PyMuPDF page-level parsing, content-hash IDs, a folder-scan contract, embedded SQLite+LanceDB, numbered citations — is perhaps 800 LOC and worth carrying forward. Everything else should be rebuilt around one hosted instance watching one NAS/Drive folder.

## Where the 6,156 LOC go

| Area | LOC | Share | Status |
|---|---|---|---|
| acquisition package + Acquire tab | 2,990 | 49% | off-domain (3D-pose/CVPR seeds), CLI-only scrapers, undeclared bs4, never feeds ingestion |
| UI tabs + settings + main | ~1,280 | 21% | 7 tabs; Explore is a placeholder; Library select handler likely broken |
| ingestion + db | ~1,060 | 17% | the keepable core, with the bugs below |
| retrieval/answering | ~560 | 9% | ~300 LOC dead (hybrid_search, synthesizer, aquery, 4 of 5 prompts) |
| MCP server | 228 | 4% | outside the package, second config path |

## Top problems (ranked)

1. **Not shareable.** `app/config.py:6-14` hardcodes all state to `<repo>/data` (violating `CONTRIBUTING.md:22-25`); `SHARK_RAG_DATA` only moves acquisition paths and is CWD-relative (`search_config.py:13`, `upload_tab.py:24`). Each member gets an empty index in their own checkout. SQLite WAL (`db.py:15`) and embedded LanceDB cannot simply be moved onto SMB/NFS (inferred).
2. **Cannot be installed or started by a non-engineer.** Verified first-run crash: `settings_tab.py:30` → `load_settings()` → `SELECT FROM settings` runs while the UI is built in `create_app()` (`main.py:76-77`), before the startup event that calls `init_db()` (`main.py:64-68`). `pyproject.toml` pulls docling + HF/torch for a path never executed (`parser.py:47`, grep confirms no `use_docling=True`), has no `[build-system]`, no pins, and misses bs4 / python-dotenv / openai-like. README says `cd SHARK_RAG`.
3. **Citations are untrustworthy.** Authors blank (`parser.py:151-152`), title = filename, year = modal 20xx over the full text incl. bibliography (`metadata.py:38-47`), `year or 0` (`chunker.py:54`) defeats `'n.d.'`, `page_numbers` CSV string iterated as a list (`chunker.py:61` vs `chat_tab.py:118` → page 12 prints "pp. 1-2"). `CITATION_QA_PROMPT` has zero importers; no threshold, no filters (`engine.py:70-74,143-151`); no chat memory (`chat_tab.py:66`). Bibliographies are embedded because headings are "Page N" so the references-skip never fires.
4. **Folder sync fails its one job.** Path never persisted, watcher not restarted (`upload_tab.py:199-203`, `folder_watcher.py:172-188`); `stat()` outside the try aborts whole scans (`:99`); `path:size:mtime` dedup + no vector deletion → duplicates on rename/touch (`:29-33`, `vector_store.py:45` no callers); `INSERT OR REPLACE` wipes notes/status (`db.py:127-146`); failures unrecorded and discarded (`:138-141,159-165`); Scan Now blocks the UI with `sleep(2)` batches.
5. **Half the code is someone else's project.** Acquisition scrapers seeded with SMPL/SMAL/CVPR queries; `mbari_scraper.py` survives the "Remove MBARI work" commit; `citation_importer.py` has no callers; owner's email hardcoded (`search_config.py:231`, `semantic_scholar_crawler.py:272`) against `CONTRIBUTING.md:27-31`. Chat examples are about NeRF and Gaussian splatting.
6. **Embedding switch corrupts search silently.** `settings_tab.py:152-154` only resets the engine; `reset_vector_store` unused; `embed_dimensions` read by nothing; no model stamp, no rebuild.
7. **Security/hygiene.** `0.0.0.0` with no auth (grep `auth=` → none); plaintext keys in SQLite prefilled into every browser (`settings_tab.py:68-98`); no tests/ or scripts/ per `REPO_LAYOUT.md`.
8. **Docs describe a different program** (Docling, streaming, retrieval settings, chat history, `.env` for the web app).

## Reader disagreements resolved

- *Library row-click breaks (pandas vs list)*: `gr.Dataframe` at `library_tab.py:33` has no `type=`; `show_paper_detail` (`:97-105`) uses `not table_data` / `table_data[i][5]`. Code path verified; runtime failure inferred from Gradio's pandas default. Treated as high but flagged as inferred.
- *Chatbot dict-vs-tuples*: no `type=` at `chat_tab.py:20-23`; version-dependent; kept as medium/inferred.
- *Markdown vs chunker schema*: divergence verified (`markdown_loader.py:113-125` has `category`, nullable `year`, no `section_type`/`page_numbers`); LanceDB rejection inferred.
- *Docling default*: all readers agree; grep confirms.

## Keep (carry into redesign)

PyMuPDF per-page sections with page provenance; full-content-hash paper ID as the single key; `scan_and_ingest`'s collect→try→report shape; embedded SQLite + LanceDB in one host-local data dir; CitationQueryEngine's `[N]`↔source mapping and citation dict; the *text* of `CITATION_QA_PROMPT`; excluded-metadata discipline; `extract_doi` (first page only) + `openalex_client._parse_work` for real title/authors/year; Test Connection probes as a startup banner; per-file ✓/✗ lines streamed; lazy imports; Ollama-local as an option; the five MCP tool shapes as a thin `scripts/` entry; `.env.example`/`.gitignore`/`CITATION.cff`; `SHARK_RAG_DATA` extended to everything.

## Drop

Docling path and its dependencies; the acquisition package and Acquire tab (keep only the OpenAlex/Unpaywall primitives as an owner CLI); all dead retrieval/synthesis code and unused tables/constants; `markdown_loader`; drag-and-drop-into-checkout upload; daemon-thread watcher; `path:size:mtime` ledger; `INSERT OR REPLACE`; EPUB/HTML/.doc/.txt; the 5×3 provider matrix and 11-field settings form; API keys in SQLite; unused chunk/top_k settings and 39% overlap; Explore, Summary-as-tab, reading status/notes; modal-year heuristic; MCP's private engine/Gemini override/fallback; `0.0.0.0` default; two data roots; hardcoded emails; pyyaml/bibtexparser.

## Constraints from lab conventions

Env-var data roots documented in `docs/ENVIRONMENT.md`, never hardcoded, never inside the checkout; no data/keys/personal emails committed (repo will go public); standard project layout (README Stack/Data/Upstream header, package, `scripts/`, `tests/`, `docs/notes/`); installable from `pyproject.toml` alone with declared, pinned deps; ruff-clean, no dead code; `CITATION.cff` respected and owner consulted; reuse the existing OpenAlex client. Technical: SQLite/LanceDB on host-local disk, NAS/Drive hold PDFs only, relative source paths, one pinned embedding model per index, one hosted instance, LAN exposure gated by a password, citations with pages and an honest "not in the library".

## Target shape (for the redesign)

One process on a lab host: on start, `init_db()`, probe the LLM, scan `$SHARK_RAG_INBOX` (NAS mount / Drive-for-Desktop folder), then re-scan every N minutes. Three screens: **Ask** (with inline `[Author, Year, p.X]` citations, memory, and library status), **Papers** (what is indexed, what failed and why, Sync now with streamed progress), **About/Status**. Zero settings for members; the owner configures provider, key, model and paths in a server-side `.env` once. Onboarding for a lab member is a URL and "drop PDFs in the shared Papers folder".
