# LabRAG

> **Stack** · Python 3.11+, SQLite (FTS5), fastembed, Claude / Ollama, FastAPI  
> **Data** · not in this repo. Papers stay on the lab NAS and/or Google Drive; the index is one SQLite file next to them.  
> **Upstream** · successor to [SHARK RAG](https://github.com/dbold23/ocean-predator-ecology-lab/tree/main/knowledge-tools/shark-rag) (v1)

**Ask questions about the lab's papers. Get answers with citations to the exact page.**

LabRAG indexes every PDF (and .docx / .md / .txt / .html) in the folders you point it at,
finds the passages that answer a question, and has a language model write a short answer
that cites those passages as `[1]`, `[2]`. Every source is listed with its page, and one click
opens the PDF on that page. Nothing is copied or moved: the papers live where they already live.

```
What tag types have been used on leopard sharks in Elkhorn Slough?

Leopard sharks in Elkhorn Slough have been tracked with acoustic tags: Vemco V13 transmitters
detected by VR2W receivers, with a detection range of about 300 m [1, 2]. Aggregations form in
the upper slough in summer and disperse in winter [1]. No satellite tagging of leopard sharks
appears in the indexed papers.

Sources
 [1] Carlisle 2015 — Leopard shark movements in Elkhorn Slough, p. 4     Papers/Telemetry/Carlisle_2015_leopard.pdf
 [2] Tagging protocol — lab_notes.md, p. 1                                 Papers/lab_notes.md
```

---

## For lab members: 2 minutes, nothing to install

1. Open the LabRAG page in your browser. Someone in the lab runs it on an always-on machine;
   the address looks like `http://labmac.local:8008`. Ask whoever set it up.
2. Type a question the way you would ask a labmate. Follow-up questions work.
3. Every answer lists its sources. Highlighted sources are the ones the answer cites.
   Click a title to open the paper at the cited page. **Check the source before you cite it.**
4. To add papers, drop them into the lab's papers folder on the NAS or into the shared
   Google Drive folder. They are usually searchable within 30 minutes. In a hurry? Click
   **Update index** at the top of the page, then **Update index now**.

If there is no language model configured, LabRAG still works as a search engine over the
lab's papers: you get the best passages and where they came from, without a written summary.

---

## For the person setting it up: about 15 minutes, once

You need a machine that stays on and can see the papers: a lab desktop, a Mac mini, a Linux
box, or the NAS itself if it runs Python. It needs Python 3.11 or newer.

```bash
git clone https://github.com/dbold23/SanBox.git && cd SanBox     # or download the ZIP from GitHub and unzip it
python3 -m pip install --user pipx && pipx ensurepath           # once, if you don't have pipx; then open a new terminal
pipx install ./lab-rag                                            # or: python3 -m pip install ./lab-rag

labrag init        # answers five questions and writes ~/.labrag/labrag.env
labrag index       # first run downloads a 70 MB embedding model, then indexes every paper
labrag serve       # the web page, at http://<this machine>:8008
```

`labrag init` asks for:

| Question | What to answer |
|---|---|
| Folder(s) with the papers | The NAS share as mounted on this machine, e.g. `/Volumes/LabNAS/Papers` or `Z:\Papers`. Several folders: separate with `;`. A Google Drive for Desktop folder works here too. |
| Google Drive folder link | Only if you want LabRAG to pull straight from Drive without Drive for Desktop. See [docs/GOOGLE-DRIVE.md](docs/GOOGLE-DRIVE.md). |
| Where to keep the index | Default is `~/.labrag/data` on this machine, which is the safe choice: only the machine that runs the web page needs the index. |
| Anthropic API key | Optional. With it, Claude writes the answers. Without it LabRAG uses Ollama if it is running, otherwise it is search-only. |
| Password for the web page | Optional. Leave blank on a trusted lab network. |

Re-run `labrag init` any time; saved values are offered as defaults and typing `-` clears one.
Then check everything at once with `labrag doctor`.

Keep it running after a reboot and keep the index fresh: [docs/LAB-SERVER.md](docs/LAB-SERVER.md)
has copy-paste service files for macOS, Linux and Windows. The short version is
`labrag index --every 30` in one window and `labrag serve` in another.

---

## Where the papers can live

| Papers are in... | Do this |
|---|---|
| A folder on the NAS | Mount it on the LabRAG machine and put the path in `LABRAG_FOLDERS`. |
| Google Drive, and the LabRAG machine is a Mac or PC | Install Google Drive for Desktop, mark the folder **Available offline**, and put its path in `LABRAG_FOLDERS`. On macOS that is `~/Library/CloudStorage/GoogleDrive-<you>@csumb.edu/Shared drives/<name>`; on Windows `G:\Shared drives\<name>`. |
| Google Drive, and the LabRAG machine is a Linux server or the NAS | Set `LABRAG_DRIVE_FOLDER` to the folder link plus a credentials file. LabRAG mirrors the folder into its cache and indexes that. Setup: [docs/GOOGLE-DRIVE.md](docs/GOOGLE-DRIVE.md). |
| Several places | List several folders in `LABRAG_FOLDERS` and add a Drive folder too. Each shows up as a named source. |

Subfolders are included. Hidden folders, temp files and files over 200 MB are skipped.
With direct Drive sync, Google Docs and Slides are exported as text and indexed too; Drive
for Desktop only shows them as link files, which are skipped.

---

## How answers are produced, and how much to trust them

1. **Indexing.** Each document is split into passages of about 300 words that remember
   their page numbers. Reference lists are dropped. Every passage is stored twice: as
   words (SQLite FTS5, for exact terms like *V13* or *Notorynchus*) and as an embedding
   vector (for meaning). Title, authors and year come from the DOI via Crossref when a
   DOI is printed on the paper, otherwise from the PDF metadata or the filename.
2. **Retrieval.** A question runs against both indexes; the two rankings are fused and the
   best 8 passages (at most 3 per paper) are selected.
3. **Answering.** The model sees only those numbered passages and is told to cite them and
   to say plainly when they do not contain the answer.
4. **Citations.** `[n]` in the answer is source *n* in the list. The list shows the paper,
   page, and the passage (click *show full passage*), so you can verify every sentence.

The model can still misread a passage. Treat LabRAG as a very fast reader who hands you the
page, not as the last word.

---

## Which model writes the answers

| Setting | What happens | Privacy |
|---|---|---|
| `ANTHROPIC_API_KEY` set | Claude writes the answers (default model `claude-opus-5`; set `LABRAG_LLM_MODEL=claude-sonnet-5` for a cheaper one). A question costs a few cents. | The question and the retrieved passages are sent to Anthropic. The full papers never are. |
| `OPENAI_API_KEY` set | OpenAI or any compatible server (`LABRAG_OPENAI_BASE_URL`). | As for Claude. |
| Ollama running on the machine | Answers from a local model (default `llama3.1`). Free, slower, needs a machine with 16 GB RAM. | Nothing leaves the building. |
| None of the above | Search-only mode: passages and sources, no written answer. | Nothing leaves the building. |

Checked in that order when `LABRAG_LLM` is left on `auto`.

Embeddings are computed on the LabRAG machine by default (`fastembed`, model
`BAAI/bge-small-en-v1.5`, downloaded once). Ollama or OpenAI embeddings are available with
`LABRAG_EMBED`. The embedding model is stamped on the index; changing it later requires
`labrag index --rebuild`, and LabRAG will tell you so instead of silently returning nonsense.

Claude requests are sent with the server-side refusal fallback enabled, so on the rare
occasion a safety classifier declines a request, the API retries it on a fallback model
inside the same call rather than failing.

---

## Commands

| Command | What it does |
|---|---|
| `labrag init` | Set up by answering a few questions. Re-run any time; existing values are kept as defaults. |
| `labrag index` | Sync Google Drive (if configured), then add new, re-index changed and forget deleted files. `--rebuild` starts over. `--every 30` keeps running and checks every 30 minutes. `--no-drive` skips the Drive sync. |
| `labrag serve` | Start the web page for the lab. `--port 8010` to change the port. |
| `labrag ask "..."` | Ask from the terminal. |
| `labrag search "..."` | Show matching passages only, no model. |
| `labrag status` | What is indexed, from where, with which model. `-v` lists files that failed or need OCR (the web page lists them too, under **Update index**). |
| `labrag doctor` | Check Python, folders, Drive access, embeddings and the model in one go. |

Settings come from environment variables or from `~/.labrag/labrag.env` (or a `labrag.env` in
the current folder). Real environment variables win. Every key is documented in
[labrag.env.example](labrag.env.example).

---

## One index for the lab

The normal setup needs no sharing at all: one machine runs `labrag index` and `labrag serve`,
and the index lives on that machine's disk. Everyone uses the web page.

If other people want to run `labrag ask` from their own laptops against the same index, put
`LABRAG_DATA` on the NAS instead and point their settings at it. Only one machine may run
`labrag index` against a shared index, and the index is deliberately kept in SQLite's plain
journal mode (no WAL) so that it works on a network share. If the share is flaky, keep the
index local.

If the NAS is unmounted when an index run happens, LabRAG notices that the folder is empty
and refuses to remove anything.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `labrag status -v` lists PDFs under "scanned PDFs without text" | They are images. Run `ocrmypdf in.pdf out.pdf` (or Adobe's OCR) and replace the file; the next index picks it up. |
| "This index was built with embeddings from X but the configured embedder is Y" | You changed `LABRAG_EMBED`. Either change it back or run `labrag index --rebuild`. |
| "Folder ... does not exist (is the NAS mounted?)" | Mount the share. Nothing is removed from the index while it is missing. |
| "Could not reach Ollama" | Start Ollama (`ollama serve` or the desktop app) and `ollama pull llama3.1`. |
| "Anthropic rejected the API key" | Re-run `labrag init` and paste the key again, or edit `~/.labrag/labrag.env`. |
| First `labrag index` seems stuck | It is downloading the embedding model (about 70 MB) from huggingface.co. Campus proxies sometimes block it; `LABRAG_EMBED=hash` gets you going with keyword-only quality until it works. |
| "Another LabRAG index run is in progress" | `labrag index --every` and the page's Update button share one lock, so only one runs at a time. Wait for it; the page header shows "indexing…" meanwhile. |
| The model "ran out of tokens while reasoning" | Raise `LABRAG_MAX_TOKENS` (default 16000) or set `LABRAG_LLM_EFFORT=low`. |
| Port 8008 is taken | `labrag serve --port 8010`, or set `LABRAG_PORT`. |
| A paper's title or authors look wrong | The paper has no DOI on its first page and poor PDF metadata. Rename the file to `Author_Year_Title.pdf`; the filename is the fallback. |

---

## What changed from SHARK RAG v1

| v1 | LabRAG |
|---|---|
| Every lab member installed a 3-6 GB stack (Docling, PyTorch, LlamaIndex, LanceDB, Gradio) and built a private index in the git checkout | One machine runs it; everyone else uses a web page. Install is ~400 MB, index is one SQLite file that can live on the NAS |
| Seven tabs, eleven settings, five LLM providers x three embedding providers selectable in the UI | No settings in the UI. The maintainer picks a model once |
| Ollama required by default | Local embeddings, any of Claude / Ollama / OpenAI for answers, or no model at all |
| Sync folder re-typed after every restart; Drive path only via Drive for Desktop | Folders and Drive are configured once; Drive can be read directly with a service account |
| Path+mtime dedup, duplicates on rename, reference lists indexed, citations like `paper (, 0) - pp. 1-,-2` | Content-hash identity, re-index on change, forget on delete, reference lists dropped, citations `Jorgensen 2010, p. 4` from Crossref |
| Vector-only retrieval | Keyword + vector, fused |
| Paper acquisition crawler, markdown importer, summary tab, reading status | Removed. Drop PDFs in the folder |

---

## Development

```bash
cd lab-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Layout follows the lab archive conventions: package `labrag/`, `tests/`, `docs/`. Tests
run offline; embeddings use a deterministic hash embedder and the LLM is faked.

| Module | Role |
|---|---|
| `parse.py` | file → text + title/authors/year/DOI (PDF, DOCX, TXT, MD, HTML) |
| `chunk.py` | text → ~300-word passages with page numbers; drops reference lists |
| `embed.py` | fastembed / Ollama / OpenAI / hash embedders |
| `store.py` | the SQLite index: documents, passages, FTS5, vectors, hybrid search |
| `ingest.py` | folder scan, change detection, parse → chunk → embed → store |
| `drive.py` | Google Drive folder → local mirror |
| `lookup.py` | DOI → Crossref metadata |
| `llm.py`, `engine.py` | answer generation with numbered citations |
| `config.py`, `providers.py` | settings and provider factories |
| `cli.py`, `web.py`, `static/index.html` | the command line and the web page |

---

## Citation

If LabRAG helped produce published work, please cite it; see [CITATION.cff](CITATION.cff).
Licensed under MIT, like the rest of the lab archive.
