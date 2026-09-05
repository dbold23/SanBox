"""labrag: index the lab's papers and ask questions about them.

    labrag init      set up (asks a few questions, writes ~/.labrag/labrag.env)
    labrag index     bring the index up to date (syncs Google Drive first)
    labrag ask       ask a question from the terminal
    labrag search    just find passages, no model
    labrag serve     start the web page for the whole lab
    labrag status    what is indexed, what is configured
    labrag doctor    check that everything works
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from pathlib import Path

from . import __version__
from .config import DEFAULT_ENV_NAME, DEFAULT_USER_DIR, Settings, load_settings, read_env_file, write_env_file
from .engine import Answer, Engine
from .ingest import IndexReport, index_sources
from .providers import DEFAULT_MODELS, ProviderError, make_embedder, make_llm, resolve_llm_kind
from .store import EmbedderMismatch, Store

log = logging.getLogger("labrag")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str, code: int = 1) -> "int":
    print(f"\nProblem: {msg}", file=sys.stderr, flush=True)
    return code


# ----------------------------------------------------------------------------- init


def cmd_init(args) -> int:
    target = Path(args.file) if args.file else DEFAULT_USER_DIR / DEFAULT_ENV_NAME
    current = read_env_file(target) if target.exists() else {}
    say("LabRAG setup. Press Enter to accept the value in [brackets]; leave blank to skip.\n")

    def ask(key: str, prompt: str, default: str = "", secret: bool = False) -> str:
        cur = current.get(key, default)
        shown = ("*" * 8 if cur and secret else cur)
        label = f"{prompt}" + (f" [{shown}]" if shown else "") + ": "
        try:
            value = input(label).strip()
        except (EOFError, OSError):  # no terminal attached
            value = ""
        return value or cur

    folders = ask("LABRAG_FOLDERS", "1. Folder(s) with the lab's papers. NAS share, Google Drive for Desktop folder, any folder.\n   Separate several with ';'")
    drive = ask("LABRAG_DRIVE_FOLDER", "2. Google Drive folder link to sync directly from Drive (optional)")
    creds_sa, creds_oauth = current.get("LABRAG_GOOGLE_SERVICE_ACCOUNT", ""), current.get("LABRAG_GOOGLE_CLIENT_SECRET", "")
    if drive:
        creds = ask("LABRAG_GOOGLE_CREDENTIALS", "   Path to the Google credentials JSON (service-account key or OAuth client secret)", creds_sa or creds_oauth)
        kind = _google_credentials_kind(creds) if creds else None
        if kind == "service_account":
            creds_sa, creds_oauth = creds, ""
        elif kind == "oauth":
            creds_sa, creds_oauth = "", creds
        elif creds:
            say("   (could not tell what kind of credentials file that is; saving it as a service-account key)")
            creds_sa, creds_oauth = creds, ""
    first = folders.split(";")[0].strip() if folders else ""
    suggested = str(Path(first).expanduser().parent / "labrag-index") if first else str(DEFAULT_USER_DIR / "data")
    data = ask("LABRAG_DATA", f"3. Where to keep the index. On the NAS, next to the papers, lets the whole lab share it", suggested)
    key = ask("ANTHROPIC_API_KEY", "4. Anthropic API key so answers are written by Claude (optional; without it LabRAG\n   uses Ollama if it is running, otherwise it works as a search engine)", secret=True)
    password = ask("LABRAG_PASSWORD", "5. Password for the web page (optional; leave blank on a trusted lab network)", secret=True)

    values = dict(current)
    values.update({
        "LABRAG_FOLDERS": folders,
        "LABRAG_DRIVE_FOLDER": drive,
        "LABRAG_GOOGLE_SERVICE_ACCOUNT": creds_sa,
        "LABRAG_GOOGLE_CLIENT_SECRET": creds_oauth,
        "LABRAG_DATA": data,
        "ANTHROPIC_API_KEY": key,
        "LABRAG_PASSWORD": password,
    })
    values.pop("LABRAG_GOOGLE_CREDENTIALS", None)
    write_env_file(target, values)
    say(f"\nSaved to {target}\n")
    say("Next:\n  labrag index     # build the index (the first run downloads a small embedding model)\n  labrag ask \"what do white sharks eat near Ano Nuevo?\"\n  labrag serve     # web page for the lab at http://<this machine>:8008")
    return 0


def _google_credentials_kind(path: str) -> str | None:
    try:
        data = json.loads(Path(path).expanduser().read_text())
    except Exception:
        return None
    if data.get("type") == "service_account":
        return "service_account"
    if "installed" in data or "web" in data:
        return "oauth"
    return None


# ---------------------------------------------------------------------------- shared


def _settings(args) -> Settings:
    s = load_settings(env_file=getattr(args, "env", None))
    return s


def _open_store(settings: Settings) -> Store:
    return Store(settings.db_path)


def sync_drive(settings: Settings, progress=say):
    """Mirror the configured Drive folder into the cache. Returns a SyncResult or None."""
    if not settings.drive_folder:
        return None
    from . import drive

    folder_id = drive.parse_folder_id(settings.drive_folder)
    creds = drive.load_credentials(settings.google_service_account, settings.google_client_secret, settings.google_token_file)
    client = drive.DriveClient(drive.authorized_session(creds))
    progress(f"[drive] syncing Google Drive folder {folder_id} -> {settings.drive_cache_dir}")
    result = drive.sync_folder(client, folder_id, settings.drive_cache_dir)
    progress(f"[drive] {result.summary()}")
    for err in result.errors[:10]:
        progress(f"[drive]   could not download: {err}")
    return result


def run_index(settings: Settings, embedder, *, rebuild: bool = False, with_drive: bool = True, progress=say, store: Store | None = None) -> IndexReport:
    if with_drive and settings.drive_folder:
        try:
            sync_drive(settings, progress)
        except Exception as exc:
            progress(f"[drive] sync failed, indexing what is already cached: {exc}")
    lookup = None
    if settings.lookup:
        from .lookup import CrossrefLookup

        lookup = CrossrefLookup()
    own_store = store is None
    store = store or _open_store(settings)
    try:
        report = index_sources(store, embedder, settings.all_sources(), rebuild=rebuild, progress=progress, lookup=lookup)
    finally:
        if own_store:
            store.close()
    return report


def print_report(report: IndexReport) -> None:
    say(f"\nIndex: {report.summary()}")
    if report.needs_ocr:
        say("\nThese PDFs have no text layer (scanned images). Run them through OCR (e.g. `ocrmypdf in.pdf out.pdf`) to make them searchable:")
        for p in report.needs_ocr[:20]:
            say(f"  - {p}")
        if len(report.needs_ocr) > 20:
            say(f"  ... and {len(report.needs_ocr) - 20} more (see `labrag status`)")
    if report.errors:
        say("\nThese files could not be indexed:")
        for path, err in report.errors[:20]:
            say(f"  - {path}: {err}")


# ----------------------------------------------------------------------------- index


def cmd_index(args) -> int:
    settings = _settings(args)
    problems = settings.problems()
    if not settings.all_sources():
        return fail(problems[0])
    for p in problems:
        say(f"Warning: {p}")
    try:
        embedder = make_embedder(settings)
    except ProviderError as exc:
        return fail(str(exc))
    say(f"Embeddings: {embedder.name}  |  Index: {settings.db_path}")
    while True:
        try:
            report = run_index(settings, embedder, rebuild=args.rebuild, with_drive=not args.no_drive)
        except EmbedderMismatch as exc:
            return fail(str(exc))
        print_report(report)
        if not args.every:
            return 0
        say(f"\nNext check in {args.every} minutes (Ctrl-C to stop).")
        try:
            time.sleep(args.every * 60)
        except KeyboardInterrupt:
            return 0
        args.rebuild = False


# ------------------------------------------------------------------------------- ask


def _engine(settings: Settings, need_llm: bool = True) -> Engine:
    if not settings.db_path.exists():
        raise ProviderError(f"No index yet at {settings.db_path}. Run `labrag index` first.")
    store = _open_store(settings)
    embedder = None
    if store.embedding_model and store.embedding_model != "none":
        try:
            embedder = make_embedder(settings)
        except ProviderError as exc:
            say(f"Warning: {exc}\nFalling back to keyword search.")
    llm = None
    if need_llm:
        llm = make_llm(settings)
    return Engine(store, embedder, llm, k=settings.top_k, max_tokens=settings.max_tokens)


def print_answer(ans: Answer, show_text: bool = True) -> None:
    if ans.error:
        say(f"Could not get an answer from the model: {ans.error}\nHere are the most relevant passages instead.\n")
    elif ans.text and show_text:
        say(textwrap.fill(ans.text, width=100, replace_whitespace=False) if "\n" not in ans.text else ans.text)
        say()
    if not ans.sources:
        say("No matching passages.")
        return
    say("Sources:" + ("  (* = cited in the answer)" if ans.cited else ""))
    for i, hit in enumerate(ans.sources, start=1):
        star = "*" if i in ans.cited else " "
        doc = hit.doc
        where = f", {hit.pages}" if hit.pages else ""
        label = doc.short_citation
        title = doc.title or doc.rel_path
        head = title if label.startswith(title[:40]) else f"{label}: {title}"
        say(f" {star}[{i}] {head}{where}")
        say(f"       {doc.source}/{doc.rel_path}")
        if not ans.text or ans.error:
            snippet = " ".join(hit.text.split())[:300]
            say(f"       \"{snippet}...\"")
    if ans.model:
        say(f"\n({ans.model}, {ans.elapsed:.1f}s)")


def cmd_ask(args) -> int:
    settings = _settings(args)
    try:
        engine = _engine(settings, need_llm=True)
    except ProviderError as exc:
        return fail(str(exc))
    question = " ".join(args.question).strip()
    if not question:
        return fail("Give me a question, e.g.  labrag ask \"how deep do white sharks dive?\"")
    if engine.llm is None:
        say("No language model configured (set ANTHROPIC_API_KEY or start Ollama) - showing the best passages.\n")
    ans = engine.ask(question, k=args.k)
    print_answer(ans)
    return 0


def cmd_search(args) -> int:
    settings = _settings(args)
    try:
        engine = _engine(settings, need_llm=False)
    except ProviderError as exc:
        return fail(str(exc))
    question = " ".join(args.question).strip()
    ans = engine.ask(question, k=args.k)
    print_answer(ans)
    return 0


# ---------------------------------------------------------------------------- status


def cmd_status(args) -> int:
    settings = _settings(args)
    say(f"LabRAG {__version__}")
    say(f"Settings file: {settings.env_file or '(none found - run `labrag init`)'}")
    say(f"Index:         {settings.db_path}" + ("" if settings.db_path.exists() else "  (not built yet)"))
    for s in settings.folders:
        say(f"Folder:        {s.name} = {s.root}" + ("" if s.root.exists() else "  (NOT FOUND)"))
    if settings.drive_folder:
        say(f"Google Drive:  {settings.drive_folder} -> {settings.drive_cache_dir}")
    kind = resolve_llm_kind(settings)
    say(f"Answers:       {kind}" + (f" ({settings.llm_model or DEFAULT_MODELS.get(kind)})" if kind != "none" else " (search only - no model configured)"))
    say(f"Embeddings:    {settings.embed}" + (f" ({settings.embed_model})" if settings.embed_model else ""))
    for p in settings.problems():
        say(f"Warning:       {p}")
    if settings.db_path.exists():
        store = _open_store(settings)
        st = store.stats()
        say(f"\n{st['documents']} documents, {st['chunks']} passages, last indexed {st['last_indexed'] or 'never'}")
        for name, n in sorted(st["sources"].items()):
            say(f"  {name}: {n}")
        if st["needs_ocr"]:
            say(f"  {st['needs_ocr']} scanned PDFs without text (not searchable until OCR'd)")
        if st["errors"]:
            say(f"  {st['errors']} files failed to index")
        if args.verbose:
            for d in store.list_documents():
                if d.status != "ok":
                    say(f"    [{d.status}] {d.source}/{d.rel_path}" + (f": {d.error}" if d.error else ""))
        store.close()
    return 0


# ----------------------------------------------------------------------------- serve


def cmd_serve(args) -> int:
    settings = _settings(args)
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    try:
        from .web import create_app

        app = create_app(settings)
    except ProviderError as exc:
        return fail(str(exc))
    import uvicorn

    say(f"LabRAG is at http://{_display_host(settings.host)}:{settings.port}  (Ctrl-C to stop)")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
    return 0


def _display_host(host: str) -> str:
    if host in ("0.0.0.0", "::"):
        import socket

        try:
            return socket.gethostname()
        except Exception:
            return "localhost"
    return host


# ---------------------------------------------------------------------------- doctor


def cmd_doctor(args) -> int:
    settings = _settings(args)
    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        say(f"  [{'ok' if good else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))

    say("Checking LabRAG...\n")
    check(f"Python {sys.version_info.major}.{sys.version_info.minor}", sys.version_info >= (3, 11), "" if sys.version_info >= (3, 11) else "3.11 or newer is required")
    configured = bool(settings.all_sources())
    if settings.env_file:
        check("settings", True, f"from {settings.env_file}")
    else:
        check("settings", configured, "from environment variables" if configured else "no settings file and no LABRAG_FOLDERS; run `labrag init`")
    for p in settings.folders:
        check(f"folder '{p.name}'", p.root.exists(), str(p.root))
    if settings.drive_folder:
        try:
            from . import drive

            fid = drive.parse_folder_id(settings.drive_folder)
            creds = drive.load_credentials(settings.google_service_account, settings.google_client_secret, settings.google_token_file)
            client = drive.DriveClient(drive.authorized_session(creds))
            n = sum(1 for _ in client.walk(fid))
            check("Google Drive", True, f"{n} indexable files in folder {fid}")
        except Exception as exc:
            check("Google Drive", False, str(exc)[:300])
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.data_dir / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        check("index folder writable", True, str(settings.data_dir))
    except Exception as exc:
        check("index folder writable", False, f"{settings.data_dir}: {exc}")
    try:
        emb = make_embedder(settings)
        v = emb.embed_query("test")
        check("embeddings", True, f"{emb.name}, {len(v)} dimensions")
    except Exception as exc:
        check("embeddings", False, str(exc)[:300])
    kind = resolve_llm_kind(settings)
    if kind == "none":
        check("language model", True, "none configured - search-only mode (set ANTHROPIC_API_KEY or start Ollama for written answers)")
    else:
        try:
            llm = make_llm(settings)
            reply = llm.complete("Reply with the single word OK.", "Ready?", max_tokens=2000)
            check("language model", bool(reply), f"{llm.name} replied: {reply[:60]!r}")
        except Exception as exc:
            check("language model", False, str(exc)[:300])
    say("\nAll good." if ok else "\nSomething needs attention (see FAIL lines above).")
    return 0 if ok else 1


# ------------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="labrag", description="Ask questions about the lab's papers, with citations.")
    p.add_argument("--env", help="settings file to use (default: ./labrag.env or ~/.labrag/labrag.env)")
    p.add_argument("-v", "--verbose", action="store_true", help="show more detail")
    p.add_argument("--version", action="version", version=f"labrag {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="set up LabRAG by answering a few questions")
    s.add_argument("--file", help="where to write the settings (default ~/.labrag/labrag.env)")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("index", help="bring the index up to date")
    s.add_argument("--rebuild", action="store_true", help="throw the index away and re-index everything")
    s.add_argument("--no-drive", action="store_true", help="skip the Google Drive sync")
    s.add_argument("--every", type=float, metavar="MINUTES", help="keep running and re-check every N minutes")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("ask", help="ask a question")
    s.add_argument("question", nargs="+")
    s.add_argument("-k", type=int, help="number of passages to use")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("search", help="find passages without using a language model")
    s.add_argument("question", nargs="+")
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("status", help="show what is indexed and configured")
    s.add_argument("-v", "--verbose", action="store_true", help="also list files that failed or need OCR")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("serve", help="start the web page")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("doctor", help="check that everything works")
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
