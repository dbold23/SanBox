import os
import time

from labrag.config import Source
from labrag.embed import HashEmbedder
from labrag.ingest import IndexInProgress, IndexLock, index_sources, scan_folder
from labrag.store import Store


def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_scan_skips_hidden_temp_and_unsupported(tmp_path):
    write(tmp_path / "a.txt", "x")
    write(tmp_path / ".hidden" / "b.txt", "x")
    write(tmp_path / "@eaDir" / "c.txt", "x")
    write(tmp_path / "sub" / "d.md", "x")
    write(tmp_path / "sub" / "e.part", "x")
    write(tmp_path / "sub" / "~$f.docx", "x")
    write(tmp_path / "sub" / ".DS_Store", "x")
    write(tmp_path / "img.jpg", "x")
    names = sorted(p.name for p in scan_folder(tmp_path))
    assert names == ["a.txt", "d.md"]


def test_index_add_update_remove_and_errors(tmp_path):
    root = tmp_path / "papers"
    f1 = write(root / "Smith_2019_sharks.txt", "White sharks eat seals. " * 30)
    f2 = write(root / "sub" / "notes.md", "# Rays\n\nBat rays dig pits. " * 30)
    write(root / "broken.txt", "text")
    store = Store(tmp_path / "i.db")
    emb = HashEmbedder(dim=32)
    src = [Source("nas", root)]
    log = []

    from labrag.parse import parse_file

    def parse(path):
        if path.name == "broken.txt":
            raise ValueError("kaboom")
        return parse_file(path)

    r1 = index_sources(store, emb, src, progress=log.append, parse=parse)
    assert (r1.added, r1.updated, r1.removed, r1.unchanged) == (2, 0, 0, 0)  # the broken file is recorded as an error, not counted as added
    assert r1.errors and r1.errors[0][1] == "kaboom"
    docs = {d.rel_path: d for d in store.list_documents()}
    assert docs["Smith_2019_sharks.txt"].authors == "Smith" and docs["Smith_2019_sharks.txt"].year == 2019
    assert docs["broken.txt"].status == "error"
    assert os.path.join("sub", "notes.md") in docs
    assert store.stats()["chunks"] >= 2

    # nothing changed -> everything unchanged except the errored file, which is retried
    r2 = index_sources(store, emb, src, parse=parse)
    assert r2.added == 0 and r2.updated == 0 and r2.unchanged == 2 and len(r2.errors) == 1

    # modify one, delete one, add one; fix the broken one
    time.sleep(0.01)
    f1.write_text("Completely new text about plankton blooms. " * 30)
    os.utime(f1, None)
    f2.unlink()
    write(root / "new.txt", "Leopard sharks in the estuary. " * 30)
    r3 = index_sources(store, emb, src)  # default parser now succeeds on broken.txt
    assert r3.updated == 2  # f1 and the previously-broken file
    assert r3.added == 1 and r3.removed == 1
    docs = {d.rel_path: d for d in store.list_documents()}
    assert set(docs) == {"Smith_2019_sharks.txt", "broken.txt", "new.txt"}
    assert docs["broken.txt"].status == "ok"
    hits = store.search("plankton", emb.embed_query("plankton"))
    assert hits and hits[0].doc.rel_path == "Smith_2019_sharks.txt"

    # same content copied elsewhere with a new mtime -> hash match, no re-embed
    os.utime(root / "new.txt", (time.time() + 100, time.time() + 100))
    r4 = index_sources(store, emb, src)
    assert r4.unchanged == 3 and r4.updated == 0

    # rebuild wipes and re-adds
    r5 = index_sources(store, emb, src, rebuild=True)
    assert r5.added == 3 and store.stats()["documents"] == 3
    assert store.get_meta("last_indexed")
    store.close()


def test_missing_folder_is_reported_not_fatal(tmp_path):
    store = Store(tmp_path / "i.db")
    log = []
    r = index_sources(store, HashEmbedder(16), [Source("nas", tmp_path / "missing")], progress=log.append)
    assert r.changed == 0 and any("not found" in m for m in log)
    store.close()


def test_empty_folder_does_not_wipe_index(tmp_path):
    root = tmp_path / "nas"
    write(root / "a.txt", "sharks " * 50)
    store = Store(tmp_path / "i.db")
    emb = HashEmbedder(16)
    index_sources(store, emb, [Source("nas", root)])
    assert store.stats()["documents"] == 1
    (root / "a.txt").unlink()  # share unmounted: folder exists but is empty
    log = []
    r = index_sources(store, emb, [Source("nas", root)], progress=log.append)
    assert r.removed == 0 and store.stats()["documents"] == 1
    assert any("not removing" in m for m in log) and r.errors
    # a genuinely emptied folder that still has other files does remove
    write(root / "b.txt", "rays " * 50)
    r2 = index_sources(store, emb, [Source("nas", root)])
    assert r2.added == 1 and r2.removed == 1
    store.close()


def test_scan_survives_symlink_loop(tmp_path):
    root = tmp_path / "papers"
    write(root / "a.txt", "x")
    sub = root / "sub"
    sub.mkdir()
    try:
        (sub / "loop").symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        return  # no symlinks here
    names = sorted(p.name for p in scan_folder(root))
    assert names == ["a.txt"]


def test_index_folder_inside_papers_is_not_scanned(tmp_path):
    root = tmp_path / "papers"
    write(root / "a.txt", "sharks " * 30)
    data = root / "labrag-index"
    write(data / "drive" / "d.txt", "drive copy " * 30)
    store = Store(data / "labrag.db")
    r = index_sources(store, HashEmbedder(16), [Source("nas", root)], exclude=(data,))
    assert r.added == 1 and [d.rel_path for d in store.list_documents()] == ["a.txt"]
    store.close()


def test_moved_file_is_relinked_not_reembedded(tmp_path):
    root = tmp_path / "papers"
    f = write(root / "old" / "a.txt", "leopard sharks " * 30)
    store = Store(tmp_path / "i.db")
    emb = HashEmbedder(16)
    index_sources(store, emb, [Source("nas", root)])
    doc_id = store.list_documents()[0].id
    calls = []

    def counting_parse(path):
        calls.append(path)
        from labrag.parse import parse_file

        return parse_file(path)

    new = root / "new" / "renamed.txt"
    new.parent.mkdir()
    f.rename(new)
    r = index_sources(store, emb, [Source("nas", root)], parse=counting_parse)
    assert not calls  # no re-parse, no re-embed
    assert r.added == 0 and r.removed == 0 and r.unchanged == 1
    docs = store.list_documents()
    assert len(docs) == 1 and docs[0].id == doc_id and docs[0].rel_path == os.path.join("new", "renamed.txt")
    store.close()


def test_index_lock(tmp_path):
    import json

    data = tmp_path / "data"
    with IndexLock(data) as lock:
        assert lock.held
        try:
            with IndexLock(data):
                raise AssertionError("second lock should not be granted")
        except IndexInProgress as exc:
            assert "in progress" in str(exc)
    assert not (data / "labrag.lock").exists()
    # a stale lock (older than the threshold) is removed and taken over
    (data / "labrag.lock").write_text(json.dumps({"pid": 1, "host": "x", "started": "2000-01-01T00:00:00+00:00"}))
    with IndexLock(data):
        pass
    assert not (data / "labrag.lock").exists()


def test_parser_version_bump_reparses_unchanged_files(tmp_path, monkeypatch):
    import labrag.ingest as ing

    root = tmp_path / "papers"
    write(root / "a.txt", "sharks " * 30)
    store = Store(tmp_path / "i.db")
    emb = HashEmbedder(16)
    index_sources(store, emb, [Source("nas", root)])
    assert store.get_meta("parser_version") == str(ing.PARSER_VERSION)
    store.set_meta("parser_version", "1")  # index built by an older LabRAG
    log = []
    r = index_sources(store, emb, [Source("nas", root)], progress=log.append)
    assert r.updated == 1 and r.unchanged == 0
    assert any("parser changed" in m for m in log)
    assert store.get_meta("parser_version") == str(ing.PARSER_VERSION)
    r2 = index_sources(store, emb, [Source("nas", root)])
    assert r2.unchanged == 1 and r2.updated == 0
    store.close()
