import os

from labrag.cli import main


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "Smith_2019_sharks.txt").write_text("White sharks eat seals near Ano Nuevo. " * 20)
    (papers / "notes.md").write_text("# Tagging protocol\n\nWe use V13 acoustic tags on leopard sharks.\n")
    env = {
        "LABRAG_FOLDERS": str(papers),
        "LABRAG_DATA": str(tmp_path / "index"),
        "LABRAG_EMBED": "hash",
        "LABRAG_LLM": "none",
        "LABRAG_LOOKUP": "off",
        "HOME": str(tmp_path / "home"),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(tmp_path)

    assert main(["index"]) == 0
    out = capsys.readouterr().out
    assert "2 added" in out
    assert main(["index"]) == 0
    assert "2 unchanged" in capsys.readouterr().out

    assert main(["search", "acoustic", "tags"]) == 0
    out = capsys.readouterr().out
    assert "[1] Tagging protocol, p. 1" in out and "notes.md" in out

    assert main(["ask", "what do white sharks eat?"]) == 0
    out = capsys.readouterr().out
    assert "No language model configured" in out and "Smith 2019: Smith 2019 sharks" in out

    assert main(["status", "-v"]) == 0
    out = capsys.readouterr().out
    assert "2 documents" in out and "search only" in out

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "[ok] settings - from environment variables" in out and "All good." in out

    # init: answers come from input(); simulate a user who accepts every default
    answers = iter([str(papers), "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers, ""))
    assert main(["init", "--file", str(tmp_path / "labrag.env")]) == 0
    text = (tmp_path / "labrag.env").read_text()
    assert f"LABRAG_FOLDERS={papers}" in text and "ANTHROPIC_API_KEY=" in text
    assert "LABRAG_DATA=" in text


def test_cli_index_without_sources_fails_clearly(tmp_path, capsys, monkeypatch):
    for k in list(os.environ):
        if k.startswith("LABRAG_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert main(["index"]) == 1
    assert "No papers configured" in capsys.readouterr().err
    assert main(["ask", "anything"]) == 1
    assert "No index yet" in capsys.readouterr().err
