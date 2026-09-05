from pathlib import Path

from labrag.config import load_settings, parse_folders, read_env_file, write_env_file


def test_parse_folders_names_and_dedup(tmp_path):
    a = tmp_path / "Papers"
    a.mkdir()
    src = parse_folders(f"{a}; nas=/mnt/nas/Lab Papers ; {a}")
    assert [s.name for s in src] == ["Papers", "nas", "Papers2"]
    assert src[1].root == Path("/mnt/nas/Lab Papers")


def test_env_file_roundtrip_and_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / "labrag.env"
    write_env_file(env_file, {"LABRAG_FOLDERS": "/a/b;/c d", "ANTHROPIC_API_KEY": "sk-123", "LABRAG_PORT": "9000", "LABRAG_LLM": ""})
    text = env_file.read_text()
    assert "# Folders with papers" in text and 'LABRAG_FOLDERS="/a/b;/c d"' in text
    values = read_env_file(env_file)
    assert values["LABRAG_FOLDERS"] == "/a/b;/c d" and values["LABRAG_PORT"] == "9000" and values["LABRAG_LLM"] == ""

    s = load_settings(env={"LABRAG_PORT": "8123"}, env_file=str(env_file))
    assert [str(x.root) for x in s.folders] == ["/a/b", "/c d"]
    assert s.port == 8123  # real env wins
    assert s.anthropic_api_key == "sk-123"
    assert s.llm == "auto"  # blank in file -> default
    assert s.env_file == env_file
    assert s.db_path == s.data_dir / "labrag.db"


def test_problems_lists_missing_things(tmp_path):
    s = load_settings(env={"LABRAG_FOLDERS": str(tmp_path / "nope"), "LABRAG_DRIVE_FOLDER": "abc123abc123"}, env_file=str(tmp_path / "none.env"))
    probs = s.problems()
    assert any("does not exist" in p for p in probs)
    assert any("LABRAG_GOOGLE_SERVICE_ACCOUNT" in p for p in probs)
    assert len(s.all_sources()) == 2 and s.all_sources()[1].name == "drive"
    assert load_settings(env={}, env_file=str(tmp_path / "none.env")).problems()
