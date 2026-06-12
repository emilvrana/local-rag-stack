"""Tests for batch_ingest.py"""

import tempfile
from pathlib import Path

from batch_ingest import collect_files, load_file, SUPPORTED_EXTENSIONS


class TestCollectFiles:
    def test_single_supported_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        result = collect_files(str(f))
        assert result == [str(f)]

    def test_unsupported_extension_skipped(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")
        result = collect_files(str(f))
        assert result == []

    def test_directory_non_recursive(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.md").write_text("b")
        (tmp_path / "c.png").write_bytes(b"x")
        result = collect_files(str(tmp_path), recursive=False)
        names = [Path(r).name for r in result]
        assert "a.txt" in names
        assert "b.md" in names
        assert "c.png" not in names

    def test_directory_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep")
        (tmp_path / "top.md").write_text("top")
        result = collect_files(str(tmp_path), recursive=True)
        assert len(result) == 2

    def test_nonexistent_path(self):
        result = collect_files("/nonexistent/path/12345")
        assert result == []

    def test_supported_extensions(self):
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".pdf" not in SUPPORTED_EXTENSIONS


class TestLoadFile:
    def test_utf8_file(self, tmp_path):
        f = tmp_path / "utf8.txt"
        f.write_text("Příliš žluťoučký kůň", encoding="utf-8")
        content, source = load_file(str(f))
        assert "kůň" in content
        assert source == "utf8.txt"

    def test_latin1_file(self, tmp_path):
        f = tmp_path / "latin.txt"
        f.write_bytes("café".encode("latin-1"))
        content, source = load_file(str(f))
        assert len(content) > 0