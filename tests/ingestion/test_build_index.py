from pyrag.ingestion.build_index import collect_chunks, save_chunks, load_chunks, build_index


def test_collect_chunks_reads_all_rst_files(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.rst").write_text("Section A\n=========\n\nContent A.\n")
    (docs_dir / "b.rst").write_text("Section B\n=========\n\nContent B.\n")

    chunks = collect_chunks(str(docs_dir))
    assert len(chunks) == 2
    sources = {c.source_file for c in chunks}
    assert sources == {"a.rst", "b.rst"}


def test_collect_chunks_skips_unreadable_file_without_crashing(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "good.rst").write_text("Section A\n=========\n\nContent A.\n")
    (docs_dir / "bad.rst").write_text("Section B\n=========\n\nContent B.\n")

    import pyrag.ingestion.build_index as build_index_module
    original_read_text = build_index_module.Path.read_text

    def flaky_read_text(self, *args, **kwargs):
        if self.name == "bad.rst":
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(build_index_module.Path, "read_text", flaky_read_text)

    chunks = collect_chunks(str(docs_dir))
    sources = {c.source_file for c in chunks}
    assert sources == {"good.rst"}


def test_save_and_load_chunks_roundtrip(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.rst").write_text("Section A\n=========\n\nContent A.\n")
    chunks = collect_chunks(str(docs_dir))

    out_path = tmp_path / "chunks.json"
    save_chunks(chunks, str(out_path))
    loaded = load_chunks(str(out_path))

    assert len(loaded) == len(chunks)
    assert loaded[0].text == chunks[0].text


def test_build_index_creates_chroma_collection(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.rst").write_text(
        "Section A\n=========\n\nDictionaries map keys to values.\n"
    )

    chunks_out = tmp_path / "chunks.json"
    chroma_path = tmp_path / "chroma"

    count = build_index(str(docs_dir), str(chunks_out), str(chroma_path))

    assert count == 1
    assert chunks_out.exists()
    assert chroma_path.exists()
