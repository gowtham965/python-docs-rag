from pyrag.ingestion.chunker import chunk_rst_file, split_into_sections


def test_split_into_sections_detects_headings():
    text = (
        "Intro text here.\n\n"
        "Section One\n"
        "===========\n\n"
        "Body of section one.\n\n"
        "Section Two\n"
        "-----------\n\n"
        "Body of section two.\n"
    )
    sections = split_into_sections(text)
    headings = [h for h, _ in sections]
    assert "Section One" in headings
    assert "Section Two" in headings


def test_chunk_rst_file_produces_chunks_with_metadata():
    text = "My Section\n==========\n\nSome content about Python.\n"
    chunks = chunk_rst_file(text, source_file="library/example.rst")
    assert len(chunks) == 1
    assert chunks[0].section_title == "My Section"
    assert chunks[0].source_file == "library/example.rst"
    assert "Python" in chunks[0].text


def test_chunk_rst_file_splits_long_sections():
    long_body = " ".join(["word"] * 900)
    text = f"Big Section\n===========\n\n{long_body}\n"
    chunks = chunk_rst_file(text, source_file="library/big.rst")
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.section_title == "Big Section"
