from labrag.chunk import chunk_pages, drop_references


def words(n, tag="w"):
    return " ".join(f"{tag}{i}" for i in range(n))


def test_short_document_is_one_chunk():
    chunks = chunk_pages(["Just a few words here."])
    assert len(chunks) == 1
    assert chunks[0].page_start == chunks[0].page_end == 1


def test_packs_paragraphs_with_overlap_and_pages():
    pages = [
        "\n\n".join(words(120, f"p{i}_") for i in range(3)),  # 360 words on page 1
        "\n\n".join(words(120, f"q{i}_") for i in range(3)),  # 360 words on page 2
    ]
    chunks = chunk_pages(pages, target_words=250, overlap_words=30, min_words=40)
    assert len(chunks) >= 3
    for c in chunks:
        assert c.n_words <= 250 + 130  # a paragraph may push slightly over target
    # overlap: the first words of chunk 2 are the last words of chunk 1
    tail = chunks[0].text.split()[-30:]
    assert chunks[1].text.split()[:30] == tail
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert [c.idx for c in chunks] == list(range(len(chunks)))


def test_long_paragraph_is_split_at_sentences():
    para = " ".join(f"Sentence number {i} says something about sharks." for i in range(120))  # ~840 words
    chunks = chunk_pages([para], target_words=200, overlap_words=0)
    assert len(chunks) >= 4
    assert all(c.n_words <= 260 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_references_are_dropped():
    body = "\n\n".join(words(100, f"b{i}_") for i in range(5))
    refs = "References\n\nSmith J (2019) A paper. Journal 1:1-10.\n\nJones K (2020) Another paper."
    chunks = chunk_pages([body + "\n\n" + refs])
    assert not any("Smith J (2019)" in c.text for c in chunks)
    # but an early 'References' heading (e.g. in a table of contents) is not treated as the end
    chunks2 = chunk_pages(["References\n\n" + body])
    assert sum(c.n_words for c in chunks2) >= 400


def test_no_words_no_chunks():
    assert chunk_pages(["", "   "]) == []


def test_references_detected_at_line_level_across_pages():
    body = words(400, "b")
    pages = [body, "Last results.\nReferences\nSmith J (2019) A paper.\nJones K (2020) Another.", "More refs (2021)."]
    kept = drop_references(pages)
    assert kept == [body, "Last results."]
    assert chunk_pages(pages) and not any("Smith J" in c.text for c in chunk_pages(pages))
    # heading too early (a table of contents) is ignored
    assert drop_references(["References\n" + body]) == ["References\n" + body]
    assert drop_references(["", None]) == ["", None]
