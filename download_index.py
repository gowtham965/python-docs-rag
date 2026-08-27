"""Downloads the prebuilt search index (chunks + Chroma vector store) from
the public Hugging Face dataset Gowtham8Ai/python-docs-rag-index.

Run at Docker build time only. chroma.sqlite3 alone is 110MB, over
GitHub's 100MB file-size limit, so the index is hosted on HF instead of
committed to this repo.
"""

import os
import urllib.request

BASE_URL = "https://huggingface.co/datasets/Gowtham8Ai/python-docs-rag-index/resolve/main"

FILES = {
    "chunks.json": "data/processed/chunks.json",
    "chroma/chroma.sqlite3": "data/chroma/chroma.sqlite3",
    "chroma/479580c9-a005-458f-b71f-1b974dbd14c6/data_level0.bin": (
        "data/chroma/479580c9-a005-458f-b71f-1b974dbd14c6/data_level0.bin"
    ),
    "chroma/479580c9-a005-458f-b71f-1b974dbd14c6/header.bin": (
        "data/chroma/479580c9-a005-458f-b71f-1b974dbd14c6/header.bin"
    ),
    "chroma/479580c9-a005-458f-b71f-1b974dbd14c6/index_metadata.pickle": (
        "data/chroma/479580c9-a005-458f-b71f-1b974dbd14c6/index_metadata.pickle"
    ),
    "chroma/479580c9-a005-458f-b71f-1b974dbd14c6/length.bin": (
        "data/chroma/479580c9-a005-458f-b71f-1b974dbd14c6/length.bin"
    ),
    "chroma/479580c9-a005-458f-b71f-1b974dbd14c6/link_lists.bin": (
        "data/chroma/479580c9-a005-458f-b71f-1b974dbd14c6/link_lists.bin"
    ),
}


def main():
    for remote_path, local_path in FILES.items():
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = f"{BASE_URL}/{remote_path}"
        print(f"downloading {url} -> {local_path}")
        urllib.request.urlretrieve(url, local_path)


if __name__ == "__main__":
    main()
