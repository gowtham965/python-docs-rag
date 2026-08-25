import subprocess
from pathlib import Path

CPYTHON_REPO = "https://github.com/python/cpython.git"


def fetch_python_docs(dest_dir: str) -> None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git", "clone", "--depth", "1", "--filter=blob:none",
            "--sparse", CPYTHON_REPO, str(dest),
        ],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", "Doc"], cwd=str(dest), check=True)
