from unittest.mock import patch

from pyrag.ingestion.fetch_docs import fetch_python_docs, CPYTHON_REPO


def test_fetch_python_docs_runs_sparse_clone(tmp_path):
    dest = tmp_path / "cpython"
    with patch("pyrag.ingestion.fetch_docs.subprocess.run") as mock_run:
        fetch_python_docs(str(dest))

    clone_call = mock_run.call_args_list[0]
    assert "clone" in clone_call.args[0]
    assert CPYTHON_REPO in clone_call.args[0]

    sparse_call = mock_run.call_args_list[1]
    assert sparse_call.args[0] == ["git", "sparse-checkout", "set", "Doc"]
    assert sparse_call.kwargs["cwd"] == str(dest)
