"""Test markdown code blocks"""
import pathlib
from mktestdocs import check_md_file

def test_docs():
    fpath = pathlib.Path("docs") / "index.md"
    check_md_file(fpath=fpath)
