"""Smoke-test: every page renders without exceptions.  Run: python -m pytest tests -q"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "app" / "views").glob("*.py"))

TEMPLATE = """
import sys, runpy
sys.path.insert(0, r"{app}")
import streamlit as st
from core.ui import apply_theme, get_df
apply_theme(); get_df()
runpy.run_path(r"{page}", run_name="__main__")
"""


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_renders(page):
    at = AppTest.from_string(TEMPLATE.format(app=ROOT / "app", page=page), default_timeout=240)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
