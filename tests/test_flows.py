"""Interactive flows: agent question, projection save, report generation."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
T = """
import sys, runpy
sys.path.insert(0, r"{app}")
from core.ui import apply_theme, get_df
apply_theme(); get_df()
runpy.run_path(r"{page}", run_name="__main__")
"""


def app(page):
    return AppTest.from_string(T.format(app=ROOT / "app", page=ROOT / "app" / "views" / page), default_timeout=240)


def test_agent_answers():
    at = app("ai_agent.py")
    at.run()
    at.chat_input[0].set_value("compare Lahore and Karachi Central").run()
    assert not at.exception
    txt = " ".join(m.value for m in at.markdown)
    assert "Lahore" in txt and "13.00 M" in txt


def test_projector_and_report():
    at = app("m8_projector.py")
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["projections"][0]["district"] == "Sheikhupura"
    assert any(t.startswith("Projection") for t, _ in at.session_state["session_images"])


def test_report_pdf():
    at = app("m9_report.py")
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Report ready" in s.value for s in at.success)
