"""Optional web adapters for Lab features.

Desktop (Tkinter) is the default UI.
Add Streamlit (or other web) pages here only when a feature is
explicitly requested for the browser.

Rules:
  - Import algorithms from `core` (or call `runtimes/vision` via CLI).
  - Do not reimplement filters / detection in the adapter.
  - Install web deps with: uv sync --group web
"""
