"""Shared helpers for the NFL data pipeline."""
import datetime
import os

# Enables SSL verification against the OS trust store instead of only the
# certifi bundle. Needed on machines behind a TLS-inspecting proxy (common on
# corporate networks); harmless elsewhere. Must be imported before nflreadpy.
try:
    import pip_system_certs.wrapt_requests  # noqa: F401
except ImportError:
    pass

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def current_season() -> int:
    """NFL season label for 'today' (a season that starts in September is
    labeled by that calendar year, and runs into the following February)."""
    today = datetime.date.today()
    return today.year if today.month >= 3 else today.year - 1
