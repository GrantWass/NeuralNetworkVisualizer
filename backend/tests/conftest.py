import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SESSION_BACKEND", "local")

import pytest


@pytest.fixture()
def clean_local_state():
    import app

    app._local_sessions.clear()
    app._session_cache.clear()
    yield app
