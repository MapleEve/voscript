# tests/aspirational/conftest.py
import pytest

# All tests in this directory define planned-but-not-yet-implemented features.
# They serve as living specs and are intentionally skipped from the main suite.
collect_ignore_glob = []


def pytest_collection_modifyitems(items):
    for item in items:
        if "aspirational" in str(item.fspath):
            item.add_marker(pytest.mark.skip(reason="feature not yet implemented"))
