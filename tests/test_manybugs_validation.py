"""Non-network regression tests for manybugs.py's scenario-name validation.

Kept in a separate module from test_manybugs.py, which is module-wide
`pytest.mark.network` (its tests fetch real tarballs) -- these tests must
run by default since they check that unsafe input is rejected *before* any
network call or filesystem write happens.
"""

from pathlib import Path

import pytest

from badcode_ft.data.manybugs import _download_scenario


@pytest.mark.parametrize(
    "scenario",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "lighttpd-bug-2785-2786/../../evil",
        "lighttpd/bug",
        "",
    ],
)
def test_download_scenario_rejects_unsafe_scenario_names(tmp_path, scenario):
    with pytest.raises(ValueError):
        _download_scenario(scenario, tmp_path)

    # No cache directory contents should have been created for a rejected name.
    assert list(tmp_path.iterdir()) == []


def test_download_scenario_accepts_realistic_scenario_names():
    from badcode_ft.data.manybugs import _SCENARIO_NAME_RE

    assert _SCENARIO_NAME_RE.match("lighttpd-bug-2785-2786")
    assert _SCENARIO_NAME_RE.match("gzip-bug-2004-07-27-c1e2c39-fdd4784")
