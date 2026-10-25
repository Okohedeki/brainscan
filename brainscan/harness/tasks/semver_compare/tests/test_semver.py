import pytest

from semver import compare


def test_core_numbers_compare_as_numbers():
    assert compare("1.10.0", "1.9.0") == 1
    assert compare("2.0.0", "10.0.0") == -1
    assert compare("1.2.3", "1.2.3") == 0


def test_release_sorts_after_its_prereleases():
    assert compare("1.0.0", "1.0.0-rc.1") == 1
    assert compare("1.0.0-alpha", "1.0.0") == -1


def test_precedence_chain():
    chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2",
             "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
    for lo, hi in zip(chain, chain[1:]):
        assert compare(lo, hi) == -1, (lo, hi)
        assert compare(hi, lo) == 1, (hi, lo)


def test_build_metadata_is_ignored():
    assert compare("1.0.0+build.1", "1.0.0+build.2") == 0
    assert compare("1.0.0-rc.1+exp.sha.5114f85", "1.0.0-rc.1") == 0


@pytest.mark.parametrize("bad", ["1.2", "1.2.3.4", "01.2.3", "1.2.3-", "1.2.3-alpha..1", "1.2.3-01", "v1.2.3", ""])
def test_invalid_versions_raise(bad):
    with pytest.raises(ValueError):
        compare(bad, "1.0.0")
