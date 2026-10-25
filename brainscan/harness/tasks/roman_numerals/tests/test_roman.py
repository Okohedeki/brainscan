import pytest

from roman import from_roman, to_roman

KNOWN = {1: "I", 3: "III", 4: "IV", 9: "IX", 14: "XIV", 40: "XL", 90: "XC", 400: "CD", 1994: "MCMXCIV",
         2024: "MMXXIV", 3999: "MMMCMXCIX"}


def test_to_roman():
    for n, s in KNOWN.items():
        assert to_roman(n) == s, n


def test_from_roman():
    for n, s in KNOWN.items():
        assert from_roman(s) == n, s


def test_round_trip():
    for n in range(1, 4000):
        assert from_roman(to_roman(n)) == n


@pytest.mark.parametrize("n", [0, -5, 4000])
def test_to_roman_range(n):
    with pytest.raises(ValueError):
        to_roman(n)


@pytest.mark.parametrize("bad", ["", "IIII", "VV", "IC", "IL", "XM", "VX", "MMMM", "IIV", "iv", "ABC", "XCX"])
def test_from_roman_rejects_non_standard_numerals(bad):
    with pytest.raises(ValueError):
        from_roman(bad)
