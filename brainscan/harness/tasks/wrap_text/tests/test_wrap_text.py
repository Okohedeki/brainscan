import pytest

from wrap_text import wrap


def test_fills_lines_greedily():
    assert wrap("the quick brown fox jumps over the lazy dog", 10) == [
        "the quick", "brown fox", "jumps over", "the lazy", "dog"]


def test_a_line_can_be_exactly_width():
    assert wrap("abc def", 7) == ["abc def"]
    assert wrap("abc def", 6) == ["abc", "def"]


def test_whitespace_is_collapsed():
    assert wrap("  one\ttwo   three \n", 20) == ["one two three"]


def test_long_words_are_cut():
    assert wrap("supercalifragilistic is long", 8) == ["supercal", "ifragili", "stic is", "long"]
    assert wrap("a supercalifragilistic", 8) == ["a", "supercal", "ifragili", "stic"]


def test_paragraphs_are_kept():
    text = "first paragraph here\n\nsecond one\n   \nthird"
    assert wrap(text, 10) == ["first", "paragraph", "here", "", "second one", "", "third"]


def test_empty_text():
    assert wrap("", 10) == []
    assert wrap("   \n\n  ", 10) == []


def test_width_must_be_positive():
    with pytest.raises(ValueError):
        wrap("abc", 0)
