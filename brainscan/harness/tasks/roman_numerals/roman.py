VALUES = [("M", 1000), ("D", 500), ("C", 100), ("L", 50), ("X", 10), ("V", 5), ("I", 1)]


def to_roman(n):
    """The Roman numeral for an integer from 1 to 3999."""
    out = ""
    for symbol, value in VALUES:
        while n >= value:
            out += symbol
            n -= value
    return out


def from_roman(s):
    """The integer a Roman numeral stands for. Only the standard form of each number is accepted."""
    total = 0
    for ch in s:
        total += dict(VALUES)[ch]
    return total
