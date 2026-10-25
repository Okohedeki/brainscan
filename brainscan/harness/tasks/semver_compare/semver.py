def parse(version):
    """Split a version like "1.4.2-rc.1" into (major, minor, patch, prerelease)."""
    core, _, pre = version.partition("-")
    major, minor, patch = core.split(".")
    return int(major), int(minor), int(patch), pre


def compare(a, b):
    """-1 if version a sorts before b, 0 if they have the same precedence, 1 if a sorts after b."""
    pa, pb = parse(a), parse(b)
    return (pa > pb) - (pa < pb)
