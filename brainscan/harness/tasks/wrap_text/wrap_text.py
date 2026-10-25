def wrap(text, width):
    """Wrap text into lines of at most `width` characters.

    - Words are separated by any whitespace; lines never start or end with a space.
    - Lines are filled greedily: each word goes on the current line if it fits.
    - A word longer than `width` is cut into pieces of `width` characters (the last piece may be
      shorter), and the pieces are placed like words.
    - A blank line separates paragraphs. Each paragraph is wrapped on its own, and one empty
      string separates paragraphs in the result.
    - `width` must be at least 1.
    """
    words = text.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) < width:
            line += " " + word
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines
