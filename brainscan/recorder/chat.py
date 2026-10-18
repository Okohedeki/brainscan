"""Render chat requests with the subject's own template and map tokens to messages.

Every token on the tape gets a role (system, user, assistant, tool or template) and
the index of the message it came from, so harness events can be pinned to exact
token positions. Tool calls in the model's output are parsed in both the Llama 3.x
JSON style and the Hermes/Qwen <tool_call> style.
"""

from __future__ import annotations

import bisect
import json
import re
import uuid
from dataclasses import dataclass

import numpy as np

_SPECIAL = re.compile(r"<\|[A-Za-z0-9_]+\|>|<end_of_turn>|<start_of_turn>|</?tool_call>")
_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_PYTHON_TAG = "<|python_tag|>"


@dataclass
class Rendered:
    text: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    roles: list[str]
    msg_index: list[int]
    spans: list[tuple[int, int]]  # per message: char span of its content, (-1, -1) if not found

    def first_token_of(self, msg: int) -> int | None:
        for i, m in enumerate(self.msg_index):
            if m == msg:
                return i
        return None


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") in ("text", "input_text")
    )


def to_template_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI chat messages into the shape Hugging Face chat templates expect."""
    names_by_call: dict[str, str] = {}
    out = []
    for m in messages:
        role = "system" if m.get("role") == "developer" else m.get("role")
        msg: dict = {"role": role, "content": text_of(m.get("content"))}
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                names_by_call[c.get("id")] = fn.get("name")
                calls.append({"id": c.get("id"), "type": "function",
                              "function": {"name": fn.get("name"), "arguments": args}})
            msg["tool_calls"] = calls
        if role == "tool":
            msg["tool_call_id"] = m.get("tool_call_id")
            name = m.get("name") or names_by_call.get(m.get("tool_call_id"))
            if name:
                msg["name"] = name
        out.append(msg)
    return out


def _next_special(text: str, start: int) -> int:
    hit = _SPECIAL.search(text, start)
    return hit.start() if hit else len(text)


def _locate_spans(text: str, messages: list[dict]) -> list[tuple[int, int]]:
    """Find each message's content in the rendered text, in order."""
    spans = []
    cursor = 0
    for m in messages:
        start = end = -1
        needle = (m.get("content") or "").strip()
        if needle:
            start = text.find(needle, cursor)
            if start >= 0:
                end = start + len(needle)
            else:  # the template reformatted the content; anchor on its opening
                start = text.find(needle[:60], cursor)
                if start >= 0:
                    end = _next_special(text, start)
        if m.get("tool_calls"):
            name = m["tool_calls"][0]["function"]["name"] or ""
            hit = text.find(name, end if end >= 0 else cursor) if name else -1
            if hit >= 0:
                if start < 0:
                    opens = [x for x in (text.rfind("{", cursor, hit), text.rfind("<tool_call>", cursor, hit)) if x >= 0]
                    start = max(opens) if opens else hit
                end = _next_special(text, hit)
        if start >= 0:
            cursor = end
        spans.append((start, end))
    return spans


def _assign(offsets, spans, messages):
    order = sorted((s, e, i) for i, (s, e) in enumerate(spans) if s >= 0)
    starts = [s for s, _, _ in order]
    roles, index = [], []
    for a, b in offsets:
        k = bisect.bisect_right(starts, a) - 1
        hit = None
        if k >= 0 and a < order[k][1]:
            hit = order[k][2]
        elif k + 1 < len(order) and b > starts[k + 1]:
            hit = order[k + 1][2]
        if hit is None:
            roles.append("template")
            index.append(-1)
        else:
            roles.append(messages[hit]["role"])
            index.append(hit)
    return roles, index


def one_call_per_turn(tmsgs: list[dict]) -> tuple[list[dict], list[int]]:
    """Split assistant turns with several tool calls into call/result pairs.

    Some templates (Llama 3.1) refuse more than one call per assistant message, but
    models still emit several. Returns the expanded messages and, for each, the index
    of the original message it came from.
    """
    results = {m.get("tool_call_id"): i for i, m in enumerate(tmsgs) if m["role"] == "tool"}
    out, origin, placed = [], [], set()
    for i, m in enumerate(tmsgs):
        calls = m.get("tool_calls") or []
        if m["role"] == "assistant" and len(calls) > 1:
            for k, call in enumerate(calls):
                out.append({**m, "content": m["content"] if k == 0 else "", "tool_calls": [call]})
                origin.append(i)
                r = results.get(call.get("id"))
                if r is not None and r not in placed:
                    out.append(tmsgs[r])
                    origin.append(r)
                    placed.add(r)
        elif i not in placed:
            out.append(m)
            origin.append(i)
    return out, origin


def render(tokenizer, messages: list[dict], tools: list[dict] | None = None,
           add_generation_prompt: bool = True) -> Rendered:
    tmsgs = to_template_messages(messages)
    origin = list(range(len(tmsgs)))
    kwargs = {"tools": tools} if tools else {}
    try:
        text = tokenizer.apply_chat_template(
            tmsgs, add_generation_prompt=add_generation_prompt, tokenize=False, **kwargs)
    except Exception:
        if not any(len(m.get("tool_calls") or []) > 1 for m in tmsgs):
            raise
        expanded, origin = one_call_per_turn(tmsgs)
        text = tokenizer.apply_chat_template(
            expanded, add_generation_prompt=add_generation_prompt, tokenize=False, **kwargs)
        tmsgs = expanded
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    spans = _locate_spans(text, tmsgs)
    roles, index = _assign(offsets, spans, tmsgs)
    # Report message indices and spans in terms of the caller's original messages.
    index = [origin[i] if i >= 0 else -1 for i in index]
    merged = [(-1, -1)] * len(messages)
    for k, (s, e) in enumerate(spans):
        if s < 0:
            continue
        ms, me = merged[origin[k]]
        merged[origin[k]] = (s if ms < 0 else min(ms, s), max(me, e))
    return Rendered(text, list(enc["input_ids"]), offsets, roles, index, merged)


def common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.nonzero(np.asarray(a[:n]) != np.asarray(b[:n]))[0]
    return int(diff[0]) if diff.size else n


def _json_objects(text: str) -> list:
    """Parse one or more JSON objects separated by whitespace, commas or semicolons."""
    decoder = json.JSONDecoder()
    objs, i = [], 0
    while i < len(text):
        while i < len(text) and text[i] in " \t\r\n,;":
            i += 1
        if i >= len(text):
            break
        try:
            obj, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return []
        objs.append(obj)
    return objs


def _as_call(obj) -> tuple[str, object] | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return name, args


def parse_tool_calls(text: str) -> tuple[str | None, list[dict]]:
    """Split model output into (content, OpenAI-style tool_calls)."""
    found = []
    for m in _TAGGED.finditer(text):
        found += [c for c in map(_as_call, _json_objects(m.group(1))) if c]
    content = _TAGGED.sub("", text) if found else text
    if not found:
        body = text.strip()
        if body.startswith(_PYTHON_TAG):
            body = body[len(_PYTHON_TAG):].strip()
        if body.startswith("{"):
            calls = [_as_call(o) for o in _json_objects(body)]
            if calls and all(calls):
                found, content = calls, ""
    tool_calls = [{
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    } for name, args in found]
    return (content.strip() or None), tool_calls
