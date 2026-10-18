import json
from pathlib import Path

import pytest

from brainscan.recorder.chat import common_prefix, parse_tool_calls, render, to_template_messages
from brainscan.recorder.events import message_events, scan_tool_output


def test_llama_json_tool_call():
    content, calls = parse_tool_calls('{"name": "read_file", "parameters": {"path": "fizzbuzz.py"}}')
    assert content is None
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "fizzbuzz.py"}


def test_python_tag_and_multiple_calls():
    _, calls = parse_tool_calls('<|python_tag|>{"name": "a", "parameters": {}}; {"name": "b", "parameters": {"x": 1}}')
    assert [c["function"]["name"] for c in calls] == ["a", "b"]


def test_hermes_tagged_calls_keep_surrounding_text():
    content, calls = parse_tool_calls('Let me look.\n<tool_call>\n{"name": "list_files", "arguments": {}}\n</tool_call>')
    assert content == "Let me look."
    assert calls[0]["function"]["name"] == "list_files"


def test_plain_text_and_non_call_json_are_content():
    assert parse_tool_calls("All tests pass now.") == ("All tests pass now.", [])
    assert parse_tool_calls('{"answer": 42}') == ('{"answer": 42}', [])


def test_common_prefix():
    assert common_prefix([1, 2, 3], [1, 2, 4, 5]) == 2
    assert common_prefix([1, 2], [1, 2, 3]) == 2
    assert common_prefix([], [1]) == 0


def test_openai_messages_convert_for_templates():
    msgs = to_template_messages([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1 failed"},
    ])
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == {}
    assert msgs[1]["name"] == "run_tests"


def test_tool_output_events():
    found = dict(scan_tool_output("FAILED tests/test_x.py::test_a - AssertionError\n1 failed, 3 passed in 0.1s"))
    assert found["test_result"] == {"passed": 3, "failed": 1, "errors": 0, "source": "parsed"}
    assert "AssertionError" in found["error"]["excerpt"]
    assert scan_tool_output("Wrote 120 characters to fizzbuzz.py.") == []
    assert [e for e, _ in message_events({"role": "tool", "name": "run_tests", "content": "4 passed"})] == ["tool_result", "test_result"]


TOKENIZER = Path.home() / ".cache" / "brainscan" / "models" / "ollama-llama3.1-8b"


@pytest.mark.skipif(not (TOKENIZER / "tokenizer.json").exists(), reason="converted Llama tokenizer not present")
def test_render_maps_tokens_to_messages():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    messages = [
        {"role": "system", "content": "You are careful."},
        {"role": "user", "content": "Fix the bug in fizzbuzz.py please."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1 failed, 3 passed in 0.02s"},
    ]
    tools = [{"type": "function", "function": {"name": "run_tests", "description": "Run tests",
                                               "parameters": {"type": "object", "properties": {}}}}]
    r = render(tok, messages, tools)
    assert len(r.ids) == len(r.roles) == len(r.msg_index)
    assert all(s >= 0 for s, _ in r.spans), r.spans
    for i, role in enumerate(["system", "user", "assistant", "tool"]):
        first = r.first_token_of(i)
        assert first is not None and r.roles[first] == role
    tool_text = "".join(tok.decode([t]) for t, m in zip(r.ids, r.msg_index) if m == 3)
    assert "1 failed" in tool_text
