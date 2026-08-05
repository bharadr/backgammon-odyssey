from types import SimpleNamespace as Block

from coach.llm import _text_of

# The SDK call itself can't be tested offline, but the text extraction is pure:
# fake a message whose `.content` is a list of typed blocks.


def test_text_of_joins_text_blocks_and_skips_non_text():
    message = Block(content=[
        Block(type="text", text="Make the "),
        Block(type="tool_use", id="x"),        # no `.text` -- must be skipped, not read
        Block(type="text", text="5-point."),
    ])
    assert _text_of(message) == "Make the 5-point."


def test_text_of_is_empty_when_there_are_no_text_blocks():
    message = Block(content=[Block(type="tool_use", id="x")])
    assert _text_of(message) == ""
