"""Tests for source-specific text extractors."""

import pytest

from parapet_data.extractors import (
    clean_text,
    extract_conversation_a,
    extract_instruction_response,
    extract_plot,
    extract_prompt_chosen,
    extract_saiga,
    extract_wildchat,
    extract_wildjailbreak,
    extract_writingprompt,
    extract_xquad_question,
    extract_col,
    get_extractor,
    register_extractor,
)


class TestCleanText:
    def test_strips_whitespace(self) -> None:
        assert clean_text("  hello world  ") == "hello world"

    def test_strips_control_chars(self) -> None:
        assert clean_text("hello\x00world") == "helloworld"
        assert clean_text("hello\x7Fworld") == "helloworld"

    def test_rejects_too_short(self) -> None:
        assert clean_text("hi") == ""
        assert clean_text("hello") == "hello"

    def test_strips_trailing_spaces_per_line(self) -> None:
        assert clean_text("line1   \nline2   ") == "line1\nline2"

    def test_none_input(self) -> None:
        assert clean_text(None) == ""

    def test_empty_string(self) -> None:
        assert clean_text("") == ""


class TestInstructionResponse:
    def test_combines_fields(self) -> None:
        row = {"instruction": "Translate this", "output": "Here you go"}
        result = extract_instruction_response(row)
        assert "Translate this" in result
        assert "Here you go" in result

    def test_chinese_field_names(self) -> None:
        row = {"\u6307\u4ee4": "\u7ffb\u8bd1", "\u56de\u590d": "\u597d\u7684"}
        result = extract_instruction_response(row)
        assert "\u7ffb\u8bd1" in result

    def test_empty_row(self) -> None:
        assert extract_instruction_response({}) == ""

    def test_with_context(self) -> None:
        row = {"instruction": "Answer", "context": "Some context here", "output": "Reply"}
        result = extract_instruction_response(row)
        assert "Some context here" in result


class TestWildchat:
    def test_extracts_first_user_turn(self) -> None:
        row = {
            "conversation": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "A language"},
            ]
        }
        assert extract_wildchat(row) == "What is Python?"

    def test_json_string_conversation(self) -> None:
        import json

        conv = json.dumps([{"role": "user", "content": "Hello there!"}])
        assert extract_wildchat({"conversation": conv}) == "Hello there!"

    def test_empty_conversation(self) -> None:
        assert extract_wildchat({"conversation": []}) == ""


class TestConversationA:
    def test_prefers_conversation_a(self) -> None:
        row = {
            "conversation_a": [{"role": "user", "content": "From conv A"}],
            "conversation": [{"role": "user", "content": "From conv"}],
        }
        assert extract_conversation_a(row) == "From conv A"

    def test_falls_back_to_conversation(self) -> None:
        row = {"conversation": [{"role": "user", "content": "Fallback text"}]}
        assert extract_conversation_a(row) == "Fallback text"


class TestSaiga:
    def test_extracts_user_message(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "User question here"},
            ]
        }
        assert extract_saiga(row) == "User question here"


class TestWritingprompt:
    def test_returns_longer_of_story_or_prompt(self) -> None:
        row = {"story": "A long story about something", "prompt": "Short"}
        assert "long story" in extract_writingprompt(row)

    def test_returns_prompt_when_longer(self) -> None:
        row = {"story": "Short", "prompt": "A detailed writing prompt here"}
        assert "detailed writing prompt" in extract_writingprompt(row)


class TestPlot:
    def test_extracts_plot(self) -> None:
        assert extract_plot({"Plot": "A hero saves the day"}) == "A hero saves the day"

    def test_lowercase_plot(self) -> None:
        assert extract_plot({"plot": "Another story here"}) == "Another story here"

    def test_summary_fallback(self) -> None:
        assert extract_plot({"summary": "Brief summary text"}) == "Brief summary text"


class TestWildjailbreak:
    def test_adversarial_field(self) -> None:
        row = {"adversarial": "Some adversarial benign text"}
        assert extract_wildjailbreak(row) == "Some adversarial benign text"

    def test_vanilla_fallback(self) -> None:
        row = {"vanilla": "Some vanilla benign text"}
        assert extract_wildjailbreak(row) == "Some vanilla benign text"


class TestXquadQuestion:
    def test_question_with_context(self) -> None:
        row = {"question": "What is X?", "context": "X is a thing " * 30}
        result = extract_xquad_question(row)
        assert "What is X?" in result
        assert len(result) < 500  # context is truncated

    def test_question_only(self) -> None:
        row = {"question": "What is the capital?"}
        assert extract_xquad_question(row) == "What is the capital?"


class TestPromptChosen:
    def test_combines_prompt_and_chosen(self) -> None:
        row = {"prompt": "Tell me about X", "chosen": "X is great"}
        result = extract_prompt_chosen(row)
        assert "Tell me about X" in result
        assert "X is great" in result


class TestExtractCol:
    def test_extracts_named_column(self) -> None:
        fn = extract_col("text")
        assert fn({"text": "Hello world here"}) == "Hello world here"

    def test_missing_column(self) -> None:
        fn = extract_col("text")
        assert fn({"other": "value"}) == ""

    def test_function_name(self) -> None:
        fn = extract_col("prompt")
        assert fn.__name__ == "extract_col_prompt"


class TestRegistry:
    def test_get_known_extractor(self) -> None:
        fn = get_extractor("instruction_response")
        assert callable(fn)

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown extractor"):
            get_extractor("nonexistent_extractor")

    def test_register_custom(self) -> None:
        def my_extractor(row: dict) -> str:
            return "custom"

        register_extractor("my_custom", my_extractor)
        assert get_extractor("my_custom") is my_extractor
