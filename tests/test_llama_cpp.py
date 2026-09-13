from __future__ import annotations

from subgen.config import TranslationConfig
from subgen.translation.llama_cpp import LlamaCppTranslator


def test_parse_keys_by_number_not_position():
    raw = "Here are the translations:\n1. One\n3. Three\n2) Two\n2. duplicate\n9. out of range"
    assert LlamaCppTranslator._parse_numbered_output(raw, 3) == {0: "One", 1: "Two", 2: "Three"}


def test_parse_handles_markdown_numbering():
    assert LlamaCppTranslator._parse_numbered_output("**1.** Hello\n**2.** World", 2) == {0: "Hello", 1: "World"}


def test_batch_retranslates_only_missing_and_arabic_lines(monkeypatch):
    tr = LlamaCppTranslator(TranslationConfig(batch_size=10))
    monkeypatch.setattr(tr, "_ensure_server", lambda: None)
    # Model skipped line 2 and echoed Arabic for line 3.
    monkeypatch.setattr(tr, "_chat_completion", lambda messages, max_tokens=0: "1. first\n3. تالت\n4. fourth")
    single_calls = []
    monkeypatch.setattr(tr, "translate", lambda text: single_calls.append(text) or f"english #{len(single_calls)}")

    out = tr.translate_batch(["اول", "تاني", "تالت", "رابع"])

    assert out == ["first", "english #1", "english #2", "fourth"]
    assert single_calls == ["تاني", "تالت"]


def test_context_translation_one_output_per_input(monkeypatch):
    tr = LlamaCppTranslator(TranslationConfig(source_lang="arz_Arab", target_lang="eng_Latn"))
    monkeypatch.setattr(tr, "_ensure_server", lambda: None)
    replies = iter(["<think>reasoning</think>\nline one", "Translation:\n\"line two\"", "تاني", "line three"])
    monkeypatch.setattr(tr, "_chat_completion", lambda messages, max_tokens=0: next(replies))

    assert tr.translate_with_context(["واحد", "اتنين", "تلاتة"], domain="a voice note") == [
        "line one", "line two", "line three",
    ]


def test_last_answer_line():
    assert LlamaCppTranslator._last_answer_line('Sure.\n\n2. "Hello there"') == "Hello there"
    assert LlamaCppTranslator._last_answer_line("   ") == ""
