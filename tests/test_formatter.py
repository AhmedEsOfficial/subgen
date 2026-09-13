from __future__ import annotations

from subgen.config import SubtitleConfig
from subgen.models import TimedSegment, TimedWord
from subgen.subtitle.formatter import format_subtitles


def _words(text: str, start: float) -> list[TimedWord]:
    return [TimedWord(start + i, start + i + 0.9, w) for i, w in enumerate(text.split())]


def test_translated_segment_is_not_rebuilt_from_source_words():
    # Long enough (> max_duration) to be split; words are still the Arabic ASR words.
    arabic = "هو يعمل او في التكلفة فعايز افهم منك"
    seg = TimedSegment(40.9, 50.9, "What he does, or in the cost. So I want to understand from you.",
                       words=_words(arabic, 40.9))
    text = " ".join(e.text.replace("\n", " ") for e in format_subtitles([seg], SubtitleConfig()))
    assert text == seg.text


def test_long_text_is_split_not_truncated():
    cfg = SubtitleConfig()
    sentence = ("For this thing, so we keep it in mind, but even when we're pricing "
                "for the client, and I get you what you're building it on, a little bit.")
    entries = format_subtitles([TimedSegment(0.0, 6.0, sentence)], cfg)

    assert len(entries) > 1
    assert " ".join(e.text.replace("\n", " ") for e in entries) == sentence
    for e in entries:
        lines = e.text.split("\n")
        assert len(lines) <= cfg.max_lines
        assert all(len(line) <= cfg.max_chars_per_line for line in lines)
        assert "..." not in e.text
    assert entries[0].start == 0.0 and entries[-1].end == 6.0
    assert all(a.end == b.start for a, b in zip(entries, entries[1:]))


def test_split_by_text_respects_max_duration():
    cfg = SubtitleConfig()
    sentence = ("Things still aren't getting edited. When I look for something to edit, I can't find "
                "the content. And another thing, say it's showing twenty-five columns and I want more.")
    entries = format_subtitles([TimedSegment(0.24, 23.22, sentence)], cfg)
    assert " ".join(e.text.replace("\n", " ") for e in entries) == sentence
    assert all(e.end - e.start <= cfg.max_duration + 1.0 for e in entries)  # char-share timing
    assert all(len(e.text.split("\n")) <= cfg.max_lines for e in entries)


def test_split_by_words_uses_source_words_when_they_match():
    cfg = SubtitleConfig()
    text = " ".join(["كلمة"] * 40)
    entries = format_subtitles([TimedSegment(0.0, 40.0, text, words=_words(text, 0.0))], cfg)
    assert len(entries) > 1
    assert " ".join(e.text.replace("\n", " ") for e in entries) == text
    assert all(len(e.text.split("\n")) <= cfg.max_lines for e in entries)


def test_split_by_words_respects_max_duration():
    cfg = SubtitleConfig()
    text = "لسه ما فيش حاجات تتعدل لما بدور على حاجة اعدلها ما بلقيش"  # short text, 11 words over 11s
    entries = format_subtitles([TimedSegment(0.0, 11.0, text, words=_words(text, 0.0))], cfg)
    assert len(entries) > 1
    assert all(e.end - e.start <= cfg.max_duration for e in entries)
    assert " ".join(e.text for e in entries) == text


def test_short_segment_unchanged():
    entries = format_subtitles([TimedSegment(1.0, 3.5, "Test subtitle")], SubtitleConfig())
    assert [(e.index, e.start, e.end, e.text) for e in entries] == [(1, 1.0, 3.5, "Test subtitle")]
