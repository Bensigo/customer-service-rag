from app.ingestion.chunker import chunk_text, estimate_tokens, split_sentences
from app.models import ChunkDraft


class TestSplitSentences:
    def test_basic_punctuation_boundaries(self):
        text = "First sentence. Second sentence? Third one!"

        assert split_sentences(text) == [
            "First sentence.",
            "Second sentence?",
            "Third one!",
        ]

    def test_abbreviations_do_not_split(self):
        text = "Dr. Smith arrived at 9 a.m. sharp. He was early."

        assert split_sentences(text) == [
            "Dr. Smith arrived at 9 a.m. sharp.",
            "He was early.",
        ]

    def test_eg_and_ie_do_not_split(self):
        text = "Use fruit, e.g. apples or pears. They keep well."

        assert split_sentences(text) == [
            "Use fruit, e.g. apples or pears.",
            "They keep well.",
        ]

    def test_markdown_newlines_are_boundaries(self):
        text = "# Refund policy\nRefunds take 5 days. Contact support.\nNo receipt needed"

        assert split_sentences(text) == [
            "# Refund policy",
            "Refunds take 5 days.",
            "Contact support.",
            "No receipt needed",
        ]

    def test_empty_and_whitespace_return_nothing(self):
        assert split_sentences("") == []
        assert split_sentences("   \n\n  \t ") == []


class TestEstimateTokens:
    def test_scales_word_count_conservatively(self):
        # heuristic: ceil(words * 1.4) — documented approximation of the
        # bge-small tokenizer so chunk budgets stay under its 512 window
        assert estimate_tokens("one two three four five") == 7

    def test_empty_text_is_zero(self):
        assert estimate_tokens("") == 0


class TestChunkText:
    def make_sentences(self, n: int) -> str:
        # each sentence = 6 words -> estimate_tokens == ceil(6 * 1.4) == 9
        return " ".join(f"This is test sentence number {i}." for i in range(n))

    def test_single_short_text_is_one_chunk(self):
        drafts = chunk_text("Short text here.", max_tokens=100)

        assert len(drafts) == 1
        assert drafts[0].seq == 0
        assert drafts[0].text == "Short text here."

    def test_returns_chunk_drafts(self):
        drafts = chunk_text(self.make_sentences(4), max_tokens=100)

        assert all(isinstance(d, ChunkDraft) for d in drafts)

    def test_respects_token_budget(self):
        drafts = chunk_text(self.make_sentences(12), max_tokens=21)

        assert len(drafts) > 1
        assert all(d.token_estimate <= 21 for d in drafts)

    def test_consecutive_chunks_share_overlap_sentences(self):
        # sentences are 6 words -> 9 tokens; 27-token budget = 3 per window,
        # leaving room for the full 2-sentence overlap
        drafts = chunk_text(self.make_sentences(12), max_tokens=27, overlap_sentences=2)

        assert len(drafts) > 1
        for prev, cur in zip(drafts, drafts[1:], strict=False):
            prev_sentences = split_sentences(prev.text)
            overlap = " ".join(prev_sentences[-2:])
            assert cur.text.startswith(overlap)

    def test_overlap_degrades_when_window_is_too_small_to_repeat(self):
        # 21-token budget fits only 2 sentences per window; a full
        # 2-sentence overlap would repeat the same window forever, so the
        # chunker must shrink the overlap and still advance
        drafts = chunk_text(self.make_sentences(6), max_tokens=21, overlap_sentences=2)

        texts = [d.text for d in drafts]
        assert len(texts) == len(set(texts))  # no repeated windows
        for prev, cur in zip(drafts, drafts[1:], strict=False):
            prev_last = split_sentences(prev.text)[-1]
            assert cur.text.startswith(prev_last)  # still overlaps by one

    def test_seq_is_contiguous_from_zero(self):
        drafts = chunk_text(self.make_sentences(12), max_tokens=21)

        assert [d.seq for d in drafts] == list(range(len(drafts)))

    def test_oversized_single_sentence_becomes_own_chunk(self):
        long_sentence = "word " * 50 + "end."
        text = f"Tiny one. {long_sentence} Tiny two."

        drafts = chunk_text(text, max_tokens=20)

        assert any(d.text == long_sentence.strip() for d in drafts)

    def test_empty_and_whitespace_text_return_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("  \n \t ") == []

    def test_no_sentence_is_lost(self):
        text = self.make_sentences(25)
        sentences = split_sentences(text)

        drafts = chunk_text(text, max_tokens=21, overlap_sentences=2)

        for sentence in sentences:
            assert any(sentence in d.text for d in drafts)
