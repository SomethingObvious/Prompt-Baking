import pytest

import test_instruct_model as instruct
from test_math_model import assess_correct, reformat_answer_string


def test_svamp_answers_are_stored_as_floats():
    assert reformat_answer_string(51.0, "svamp") == "51"
    assert reformat_answer_string(2.5, "svamp") == "2.5"


def test_answers_lose_their_thousands_separators():
    assert reformat_answer_string("Some working.\n#### 1,250", "gsm8k") == "1250"
    assert reformat_answer_string("1,200 (dollars)", "asdiv") == "1200"


def test_commas_in_numbers_are_ignored():
    exact, anywhere, last = assess_correct(
        ["So it is 1,250. The answer is 1,250."], ["#### 1250"], "gsm8k"
    )
    assert (exact, anywhere, last) == ([True], [True], [True])


def test_a_reply_without_a_period_is_one_sentence():
    exact, anywhere, last = assess_correct(["the answer is 7 apples"], ["7 (apples)"], "asdiv")
    assert (exact, anywhere, last) == ([False], [True], [True])


def test_last_sentence_ends_at_the_last_period():
    _, _, last = assess_correct(["It is 7. Or maybe 8. Then more"], ["8"], "svamp")
    assert last == [True]
    _, _, last = assess_correct(["It is 7. Or maybe 9. Then 8"], ["8"], "svamp")
    assert last == [False]


def test_french_score_of_no_letters_is_zero():
    assert instruct.french_score("q", "") == 0.0
    assert instruct.french_score("q", "12345 !!!") == 0.0


def test_french_score_is_high_for_french():
    reply = "Je pense que le ciel est bleu parce que la lumière du soleil se disperse dans l'air."
    assert instruct.french_score("q", reply) > 0.9


def test_no_e_score_counts_capital_e_too():
    assert instruct.no_e_score("q", "Eat") == 0.5
    assert instruct.no_e_score("q", "good") == 1.0


def test_rare_lexicon_score_handles_an_empty_reply(monkeypatch):
    monkeypatch.setattr(instruct, "common_words", lambda: frozenset({"the", "cat"}))
    assert instruct.rare_lexicon_score("q", "...") == 0.0
    assert instruct.rare_lexicon_score("q", "The cat perambulates, sesquipedalian!") == 0.5


def test_reversed_score_is_one_for_a_perfect_reversal():
    assert instruct.reversed_score("How are you doing?", "?doing you are How") == pytest.approx(1.0)


def test_capital_scores():
    assert instruct.second_capital_score("q", "THE cat SAT on") == 1.0
    assert instruct.second_capital_score("q", "the cat sat on") == 0.5
    # Words 1, 2, 3 and 5 count as prime, so 4 and 6 have to be lowercase.
    assert instruct.prime_capital_score("q", "ONE TWO THREE four FIVE six") == 1.0
    assert instruct.prime_capital_score("q", "one two three four five six") == 0.5


def test_blue_score_wants_exactly_one_blue_per_sentence():
    assert instruct.blue_score(
        "q", "The sky is blue. The sea is blue blue. Grass is green"
    ) == pytest.approx(1 / 3)


def test_scorer_is_picked_by_the_prompt_file_name():
    assert instruct.scorer_for("data/InstructionX0/always_french_x0.md") is instruct.french_score
    with pytest.raises(ValueError, match="no scorer"):
        instruct.scorer_for("data/InstructionX0/secret_number_x0.md")
