from reliquary.validator.rollout_patterns import detect_opposite_reward_clones


def _text(answer: int, *, correct: bool) -> str:
    verdict = "CORRECT" if correct else "near miss"
    return (
        "We solve the problem by expanding the expression, collecting terms, "
        "checking the arithmetic, and writing the final result. The full "
        "reasoning is shared by a paired rollout, with only the final numeric "
        "answer changed. The same equations, substitutions, and simplification "
        "steps appear in both completions, which is the manufactured pattern "
        "the detector is meant to catch. "
        f"The final numeric answer is {answer}. Final answer: \\boxed{{{answer}}}. "
        f"{verdict}"
    )


def test_detects_three_opposite_reward_clone_pairs():
    texts = [
        _text(40, correct=True),
        _text(40, correct=True),
        _text(40, correct=True),
        "A natural correct solution with a different structure. CORRECT",
        _text(41, correct=False),
        _text(42, correct=False),
        _text(43, correct=False),
        "A natural wrong solution with a different structure.",
    ]
    metrics = detect_opposite_reward_clones(texts, [1, 1, 1, 1, 0, 0, 0, 0])

    assert metrics.suspicious is True
    assert metrics.reward_vector == "11110000"
    assert metrics.matched_pairs == 3


def test_ignores_reward_vector_without_text_clones():
    texts = [
        "Correct path A with its own explanation. CORRECT",
        "Correct path B with another explanation. CORRECT",
        "Correct path C with a symbolic explanation. CORRECT",
        "Correct path D with a short explanation. CORRECT",
        "Wrong path A makes a sign error.",
        "Wrong path B uses the wrong formula.",
        "Wrong path C estimates the value.",
        "Wrong path D stops too early.",
    ]
    metrics = detect_opposite_reward_clones(texts, [1, 1, 1, 1, 0, 0, 0, 0])

    assert metrics.suspicious is False
    assert metrics.matched_pairs == 0
