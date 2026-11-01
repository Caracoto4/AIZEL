"""Тесты локальной оценки эмоциональных сигналов."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aizel.core.appraisal import appraise_user_message


def test_neutral_message_does_not_change_mood():
    assert appraise_user_message("Как устроена память в Python?") is None


def test_appreciation_creates_positive_reaction():
    reaction = appraise_user_message("Спасибо, ты очень помогла!")
    assert reaction is not None
    assert reaction.pleasure > 0
    assert reaction.arousal > 0
    assert reaction.cause == "благодарность или похвала"


def test_direct_hostility_creates_stronger_negative_reaction():
    reaction = appraise_user_message("Ты бесполезная, отстань")
    assert reaction is not None
    assert reaction.pleasure < -0.05
    assert reaction.arousal > 0
    assert reaction.cause == "прямая враждебность"


def test_frustration_with_answer_is_milder_than_hostility():
    frustration = appraise_user_message("Нет, это опять не то")
    hostility = appraise_user_message("Ты тупая")
    assert frustration is not None and hostility is not None
    assert abs(frustration.pleasure) < abs(hostility.pleasure)


def test_negative_topic_is_not_treated_as_personal_attack():
    assert appraise_user_message("Я ненавижу плохую погоду") is None


def test_touchiness_scales_only_negative_reactions():
    calm = appraise_user_message("Ты тупая", touchiness=0.1)
    sensitive = appraise_user_message("Ты тупая", touchiness=0.9)
    praise_calm = appraise_user_message("Спасибо", touchiness=0.1)
    praise_sensitive = appraise_user_message("Спасибо", touchiness=0.9)
    assert calm is not None and sensitive is not None
    assert praise_calm is not None and praise_sensitive is not None
    assert sensitive.sensitivity > calm.sensitivity
    assert praise_calm.sensitivity == praise_sensitive.sensitivity == 1.0


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failed += 1
                print(f"  FAIL {name}: {exc}")
    print("\n", "ВСЁ ЗЕЛЁНОЕ" if not failed else f"провалено: {failed}")
    sys.exit(1 if failed else 0)
