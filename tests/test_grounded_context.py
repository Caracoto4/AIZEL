"""Regression tests for grounded self-description and dialogue continuity."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aizel.body.boot import ContextBuilder, Soul
from aizel.core.drives import DriveState
from aizel.core.emotions import Emotions
from aizel.core.personality import Personality


def make_context() -> ContextBuilder:
    return ContextBuilder(Soul(), Personality(), Emotions(), drives=DriveState())


def test_identity_keeps_persona_without_claiming_human_life():
    prompt = make_context().system_prompt()
    assert "локальная AI-система с устойчивой разговорной персоной" in prompt
    assert "Ты не программа и не ассистент" not in prompt
    assert "не ждёшь собеседника" in prompt
    assert "не выдумывая событий" in prompt


def test_dialogue_rule_uses_prior_answers_and_avoids_repetition():
    prompt = make_context().system_prompt()
    assert "не задавай повторно тот же вопрос" in prompt
    assert "прими это и естественно двигайся дальше" in prompt
    assert "Не заканчивай каждый ответ обязательным вопросом" in prompt


def test_greeting_inherits_grounded_identity_rules():
    messages = make_context().build_greeting_messages([])
    system = messages[0]["content"]
    assert "не ждёшь собеседника" in system
    assert "СЕЙЧАС ТЫ ПРОСНУЛАСЬ" in system


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
