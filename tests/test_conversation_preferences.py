"""Tests for explicit, persistent conversational preferences."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aizel.core.preferences import ConversationPreferences
from aizel.body.boot import ContextBuilder, Soul
from aizel.core.drives import DriveState
from aizel.core.emotions import Emotions
from aizel.core.personality import Personality


def test_explicit_greeting_feedback_is_saved_and_loaded():
    prefs = ConversationPreferences()
    assert prefs.observe_feedback(
        "Не начинай каждый ответ с приветствия, используй приветствия пореже"
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "conversation_preferences.json"
        prefs.save(path)
        loaded = ConversationPreferences.load(path)
    assert loaded.avoid_repeated_greetings
    assert "не начинай обычные ответы с приветствия" in loaded.directive()


def test_question_about_greetings_does_not_create_preference():
    prefs = ConversationPreferences()
    assert not prefs.observe_feedback("Почему ты постоянно здороваешься со мной?")
    assert not prefs.avoid_repeated_greetings


def test_preference_allows_real_new_session_greeting():
    directive = ConversationPreferences(avoid_repeated_greetings=True).directive()
    assert "начале новой сессии допустимо" in directive
    assert "не запрещает его полностью" in directive


def test_preference_is_in_system_prompt():
    prefs = ConversationPreferences(avoid_repeated_greetings=True)
    ctx = ContextBuilder(Soul(), Personality(), Emotions(), drives=DriveState(), preferences=prefs)
    assert "УСТОЙЧИВОЕ ПРЕДПОЧТЕНИЕ СОБЕСЕДНИКА" in ctx.system_prompt()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
    print("\n", "ВСЁ ЗЕЛЁНОЕ" if not failures else f"провалено: {failures}")
    sys.exit(1 if failures else 0)
