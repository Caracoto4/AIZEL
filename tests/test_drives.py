"""Тесты детерминированной модели побуждений."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aizel.core.drives import DriveState
from aizel.core.personality import Personality


def test_values_are_clamped():
    drives = DriveState(values={"curiosity": 4, "engagement": -2})
    assert drives.values["curiosity"] == 1.0
    assert drives.values["engagement"] == 0.0


def test_neutral_message_decays_toward_baseline():
    drives = DriveState(values={"curiosity": 0.9})
    drives.observe("Привет")
    assert drives.values["curiosity"] < 0.9
    assert drives.values["curiosity"] > 0.22


def test_question_increases_curiosity():
    drives = DriveState()
    before = drives.values["curiosity"]
    drives.observe("Почему память работает именно так?")
    assert drives.values["curiosity"] > before


def test_follow_up_increases_continuity_only_with_history():
    without_history = DriveState()
    with_history = DriveState()
    without_history.observe("А если сделать иначе?", has_history=False)
    with_history.observe("А если сделать иначе?", has_history=True)
    assert with_history.values["continuity"] > without_history.values["continuity"]


def test_correction_increases_reflection():
    drives = DriveState()
    before = drives.values["reflection"]
    drives.observe("Ты не поняла, я имел в виду другое")
    assert drives.values["reflection"] > before


def test_weak_signals_do_not_force_initiative():
    drives = DriveState()
    assert drives.choose_intention(Personality()) == "respond"


def test_explicit_continuation_can_select_continue():
    drives = DriveState()
    drives.observe("Продолжи, расскажи подробнее об этой теме")
    assert drives.choose_intention(Personality()) in {"continue", "clarify", "suggest"}


def test_state_round_trips():
    drives = DriveState()
    drives.observe("Почему это работает?")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "drives.json"
        drives.save(path)
        loaded = DriveState.load(path)
    assert loaded.values == drives.values
    assert loaded.last_intention == drives.last_intention


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
