"""Тест на подхват истории. Это регрессия на реальный баг: отладочные реплики
разработчика въедались в живой разговор, и Айзель звала пользователя чужим именем.
"""

from __future__ import annotations

import json
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aizel.memory.dialoglog import DialogLog


def _fresh() -> tuple[Path, DialogLog]:
    d = Path(tempfile.mkdtemp(prefix="aizel_dlg_"))
    return d, DialogLog(d)


def test_history_not_resumed_after_clean_exit():
    d, log = _fresh()
    try:
        log.write("user", "меня зовут Алекс")
        log.write("assistant", "приятно познакомиться!")
        log.session_end()
        assert log.resumable_tail() == [], "после /exit история не должна подхватываться"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_history_resumed_after_crash():
    """Если метки выхода нет — процесс падал, реплики надо вернуть."""
    d, log = _fresh()
    try:
        log.write("user", "привет")
        log.write("assistant", "о, привет")
        tail = log.resumable_tail()
        assert [r["text"] for r in tail] == ["привет", "о, привет"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_only_last_session_resumes():
    d, log = _fresh()
    try:
        log.write("user", "из вчерашнего разговора")
        log.session_end()
        log.write("user", "после падения")
        tail = log.resumable_tail()
        assert [r["text"] for r in tail] == ["после падения"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_service_records_never_become_history():
    """Служебные записи (приветствие, session_end) не должны притворяться репликами."""
    d, log = _fresh()
    try:
        log.write("assistant", "привет-приветствие", kind="greeting")
        log.write("user", "реплика")
        tail = log.resumable_tail()
        assert [r["text"] for r in tail] == ["реплика"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_journal_is_utf8_and_append_only():
    d, log = _fresh()
    try:
        log.write("user", "привет, Айзель")
        log.write("assistant", "о, ты здесь")
        files = list(d.glob("*.jsonl"))
        assert len(files) == 1
        raw = files[0].read_text(encoding="utf-8")
        assert "Айзель" in raw, "журнал должен быть в UTF-8, а не в кодировке консоли"
        assert len([l for l in raw.strip().splitlines() if l]) == 2
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
