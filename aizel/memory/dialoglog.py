"""Сырой журнал диалога: каждая реплика — строкой в data/dialoglog/YYYY-MM-DD.jsonl.

Это ЕЩЁ НЕ память: Айзель по этому журналу ничего не вспоминает.
Зачем нужен сейчас:
  1. История не теряется при /exit (раньше реплики жили только в оперативке).
  2. К Фазе 2 будет накоплен настоящий корпус, который скормим BGE-M3.
  3. Аварийный случай: процесс упал посреди разговора — слова целы.

Пишем append-only и сразу на диск: обрыв процесса не должен стоить реплики.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class DialogLog:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _today(self) -> Path:
        return self.root / f"{datetime.now():%Y-%m-%d}.jsonl"

    def write(self, role: str, content: str, **meta) -> None:
        # Пустой текст допустим только для служебных записей (session_end):
        # сама метка и есть смысл, текст у неё нулевой.
        if not content and not meta.get("kind"):
            return
        record = {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "role": role,
            "text": content,
        }
        record.update(meta)
        with self._today().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    def session_end(self) -> None:
        """Штатный выход. По этой метке видно, что процесс не падал."""
        self.write("meta", "", kind="session_end")

    def resumable_tail(self, limit: int = 12) -> list[dict]:
        """Реплики для подхвата истории — и только если прошлый запуск упал.

        Иначе вчерашний разговор лез бы в каждый новый как «недавняя история»,
        и она приписывала бы себе слова из отладочных прогонов.
        """
        path = self._today()
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not records:
            return []
        # Считаем от последнего штатного выхода: всё до него уже «пережито».
        for i in range(len(records) - 1, -1, -1):
            if records[i].get("kind") == "session_end":
                records = records[i + 1 :]
                break
        return [
            r
            for r in records[-limit:]
            if r.get("role") in ("user", "assistant") and not r.get("kind")
        ]

    def tail(self, limit: int = 10) -> list[dict]:
        """Последние записи за сегодня, включая служебные."""
        path = self._today()
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
