"""Точка входа Айзель.

    python run.py doctor   — проверка железа, модели и связности
    python run.py chat     — терминальный чат
    python run.py core     — фоновое сознание (Фаза 4)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import platform
import sys

from aizel.body.chat import run_chat
from aizel.core.brain import Brain, BrainUnavailable
from aizel.core.config import load_config


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx пишет каждую строку "HTTP Request: POST ..." — в чате это мусор.
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def doctor(cfg) -> int:
    """Диагностика: чтобы «не работает» превращалось в конкретную строчку."""
    print("─ Айзель: диагностика " + "─" * 30)
    ok = True

    print(f"  Python      : {platform.python_version()}")
    print(f"  ОС          : {platform.system()} {platform.release()}")

    try:
        import httpx  # noqa: F401
        import yaml   # noqa: F401
        print("  зависимости : ok (httpx, PyYAML)")
    except ImportError as exc:
        ok = False
        print(f"  зависимости : НЕТ ({exc}). Выполни: pip install -r requirements.txt")

    try:
        from aizel.body.boot import Soul
        soul = Soul.load(cfg.soul_path)
        print(f"  душа        : ok — «{soul.name}», OCEAN={soul.ocean}")
    except Exception as exc:  # файл души кривой/потерян
        ok = False
        print(f"  душа        : СБОЙ — {exc!r}")

    async with Brain(cfg.brain) as brain:
        try:
            info = await brain.ping()
            print(f"  мозг        : ok — {info['url']} → {', '.join(info['models'])}")
        except BrainUnavailable as exc:
            ok = False
            print(f"  мозг        : {exc}")

    # Память проверяем последней: загрузка BGE-M3 — самая долгая часть.
    try:
        from aizel.core.config import DATA_DIR
        from aizel.memory.facade import Memory

        mem = Memory(DATA_DIR, force_lexical=cfg.memory.force_lexical)
        status = mem.warm_up()
        st = mem.store.stats()
        print(f"  память      : {status}")
        print(f"                эпизодов {st['episodes']}, фактов {st['facts']}, "
              f"реплик {st['messages']}, база {st['size_kb']} КБ")
        if mem.degraded:
            print("                [!] семантика недоступна — поиск только по словам")
        mem.close()
    except Exception as exc:
        ok = False
        print(f"  память      : СБОЙ — {exc!r}")

    print("─" * 47)
    print("  готово к разговору" if ok else "  есть проблемы, см. выше")
    return 0 if ok else 1


def main() -> None:
    # Windows-консоль по умолчанию живёт в OEM-кодировке и рисует кириллицу
    # квадратами/кракозябрами. Перестраховываемся явно — включая stdin: иначе
    # введённые тобой фразы уходят в память битыми байтами.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="aizel", description="Айзель — живой ИИ-компаньон")
    parser.add_argument("command", choices=["doctor", "chat", "core"], nargs="?", default="chat")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg.logging.level)

    if args.command == "doctor":
        sys.exit(asyncio.run(doctor(cfg)))
    if args.command == "core":
        print("Фоновое сознание — Фаза 4 roadmap (см. README). Сейчас доступны doctor и chat.")
        sys.exit(0)
    asyncio.run(run_chat(cfg))


if __name__ == "__main__":
    main()
