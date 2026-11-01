"""Терминальный чат с Айзель.

Три слоя переживают закрытие терминала:
  состояние  — настроение и черты (data/*.json), завтрашняя будет другой по тону;
  журнал     — дословные реплики (data/dialoglog/), на случай падения;
  память     — эпизоды и факты с векторами (data/aizel.db), то, что она помнит сама.

Перед ответом память спрашивается («что всплывает в голове по этому поводу»),
после обмена — записывается. Ядро (имя, город, возраст) обновляется МГНОВЕННО,
на той же реплике, регуляркой и без обращения к модели: очередь к KoboldCpp
одна, и фоновый прогон извлечения встал бы впереди следующего ответа — при
7 ток/с это минута зависшего терминала. Спокойный смысловой прогон остаётся
только на выходе, когда ждать уже нечего.
"""

from __future__ import annotations

import logging
import sys
import time

from ..body.boot import ContextBuilder, Soul
from ..core.brain import Brain, BrainUnavailable
from ..core.config import DATA_DIR, AizelConfig
from ..core.emotions import Emotions
from ..core.personality import Personality
from ..memory.dialoglog import DialogLog
from ..memory.facade import Memory
from ..memory.semantic import extract_facts

log = logging.getLogger("aizel.chat")

PERSONALITY_PATH = DATA_DIR / "personality.json"
EMOTIONS_PATH = DATA_DIR / "emotions.json"
DIALOG_DIR = DATA_DIR / "dialoglog"

CMD_EXIT = ("/exit", "/quit", "/q")
CMD_DUMP = ("/state",)
CMD_MEM = ("/mem", "/memory")
CMD_FORGET = ("/forget",)

# Размер одного запроса извлечения: 7 обменов (или меньше).
EXTRACT_WINDOW = 14


async def run_chat(cfg: AizelConfig) -> None:
    soul = Soul.load(cfg.soul_path)
    # Состояние догружаем из data/, если оно уже накоплено.
    personality = (
        Personality.load(PERSONALITY_PATH)
        if PERSONALITY_PATH.exists()
        else soul.to_personality()
    )
    emotions = Emotions.load(EMOTIONS_PATH)

    ctx = ContextBuilder(soul, personality, emotions)
    journal = DialogLog(DIALOG_DIR)
    memory = Memory(DATA_DIR, force_lexical=cfg.memory.force_lexical)

    # Подхватываем историю ТОЛЬКО если прошлый запуск оборвался. Иначе вчерашние
    # (или отладочные) реплики въедались бы в каждый новый разговор, и она
    # начинала приписывать собеседнику чужие слова.
    history: list[dict] = [
        {"role": r["role"], "content": r["text"]} for r in journal.resumable_tail(12)
    ]
    resumed = len(history) // 2

    print(f"\n  {soul.name} просыпается...", flush=True)

    async with Brain(cfg.brain) as brain:
        try:
            info = await brain.ping()
            print(f"  мозг на связи: {', '.join(info['models']) or 'локальная модель'}")
        except BrainUnavailable as exc:
            print(f"\n  [!] {exc}", file=sys.stderr)
            print("      запусти KoboldCpp с включённым API и повтори.\n", file=sys.stderr)
            memory.close()
            return

        print(f"  память: {memory.warm_up()}", flush=True)
        stats = memory.store.stats()
        if stats["episodes"] or stats["facts"]:
            print(f"  помнит: эпизодов {stats['episodes']}, фактов {stats['facts']}")
        if resumed:
            print(f"  продолжает сегодняшний разговор ({resumed} реплик)")

        # Ядро памяти всегда в контексте: имя, город, работа. Это не поиск —
        # то, что обязано вспомниться даже без похожих слов в запросе.
        ctx.core_memory = memory.store.all_core()

        # Приветствие она генерирует сама, из своего состояния — не захардкоженное.
        greet_msgs = ctx.build_greeting_messages(history)
        try:
            first = await brain.chat(greet_msgs, max_tokens=160)
            greeting = first.text or "…"
        except BrainUnavailable:
            greeting = "…что-то с головой. ты здесь?"
        journal.write("assistant", greeting, kind="greeting")
        print(f"\n  {soul.name}: {greeting}\n")

        # Восстанавливаем ВСЮ незавершённую очередь из append-only журнала.
        # История для генерации ограничена 12 репликами, но усвоение не должно
        # терять более ранние реплики после аварийного завершения.
        pending = journal.resumable_all()
        turn_log: list[str] = [
            f"{'СОБЕСЕДНИК' if row['role'] == 'user' else 'АЙЗЕЛЬ'}: {row['text']}"
            for row in pending
        ]

        def refresh_core() -> None:
            """Догружает ядро в контекст — сразу, чтобы имя стало известно этому же ответу."""
            ctx.core_memory = memory.store.all_core()

        while True:
            try:
                user_text = input("  ты › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_text:
                continue
            if user_text.lower() in CMD_EXIT:
                break

            if user_text.lower() in CMD_DUMP:
                st = memory.store.stats()
                print(f"\n  {personality.snapshot()}")
                print(f"  настроение: {emotions.describe()} (интенсивность {emotions.intensity:.2f})")
                print(f"  память: эпизодов {st['episodes']}, фактов {st['facts']}, "
                      f"реплик {st['messages']}, база {st['size_kb']} КБ")
                print(f"  режим: {'лексический (без BGE-M3)' if memory.degraded else 'семантический BGE-M3'}")
                print(f"  системный блок: ~{len(ctx.system_prompt())} симв.\n")
                continue

            if user_text.lower() in CMD_MEM:
                st = memory.store.stats()
                if not st["episodes"] and not st["facts"]:
                    print("\n  пока пустовато: она ещё ничего не отложила.\n")
                else:
                    print(f"\n  последние воспоминания:\n{memory.store.dump_debug(8)}\n")
                core = memory.store.all_core()
                if core:
                    print("  ядро (то, что всегда с собой):")
                    for c in core:
                        print(f"    - {c}")
                    print()
                continue

            if user_text.lower() in CMD_FORGET:
                backup = memory.forget_all(DIALOG_DIR)
                # Не даём текущей истории и очереди усвоения восстановить
                # только что удалённые сведения при следующем ответе или /exit.
                history.clear()
                turn_log.clear()
                journal.session_end()
                ctx.core_memory = []
                ctx.memories = []
                print(f"\n  стёрла всё. копия на всякий случай: {backup}")
                print("      если передумаешь — верни файлы из этой папки.\n")
                continue

            # Ядро выхватываем ДО генерации: сказав «меня зовут Даниил», ты
            # должен получить «приятно познакомиться, Даниил» в этом же ответе,
            # а не в следующем разговоре. Регулярка, без прогона через модель.
            if memory.quick_core(user_text):
                refresh_core()

            # --- что всплывает в голове по этому поводу --------------------
            # Спрашиваем память не только новой репликой: на «привет» или «ну
            # как ты» по смыслу всплывает всё, что обсуждалось минут пять назад.
            mood_p = emotions.state.pleasure
            probe = memory.probe_text(history, user_text)
            ctx.memories = memory.context_lines(probe, mood=mood_p, top_k=cfg.memory.top_k)
            if ctx.memories:
                log.info("вспомнила %d шт: %s", len(ctx.memories), ctx.memories[0][:60])

            # Реплика пишется ДО генерации: если процесс упадет или модель
            # зависнет, сказанное тобой не пропадет.
            journal.write("user", user_text)
            memory.remember_message("user", user_text)

            messages = ctx.build_messages(history, user_text)
            started = time.perf_counter()
            print(f"\n  {soul.name}: ", end="", flush=True)
            collected: list[str] = []
            try:
                async for piece in brain.stream_chat(messages):
                    collected.append(piece)
                    print(piece, end="", flush=True)
            except BrainUnavailable as exc:
                if not collected:
                    print(f"[молчание: {exc}]", end="")
            elapsed = time.perf_counter() - started
            answer = "".join(collected).strip()
            print(f"\n  \033[2m({len(answer.split())} слов за {elapsed:.1f} с)\033[0m\n")

            if not answer:
                continue

            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": answer})
            journal.write("assistant", answer, seconds=round(elapsed, 1))
            memory.remember_message("assistant", answer)

            # Эпизод — смысл обмена. Ответ не храним отдельно: он порождение
            # её состояния, а не событие в твоей жизни.
            memory.remember_turn(
                user_text, answer, mood=mood_p, topic=user_text[:60]
            )
            turn_log.append(f"СОБЕСЕДНИК: {user_text}\nАЙЗЕЛЬ: {answer}")

            # Простейший эмоциональный отклик (Фаза 3 заменит на анализ).
            # Пока: длинный живой ответ поднимает настроение, молчание — роняет.
            emotions.react(
                pleasure=0.04 if len(answer) > 40 else 0.0,
                arousal=0.03,
                cause="разговор",
                topic=user_text[:80],
                sensitivity=personality.touchiness + 0.5,
            )

        # --- засыпаем -------------------------------------------------------
        added = {"core": 0, "facts": 0}
        ingestion_ok = True
        if turn_log and cfg.memory.extract_on_exit:
            print("  записывает, что узнала о тебе...", flush=True)
            # Обрабатываем ВЕСЬ диалог порциями, а не только последние 7 обменов.
            # После каждой порции факты сразу сохраняются; при сбое весь
            # незакрытый журнал остаётся доступен для безопасного повтора.
            for offset in range(0, len(turn_log), EXTRACT_WINDOW):
                chunk = turn_log[offset:offset + EXTRACT_WINDOW]
                try:
                    facts = await extract_facts(brain, "\n\n".join(chunk))
                    counts = memory.apply_facts(facts)
                    added["core"] += counts["core"]
                    added["facts"] += counts["facts"]
                except BrainUnavailable as exc:
                    ingestion_ok = False
                    print(f"  [!] не удалось извлечь ({exc}) — незавершённый диалог сохранён")
                    break
        # Закрываем журнал только после успешного усвоения. Если извлечение
        # выключено настройкой, оставляем поведение штатного завершения.
        if ingestion_ok or not cfg.memory.extract_on_exit:
            journal.session_end()
        print(
            f"  {soul.name} засыпает. в ядре: {added['core']}, "
            f"новых фактов: {added['facts']}, состояние сохранено.\n"
        )

        emotions.save(EMOTIONS_PATH)
        personality.save(PERSONALITY_PATH)
        memory.close()

