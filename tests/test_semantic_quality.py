"""Проверка памяти НАСТОЯЩЕЙ BGE-M3 (не фолбэком) и разделение двух механизмов.

Два разных вопроса — два разных механизма, и это не придирка, а результат
измерения на матрице похожестей:

  A. «Как меня зовут / где я живу» — одиночные устойчивые сведения.
     Векторный поиск их НЕ тянет: правильные и случайные кандидаты различаются
     в третьем знаке косинуса (0.586 против 0.582). Такие вещи обязано отдавать
     ядро памяти — оно попадает в контекст всегда, без поиска.

  B. «Что я пережил / о чём говорили» — эпизоды и переживания. Вот здесь
     семантика сильна: находит по смыслу, без единого общего слова.

    .venv/Scripts/python.exe tests/test_semantic_quality.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from aizel.core.config import DATA_DIR
from aizel.memory.facade import Memory
from aizel.memory.semantic import ExtractedFact

# B. То, ради чего нужен векторный поиск: смысл без общих слов.
SEARCH_CASES = [
    ("я обожаю смотреть на звёзды и галактики", "что мне нравится в космосе", "звёзды"),
    ("у меня умер кот Барсик, мне очень тяжело", "как звали моего питомца", "Барсик"),
    ("я поссорился с братом и мы не разговариваем", "ладится ли с родными", "брат"),
    ("в пятницу у меня защита диплома, я ужасно волнуюсь", "какое важное событие близко", "диплом"),
    ("я бросил курить три месяца назад", "что я перестал делать", "курить"),
    ("меня повысили и теперь я руководю отделом", "как у меня на службе", "отдел"),
]

# A. То, что обязано жить в ядре.
CORE_CASES = [
    ("имя", "Его зовут Алекс", "Алекс"),
    ("город", "Он живёт в Санкт-Петербурге", "Петербург"),
]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="aizel_sem_"))
    mem = Memory(tmp, models_dir=DATA_DIR / "models")
    print("загружаю BGE-M3…")
    print("движок:", mem.warm_up())
    if mem.degraded:
        print("\n[!] BGE-M3 не загрузилась — тест про семантику бессмысленен.")
        mem.close()
        shutil.rmtree(tmp, ignore_errors=True)
        return 2

    print("\nA. ЯДРО ПАМЯТИ (без поиска, всегда в контексте)")
    for slot, text, _ in CORE_CASES:
        mem.store.upsert_core(slot, text)
    core = mem.store.all_core()
    print("   ядро:", core)
    core_ok = sum(1 for _, _, frag in CORE_CASES if any(frag.lower() in c.lower() for c in core))
    for slot, text, frag in CORE_CASES:
        hit = any(frag.lower() in c.lower() for c in core)
        print(f"   {'ok  ' if hit else 'MISS'} slot={slot!r} ожидаем {frag!r}")

    print(f"\nB. СЕМАНТИЧЕСКИЙ ПОИСК ({len(SEARCH_CASES)} эпизодов)")
    for text, _, _ in SEARCH_CASES:
        mem.remember_turn(text, "понятно", importance=0.8)

    search_ok = 0
    for text, query, expected in SEARCH_CASES:
        found = mem.recall(query, top_k=3)
        best = found[0].text if found else ""
        ok = expected.lower() in best.lower()
        search_ok += ok
        score = f"{found[0].score:.2f} (sim={found[0].similarity:.2f})" if found else "—"
        print(f"   {'ok  ' if ok else 'MISS'} {query!r:34} → {best[:44]!r:48} {score}")

    # Дедуп фактов: та же мысль второй раз не должна плодить запись.
    mem.apply_facts([ExtractedFact("он любит звёзды", 0.7)])
    before = mem.store.stats()["facts"]
    mem.apply_facts([ExtractedFact("он любит звёзды", 0.7)])
    dup = mem.store.stats()["facts"] - before
    print(f"\nC. Дедуп: повторный факт добавил записей: {dup} (ожидаем 0)")

    mem.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nядро: {core_ok}/{len(CORE_CASES)}   семантика: {search_ok}/{len(SEARCH_CASES)}   дедуп: {'ok' if dup == 0 else 'ПРОВАЛ'}")
    good = core_ok == len(CORE_CASES) and search_ok >= len(SEARCH_CASES) - 1 and dup == 0
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
