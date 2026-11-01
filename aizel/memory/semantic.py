"""Превращает слова в то, что Айзель о тебе знает.

Диалог сам по себе — шум. «А, ну у меня ещё кот сдох неделю назад, грустно»
в дословном виде бесполезно: при поиске по «как дела» оно не вспомнится.
Поэтому после разговора отдельным спокойным прогоном модель вытаскивает смысл
короткими утверждениями, которые потом ищутся по смыслу.

Отдельный вызов, а не «пусть отвечает заодно»: смешивать генерацию ответа и
анализ в одном прогоне нельзя — модель начинает выдумывать факты, чтобы
оправдать длинный ответ. Здесь же задача сформулирована сухо и в JSON.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger("aizel.memory.semantic")

EXTRACT_SYSTEM = """Ты ведёшь заметки о собеседнике, чтобы помнить его позже.
Выписывай только УСТОЙЧИВОЕ: имена, близких, привычки, интересы, важные события,
то, что о нём явно узнали. Не выписывай светскую болтовню, приветствия и свои догадки.

Ответь СТРОГО JSON-массивом объектов вида
  {"text": "...", "importance": 0.0-1.0, "slot": "..."}
text — короткое утверждение в третьем лице, одна мысль.
importance — 0.2 бытовая мелочь, 0.5 обычное, 0.8 значимое, 1.0 критичное
(имя, близкие, здоровье, травма).
slot — короткое имя поля ТОЛЬКО для одиночных неизменяемых сведений, которые
нужно знать всегда, а не искать: "имя", "город", "работа", "возраст", "семья".
Для всего остального slot: "". Если устойчивого нет — []."""


def _parse_json_array(text: str) -> list:
    """Модели любят обрамлять JSON пояснениями и ```-блоками. Вытаскиваем массив."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


@dataclass
class ExtractedFact:
    text: str
    importance: float = 0.5
    # Не пустой slot = «знать всегда», а не «искать по запросу». См. facade.apply_facts.
    slot: str = ""


# Одиночные сведения вытаскиваем регулярками, без обращения к модели.
# Причина не в лени, а в очереди: к KoboldCpp запросы идут последовательно, и
# прогон извлечения, запущенный в фоне, встал бы ВПЕРЕД следующего ответа — при
# 7 ток/с это минута простоя на пустом месте. Имя и город при этом обязаны
# оказываться в ядре немедленно, иначе она «не знает, как тебя зовут» в разговоре,
# где ты это сказал минуту назад.
#
# re.I нужен, чтобы поймать «Меня зовут…» в начале реплики, — но вместе с ним
# класс [А-ЯЁ] начинает принимать и строчные. Поэтому настоящее имя отделяем
# постпроверкой первой буквы, а не регистром в шаблоне.
_CORE_PATTERNS: tuple[tuple[str, re.Pattern, bool], ...] = (
    # «меня/мне» обязательно: без него «а тебя зовут Айзель?» записало бы её
    # собственное имя как твоё.
    (
        "имя",
        re.compile(r"(?:меня\\s+(?:тоже\\s+)?(?:зовут|звать)|я)\\s+([А-ЯЁ][а-яё-]{1,19})", re.I),
        True,   # значение — имя, пишем его само по себе
    ),
    (
        "город",
        re.compile(r"(?:живу|родом|переех\w+)\s+(?:в\s+|из\s+)?([\w-]{2,20})", re.I),
        False,  # значение в падеже («в Питере»), пишем всю фразу целиком
    ),
    (
        "возраст",
        re.compile(r"(?:мне|мне сейчас)\s+(?:уже\s+)?(\d{1,2})\s*(?:лет|год|года)", re.I),
        False,
    ),
    (
        "работа",
        re.compile(r"(?:работаю|учусь)\s+(?:кем\s+|в\s+|на\s+)?([\w -]{2,30}?)(?=[,.!?]|$)", re.I),
        False,
    ),
)

# Стоп-слова: после «я/живу/мне» они означают не имя и не город.
_STOP = frozenset(
    """сегодня завтра потом может вроде кажется честно короче впрочем кстати
    не ни он она оно они мы вы ты себя там тут здесь везде дома много мало
    уже ещё пока просто может быть""".split()
)

# Союзы, на которых обрываем выхваченную фразу: «работаю вёрсткой и ненавижу
# джунов» — это про верстку, а не про «вёрсткой и ненавижу джунов».
_CUT = re.compile(r"\s+(?:и|а|но|да|или|что|чтобы|котор\w*|потому|поэтому|хотя|если)\b")


def quick_core(text: str) -> list[ExtractedFact]:
    """Мгновенно вытаскивает одиночные сведения (имя, город, возраст, работа).

    Возвращает то же, что и `extract_facts`, но без прогона через модель:
    вызывается на каждую реплику и не создаёт очереди к GPU.
    """
    if not text:
        return []
    found: list[ExtractedFact] = []
    seen: set[str] = set()
    for slot, pattern, is_name in _CORE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        value = m.group(1).strip(" .,!:;-")
        # Имя и город — с большой буквы (под re.I шаблон принимает и строчные).
        # Возраст — цифры, работа — строчная форма.
        if slot in ("имя", "город"):
            if not value or not value[:1].isupper():
                continue
        if value.lower() in _STOP:
            continue
        phrase = value if is_name else m.group(0).strip(" .,!:;-")
        if not is_name:
            # Обрываем на первом союзе: в ядре должна быть одна мысль, а не
            # вся реплика целиком — иначе «город: живу в Питере и вообще всё
            # хорошо» потом всплывёт как воспоминание про город.
            cut = _CUT.search(phrase)
            if cut:
                phrase = phrase[: cut.start()].strip(" ,.:;-")
        if len(phrase) > 40:
            continue
        key = f"{slot}={phrase.lower()}"
        if key in seen:
            continue
        seen.add(key)
        found.append(ExtractedFact(text=phrase, importance=0.9, slot=slot))
    return found


async def extract_facts(brain, dialog_text: str) -> list[ExtractedFact]:
    """Вытаскивает из фрагмента диалога устойчивые сведения.

    Один прогон вместо двух: раньше важность считалась вторым вызовом, и это
    была лишняя минута ожидания при выходе за сомнительную точность. Модель
    возвращает importance сразу вместе с текстом.
    """
    if not dialog_text.strip():
        return []

    result = await brain.chat(
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": dialog_text},
        ],
        max_tokens=450,
        temperature=0.2,
    )
    items = _parse_json_array(result.text)
    out: list[ExtractedFact] = []
    for item in items:
        if isinstance(item, str):
            text, importance, slot = item.strip(), 0.5, ""
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            try:
                importance = max(0.0, min(1.0, float(item.get("importance", 0.5))))
            except (TypeError, ValueError):
                importance = 0.5
            slot = str(item.get("slot") or "").strip().lower()
        else:
            continue
        if text:
            out.append(ExtractedFact(text=text, importance=importance, slot=slot))
    log.info("извлечено %d фактов (слотовых: %d)",
             len(out), sum(1 for f in out if f.slot))
    return out
