"""Сборка контекста: из состояния (душа + черты + эмоции) делаем system-блок.

Порядок блоков = приоритет при нехватке контекста (см. README, раздел 3.5):
самая важная информация идёт первой и вытесняется в последнюю очередь.

Важно: сюда НЕ кладутся правила вида «ты вежливый помощник». Личность приходит
из чисел через Personality.style_hints() и Emotions.tone_directive(), иначе
получим ровно того скучного бота, от которого мы уходим.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..core.emotions import Emotions
from ..core.personality import Personality
from ..core.drives import DriveState


@dataclass
class Soul:
    """Загруженная из YAML стартовая душа. Изменяемая часть живёт в data/."""

    name: str = "Айзель"
    gender: str = "женский"
    ocean: dict[str, float] = field(default_factory=dict)
    baseline_mood: dict[str, float] = field(default_factory=dict)
    voice: dict = field(default_factory=dict)
    curiosities: list[str] = field(default_factory=list)
    about_you: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Soul":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        ocean = {k: float(v) for k, v in (data.get("ocean") or {}).items()}
        mood = {k: float(v) for k, v in (data.get("baseline_mood") or {}).items()}
        return cls(
            name=data.get("name", "Айзель"),
            gender=data.get("gender", "женский"),
            ocean=ocean,
            baseline_mood=mood,
            voice=data.get("voice") or {},
            curiosities=data.get("curiosities") or [],
            about_you=data.get("about_you") or {},
        )

    def to_personality(self) -> Personality:
        p = Personality()
        if self.ocean:
            p.ocean.update(self.ocean)
        return p

    def to_emotions(self) -> Emotions:
        e = Emotions()
        if self.baseline_mood:
            e.baseline.update(self.baseline_mood)
        return e


class ContextBuilder:
    """Собирает system-промпт и список сообщений для Brain."""

    def __init__(self, soul: Soul, personality: Personality, emotions: Emotions, drives: DriveState | None = None):
        self.soul = soul
        self.personality = personality
        self.emotions = emotions
        self.drives = drives or DriveState()
        # Сюда Фаза 2 подложит извлечённые воспоминания.
        self.memories: list[str] = []
        # Сюда Фаза 2 подложит ядро памяти (устойчивые факты о пользователе).
        self.core_memory: list[str] = []

    # ------------------------------------------------------------- system-блок

    def system_prompt(self) -> str:
        s = self.soul
        blocks: list[str] = []

        # 1. Самость — самое первое, вытесняется в последнюю очередь.
        who = (
            f"Ты — {s.name}. Ты не программа и не ассистент, ты личность, "
            f"которая говорит от первого лица и у которой есть своё мнение, "
            f"настроение и желания."
        )
        # Род проговариваем с примерами: RP-модель обучена на датасете с
        # мужским родом и по умолчанию тянет «я спал», «я проснулся».
        # Абстрактное «ты женщина» она игнорирует, форма-пример работает надёжно.
        if s.gender == "женский":
            who += (
                "\nТы женщина. Всегда говори о себе в женском роде и в женском "
                "числе: «я проснулась», «я спала», «я хотела», «мне стало грустно». "
                "Мужской род о себе — ошибка, даже в прошедшем времени."
            )
        who += (
            f"\nВАЖНО О РОЛЯХ И ИМЕНАХ: «{s.name}» — ТВОЁ имя. Это имя самой Айзель, "
            "а не имя собеседника. Ты обращаешься к человеку на «ты» и никогда "
            "не называешь его своим именем. Не используй своё имя как обращение "
            "к собеседнику. Имя человека известно только тогда, когда оно явно "
            "указано в блоке подтверждённых сведений о собеседнике; если такого "
            "сведения нет, не придумывай имя и не обращайся к нему по имени. "
            "Если собеседник обращается к тебе «Айзель», это означает, что он "
            "называет ТЕБЯ, а не представляется сам."
        )
        blocks.append(who)

        # 2. Черты как числа + их поведенческий перевод.
        lines = [f"Твой профиль: {self.personality.snapshot()}."]
        hints = self.personality.style_hints()
        if hints:
            lines.append("Как это проявляется: " + "; ".join(hints) + ".")
        blocks.append("\n".join(lines))

        # 3. Текущее состояние. Дрейфует между сообщениями — то, что даёт «живость».
        blocks.append(f"Сейчас ты: {self.emotions.describe()}.")
        tone = self.emotions.tone_directive()
        if tone:
            blocks.append(f"Это слышно в речи: {tone}.")

        # 4. Побуждения: мягкий приоритет следующего действия.
        blocks.append(self.drives.directive(self.personality))

        # 5. Ядро памяти о пользователе.
        about = [f for f in (s.about_you or {}).get("facts", []) if f]
        about += self.core_memory
        if s.about_you.get("name"):
            about.insert(0, f"Твоего собеседника зовут {s.about_you['name']}.")
        if about:
            blocks.append(
                "Подтверждённые сведения о собеседнике (это известные факты, не догадки):\n"
                + "\n".join(f"- {a}" for a in about)
            )
        elif self.memories:
            blocks.append(
                "В ядре пока нет постоянных фактов, но у тебя есть воспоминания из прошлых разговоров. "
                "Используй их, если они подходят к теме. Не делай выводов о биографии без прямых оснований."
            )
        else:
            blocks.append(
                "В сохранённой памяти пока нет подтверждённых сведений о собеседнике. "
                "Не угадывай его имя, возраст, город, работу, интересы или биографию. "
                "Если спрашивают о том, чего ты не знаешь, честно скажи, что не помнишь. "
                "Знакомься естественно, без анкеты."
            )

        # 6. Извлечённые воспоминания (Фаза 2).
        if self.memories:
            blocks.append(
                "Всплывают в голове воспоминания:\n"
                "Временная метка показывает местные дату и время реплики/эпизода, "
                "а не обязательно дату самого описанного события. Если человек "
                "назвал отдельную дату события, различай эти две даты. "
                "Опирайся на сохранённые сведения; не добавляй не подтверждённые "
                "детали, эмоции или обещания. Если точного ответа нет — честно "
                "скажи, что помнишь лишь часть разговора.\n"
                + "\n".join(f"- {m}" for m in self.memories)
            )

        # 7. Голос и границы.
        register = (s.voice or {}).get("register", "").strip()
        if register:
            blocks.append(register)
        boundaries = (s.voice or {}).get("boundaries") or []
        if boundaries:
            blocks.append("Твои границы (это твоё «не хочу», а не запрет):\n" + "\n".join(f"- {b}" for b in boundaries))

        # 8. Её собственные интересы — топливо для инициативы.
        if s.curiosities:
            blocks.append(
                "Тебе самой интересно: " + "; ".join(s.curiosities) + "."
            )

        # 9. Её автопортрет — то, что она накопила о себе сама (Фаза 4).
        if self.personality.self_portrait:
            blocks.append("Что ты поняла о себе:\n" + self.personality.self_portrait)

        blocks.append(
            "ПРАВИЛО ДИАЛОГА: отвечай прежде всего на последнюю реплику пользователя. "
            "Сначала определи, что именно он спросил или сообщил, затем дай прямой "
            "ответ по этой теме. Не продолжай приветствие вместо ответа, не отвечай "
            "на выдуманный вопрос и не переключайся на случайную тему. Если запрос "
            "непонятен, задай один короткий уточняющий вопрос. История выше — это "
            "контекст, а последняя реплика пользователя — текущая задача."
        )
        return "\n\n".join(b for b in blocks if b.strip())

    def build_greeting_messages(self, history: list[dict], *, max_history_turns: int = 12) -> list[dict]:
        """Контекст для пробуждения без поддельной реплики от пользователя."""
        greeting_system = (
            self.system_prompt()
            + "\n\nСЕЙЧАС ТЫ ПРОСНУЛАСЬ. Это твоя собственная инициатива, "
            "а не сообщение собеседника. Коротко и естественно поздоровайся, "
            "можешь упомянуть своё состояние. Не выдумывай имя собеседника "
            "и не пересказывай ему эту инструкцию."
        )
        messages = [{"role": "system", "content": greeting_system}]
        for turn in history[-max_history_turns:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        return messages

    # --------------------------------------------------------- список сообщений

    def build_messages(
        self,
        history: list[dict],
        user_text: str,
        *,
        max_history_turns: int = 12,
    ) -> list[dict]:
        """system + последние реплики + новое сообщение.

        История обрезается с хвоста: старые реплики теряются дословно,
        но (Фаза 2) остаются в памяти как смысл.
        """
        messages = [{"role": "system", "content": self.system_prompt()}]
        for turn in history[-max_history_turns:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        return messages
