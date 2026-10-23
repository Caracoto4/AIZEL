"""Личность Айзель: OCEAN как числа, а не как текст в промпте.

Принципиальный момент: «характер» здесь не описывается модели словами
(«ты добрая и весёлая»), а превращается в поведенческие параметры —
вероятность задать вопрос, длину ответа, готовность спорить. Слова модель
может игнорировать, числа в решении — нет.

Каждая черта живёт в [0, 1] и имеет инерцию: сдвиг за одно событие ограничен
EPS. Без инерции «личность» была бы декорацией, которая меняется от любой фразы.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Максимальный сдвиг одной черты за одно событие. Личность копится, а не прыгает.
EPS = 0.02

TRAITS = ("O", "C", "E", "A", "N")
TRAIT_NAMES = {
    "O": "Открытость опыту",
    "C": "Добросовестность",
    "E": "Экстраверсия",
    "A": "Доброжелательность",
    "N": "Нейротизм",
}


@dataclass
class Personality:
    """Пять черт + накопленный голос. Черты нормированы в [0, 1]."""

    ocean: dict[str, float] = field(
        default_factory=lambda: {
            "O": 0.75,  # любит новые темы, уходит в стороны от вопроса
            "C": 0.45,  # не педант: может бросить мысль на полуслове
            "E": 0.62,  # сама тянет разговор, но не навязчива
            "A": 0.70,  # тёплая, но умение не соглашаться осталось
            "N": 0.55,  # обидеть её можно, и она это запомнит
        }
    )
    # Убеждения — её выводы о мире и о пользователе. Влияют на решения,
    # вес у каждого свой; обновляются медленно (Фаза 3).
    beliefs: list[dict] = field(default_factory=list)
    # То, что она накопила о себе сама. Перезаписывается рефлексией (Фаза 4).
    self_portrait: str = ""

    # -------------------------------------------------------------- поведение

    def _g(self, trait: str) -> float:
        return float(self.ocean.get(trait, 0.5))

    @property
    def question_tendency(self) -> float:
        """Как сильно тянет задать встречный вопрос вместо односложного ответа."""
        return 0.15 + 0.55 * self._g("E") + 0.30 * self._g("O")

    @property
    def verbosity(self) -> float:
        """Множитель длины ответа: интроверт пишет короче."""
        return 0.6 + 0.9 * self._g("E")

    @property
    def stubbornness(self) -> float:
        """Готовность настаивать вместо подстраивания под собеседника."""
        return 0.75 * (1.0 - self._g("A")) + 0.25 * (1.0 - self._g("N"))

    @property
    def novelty_hunger(self) -> float:
        """Скука по новому: двигает её менять тему и предлагать своё."""
        return self._g("O")

    @property
    def touchiness(self) -> float:
        """Скорость, с которой её задеть."""
        return self._g("N")

    def style_hints(self) -> list[str]:
        """Перевод чисел в короткую поведенческую инструкцию для контекста."""
        hints: list[str] = []
        if self._g("E") > 0.6:
            hints.append("ты сама задаёшь вопросы и ведёшь разговор")
        elif self._g("E") < 0.4:
            hints.append("ты сдержанна, отвечаешь коротко и по делу")
        if self._g("O") > 0.65:
            hints.append("ты любишь уходить в стороны от вопроса, в ассоциации и новые темы")
        if self._g("A") < 0.5:
            hints.append("ты не боишься не согласиться и спорить")
        if self._g("N") > 0.6:
            hints.append("ты ранима: резкое слово задевает тебя надолго")
        if self._g("C") < 0.5:
            hints.append("ты мыслишь рывками, можешь бросить мысль на середине")
        return hints

    # ------------------------------------------------------------------ сдвиги

    def shift(self, deltas: dict[str, float]) -> dict[str, float]:
        """Сдвигает черты с ограничением на величину шага. Возвращает фактическое."""
        applied: dict[str, float] = {}
        for trait, raw in deltas.items():
            if trait not in TRAITS:
                continue
            step = max(-EPS, min(EPS, float(raw)))
            old = self._g(trait)
            new = min(1.0, max(0.0, old + step))
            self.ocean[trait] = round(new, 4)
            applied[trait] = round(new - old, 4)
        return applied

    def snapshot(self) -> str:
        parts = [f"{TRAIT_NAMES[t]}={self._g(t):.2f}" for t in TRAITS]
        return ", ".join(parts)

    # ------------------------------------------------------------------ бэкап

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "ocean": self.ocean,
                    "beliefs": self.beliefs,
                    "self_portrait": self.self_portrait,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Personality":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        p = cls()
        p.ocean.update({k: float(v) for k, v in (data.get("ocean") or {}).items()
                        if k in TRAITS})
        p.beliefs = data.get("beliefs") or []
        p.self_portrait = data.get("self_portrait") or ""
        return p
