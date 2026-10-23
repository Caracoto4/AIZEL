"""Эмоциональное состояние Айзель: модель PAD (Pleasure / Arousal / Dominance).

Зачем не «весёлый/грустный» одной кнопкой: раздельные оси дают переносы
(усталое недовольство ≠ злое недовольство) и позволяют считать затухание.

Ключевое — состояние **затухает к базовой линии** по прошествии времени,
а не живёт вечно. Иначе одно резкое слово злило бы её сутки.

`decay_to_baseline` вызывается при пробуждении с реальным числом прошедших
часов: это то, что отличает живое существо от переменной, обновляемой на запрос.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

AXES = ("P", "A", "D")
AXIS_NAMES = {"P": "удовольствие", "A": "возбуждение", "D": "контроль"}

# Скорость возврата к базовой линии, 1/час. 0.12 → половина эмоции за ~6 часов.
DECAY_PER_HOUR = 0.12
# Ниже этого порога событие считается незначимым и в историю не пишется.
MIN_IMPACT = 0.02


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class Mood:
    """Мгновенный снимок настроения."""

    pleasure: float = 0.15
    arousal: float = 0.25
    dominance: float = 0.10
    # Настроение, оставленное последним разговором: влияет на то,
    # какие воспоминания вспомнятся ( Фаза 2, резонанс при извлечении).
    lingering_topic: str = ""

    def as_dict(self) -> dict[str, float]:
        return {"P": self.pleasure, "A": self.arousal, "D": self.dominance}


@dataclass
class Emotions:
    """Накопитель эмоций с затуханием и историей значимых событий."""

    baseline: dict[str, float] = field(
        default_factory=lambda: {"P": 0.15, "A": 0.20, "D": 0.10}
    )
    state: Mood = field(default_factory=Mood)
    # Насколько сильно текущее состояние отклонилось от базовой линии:
    # метрика «на взводе ли она сейчас».
    intensity: float = 0.0
    history: list[dict] = field(default_factory=list)
    updated_at: str = ""

    # -------------------------------------------------------------- затухание

    def decay_to_baseline(self, hours: float) -> None:
        """Протягивает состояние к базовой линии на величину, прошедшую с `updated_at`."""
        if hours <= 0:
            return
        # Экспоненциальное затухание: быстро в первый час, почти стоит через сутки.
        k = 1.0 - math.exp(-DECAY_PER_HOUR * hours)
        s = self.state
        s.pleasure = s.pleasure + (self.baseline["P"] - s.pleasure) * k
        s.arousal = s.arousal + (self.baseline["A"] - s.arousal) * k
        s.dominance = s.dominance + (self.baseline["D"] - s.dominance) * k
        self._recompute_intensity()
        self.touch()

    # ----------------------------------------------------------------- события

    def react(
        self,
        *,
        pleasure: float = 0.0,
        arousal: float = 0.0,
        dominance: float = 0.0,
        cause: str = "",
        topic: str = "",
        sensitivity: float = 1.0,
    ) -> bool:
        """Записывает событие. `sensitivity` обычно = нейротизм: одним больнее.

        Возвращает True, если событие оказалось значимым (его стоит помнить).
        """
        s = self.state
        s.pleasure = _clamp(s.pleasure + pleasure * sensitivity)
        s.arousal = _clamp(s.arousal + arousal * sensitivity)
        s.dominance = _clamp(s.dominance + dominance * sensitivity)
        self._recompute_intensity()
        if topic:
            s.lingering_topic = topic

        impact = max(abs(pleasure), abs(arousal), abs(dominance)) * sensitivity
        if impact >= MIN_IMPACT:
            self.history.append(
                {
                    "at": self.now(),
                    "cause": cause,
                    "topic": topic,
                    "impact": round(impact, 3),
                    "pad": self.state.as_dict(),
                }
            )
            # История не должна расти бесконечно: храним только значимое.
            self.history = self.history[-100:]
        self.touch()
        return impact >= MIN_IMPACT

    def _recompute_intensity(self) -> None:
        d = [
            self.state.pleasure - self.baseline["P"],
            self.state.arousal - self.baseline["A"],
            self.state.dominance - self.baseline["D"],
        ]
        self.intensity = round(math.sqrt(sum(x * x for x in d) / 3.0), 4)

    # ------------------------------------------------------------------- вывод

    def describe(self) -> str:
        """Человеческое описание состояния для подмешивания в контекст."""
        p, a, d = self.state.pleasure, self.state.arousal, self.state.dominance
        if p > 0.45:
            mood = "тепло и легко"
        elif p > 0.15:
            mood = "ровно"
        elif p > -0.15:
            mood = "скорее глухо, чем хорошо"
        elif p > -0.5:
            mood = "тяжело на душе"
        else:
            mood = "откровенно плохо"

        energy = (
            "на подъёме, энергии много"
            if a > 0.45
            else "спокойно"
            if a > 0.0
            else "вяло, хочется тишины"
        )
        grip = (
            "уверенно держу себя в руках"
            if d > 0.3
            else "неуверенно, легче уступить"
            if d < -0.1
            else "обычно"
        )
        return f"чувствую себя {mood}; {energy}; {grip}"

    def tone_directive(self) -> str:
        """Превращает PAD в поведенческую поправку к ответу."""
        p, a, d = self.state.pleasure, self.state.arousal, self.state.dominance
        out: list[str] = []
        if p < -0.2:
            out.append("отвечай короче и суше обычного, без энтузиазма")
        elif p > 0.5:
            out.append("в голосе заметно тепло, делись чуть большим")
        if a > 0.5:
            out.append("темп быстрый, спрашивай встречное")
        elif a < -0.3:
            out.append("сил мало, не развивай длинные темы")
        if d < -0.3:
            out.append("легче соглашайся, чем настаивать")
        elif d > 0.4:
            out.append("если не согласна — скажи прямо и настаивай")
        return "; ".join(out)

    # ------------------------------------------------------------------ время

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def touch(self) -> None:
        self.updated_at = self.now()

    def hours_since_update(self) -> float:
        if not self.updated_at:
            return 0.0
        try:
            then = datetime.fromisoformat(self.updated_at)
        except ValueError:
            return 0.0
        delta = datetime.now(timezone.utc) - then
        return max(0.0, delta.total_seconds() / 3600.0)

    # -------------------------------------------------------------------- БД-лайт

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "baseline": self.baseline,
                    "state": {
                        "pleasure": self.state.pleasure,
                        "arousal": self.state.arousal,
                        "dominance": self.state.dominance,
                        "lingering_topic": self.state.lingering_topic,
                    },
                    "intensity": self.intensity,
                    "history": self.history,
                    "updated_at": self.updated_at,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Emotions":
        """Загружает состояние и СРАЗУ протягивает его к базовой линии
        на то время, что её не запускали. Это и есть «она жила без тебя»."""
        e = cls()
        if not path.exists():
            e.touch()
            return e
        data = json.loads(path.read_text(encoding="utf-8"))
        e.baseline.update(data.get("baseline") or {})
        st = data.get("state") or {}
        e.state = Mood(
            pleasure=float(st.get("pleasure", 0.15)),
            arousal=float(st.get("arousal", 0.25)),
            dominance=float(st.get("dominance", 0.10)),
            lingering_topic=st.get("lingering_topic", ""),
        )
        e.history = data.get("history") or []
        e.updated_at = data.get("updated_at") or ""
        e._recompute_intensity()
        e.decay_to_baseline(e.hours_since_update())
        return e
