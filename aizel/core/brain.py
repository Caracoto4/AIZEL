"""Мозг Айзель: асинхронный клиент к KoboldCpp.

KoboldCpp поднимает OpenAI-совместимый API, поэтому мы говорим с моделью по
стандартному протоколу /v1/chat/completions. Плюс — при желании движок
меняется одной строкой в конфиге, весь остальной код не трогается.

Здесь нет никакой «личности» — только доставка текста туда и обратно,
стриминг, ретраи и диагностика.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from .config import BrainConfig

log = logging.getLogger("aizel.brain")

# KoboldCpp при отсутствии Jinja подставляет Alpaca-шаблон. Если модель начнёт
# «дорисовывать» следующий ход диалога, эти маркеры обрывают генерацию.
STOP_SEQUENCES = [
    "### Human:",
    "### User:",
    "### Assistant:",
    "\n\n\n\n",
    "<|eot_id|>",
    "<|end_of_text|>",
    # RP-модель обучена на файлах диалогов и иногда «дорисовывает» следующий
    # блок разметки вместо ответа: «+++++ dialogues/dialogue_41.md».
    "+++++",
    "## Dialogue",
    "<|im_start|>",
    "<|im_end|>",
]


def sanitize(text: str) -> str:
    """Убирает только возвраты каретки.

    Отдельный \\r в консоли затирает уже напечатанное — на замере это дало
    «сказать    ЗМ точно» вместо нормального текста. Пробелы не трогает:
    для стриминга резкие strip() на кусках склеил бы слова.
    """
    return text.replace("\r\n", "\n").replace("\r", " ")


def clean(text: str) -> str:
    """Полная обработка готового ответа: каретки + обрамляющие пробелы."""
    return sanitize(text).strip()


# Страховка на случай, если сервер не оборвёт генерацию по стоп-маркеру.
LEAK_MARKERS = ("+++++", "## Dialogue", "<|im_start|>", "<|end_of_text|>")


def trim_artifacts(text: str) -> str:
    """Обрезает ответ по первому признаку «модель полезла в разметку датасета»."""
    out = clean(text)
    cut = len(out)
    for marker in LEAK_MARKERS:
        pos = out.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    return out[:cut].rstrip()


class BrainUnavailable(RuntimeError):
    """Модель не отвечает: KoboldCpp выключен, порт занят или модель грузится."""


@dataclass
class Generation:
    """Результат одного прогона с метриками — без них «тормозит или нет» гадать."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_sec: float = 0.0
    stop_reason: str | None = None

    @property
    def tps(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        return self.completion_tokens / self.elapsed_sec


class Brain:
    """Тонкая обёртка над чат-комплишнсами. Один экземпляр — одна сессия к API."""

    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "Brain":
        await self.start()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.cfg.timeout_sec, connect=10.0)
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ health

    async def ping(self) -> dict:
        """Есть ли модель на другом конце. Используется в `run.py doctor`."""
        await self.start()
        assert self._client is not None
        try:
            r = await self._client.get(self.cfg.models_url, timeout=8.0)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as exc:
            raise BrainUnavailable(
                f"KoboldCpp не отвечает на {self.cfg.base_url}: {exc!r}"
            ) from exc

        models = [m.get("id", "?") for m in data.get("data", [])]
        return {"ok": True, "models": models, "url": self.cfg.base_url}

    # ------------------------------------------------------------------- chat

    def _payload(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None,
        temperature: float | None,
        stream: bool,
        extra: dict,
    ) -> dict:
        body: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p,
            "top_k": self.cfg.top_k,
            "min_p": self.cfg.min_p,
            "repetition_penalty": self.cfg.rep_pen,
            "frequency_penalty": self.cfg.freq_pen,
            "stop": STOP_SEQUENCES,
            "stream": stream,
        }
        if self.cfg.use_jinja:
            # Просим KoboldCpp взять чат-шаблон прямо из GGUF, а не угадывать.
            body["use_jinja"] = True
        body.update(extra)
        return body

    async def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **extra,
    ) -> Generation:
        """Полный ответ одним куском. Ретраи только на сетевые сбои и 5xx."""
        await self.start()
        assert self._client is not None
        body = self._payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            extra=extra,
        )

        loop = asyncio.get_running_loop()
        started = loop.time()
        last_error: Exception | None = None

        for attempt in range(self.cfg.retries + 1):
            try:
                r = await self._client.post(self.cfg.chat_url, json=body)
                if r.status_code >= 500:
                    raise BrainUnavailable(f"KoboldCpp вернул {r.status_code}")
                r.raise_for_status()
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                usage = data.get("usage") or {}
                text = (choice.get("message") or {}).get("content") or ""
                gen = Generation(
                    text=trim_artifacts(text),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    elapsed_sec=loop.time() - started,
                    stop_reason=choice.get("finish_reason"),
                )
                log.info(
                    "генерация: %d токенов за %.1f с (%.1f ток/с), стоп=%s",
                    gen.completion_tokens,
                    gen.elapsed_sec,
                    gen.tps,
                    gen.stop_reason,
                )
                return gen
            except (httpx.HTTPError, BrainUnavailable, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.cfg.retries:
                    pause = 1.5 * (attempt + 1)
                    log.warning("попытка %d не удалась (%r), повтор через %.1f с",
                                attempt + 1, exc, pause)
                    await asyncio.sleep(pause)

        raise BrainUnavailable(f"модель не ответила: {last_error!r}")

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **extra,
    ) -> AsyncIterator[str]:
        """Постепенная выдача текста по мере генерации — для живого печатания.

        Держит скользящий буфер: кусок печатается только когда становится
        ясно, что он не начало стоп-маркера. Иначе «+++» успело бы мелькнуть
        на экране раньше, чем мы его отрежем.
        """
        await self.start()
        assert self._client is not None
        body = self._payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra=extra,
        )
        guard = max(len(m) for m in LEAK_MARKERS) - 1
        buffer = ""
        leaked = False

        async with self._client.stream(
            "POST", self.cfg.chat_url, json=body
        ) as r:
            if r.status_code >= 400:
                detail = (await r.aread()).decode("utf-8", "replace")[:300]
                raise BrainUnavailable(f"KoboldCpp {r.status_code}: {detail}")
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                delta = ((obj.get("choices") or [{}])[0].get("delta") or {})
                piece = delta.get("content")
                if not piece:
                    continue
                buffer += sanitize(piece)
                if any(m in buffer for m in LEAK_MARKERS):
                    leaked = True
                    break
                if len(buffer) > guard:
                    safe, buffer = buffer[:-guard], buffer[-guard:]
                    yield safe

        if not leaked and buffer:
            yield buffer
