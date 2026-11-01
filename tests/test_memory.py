"""Тесты памяти. Специально без BGE-M3: используется лексический фолбэк,
чтобы проверка была быстрой и не зависела от скачанных весов.

    python -m pytest tests -q        (если pytest установлен)
    python tests/test_memory.py      (без pytest)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aizel.core.embeddings import Embedder
from aizel.memory.facade import Memory, _guess_importance
from aizel.memory.retrieval import Retriever, RetrievalSettings, _human_age
from aizel.memory.store import MemoryStore
from aizel.memory.semantic import quick_core
from aizel.body.boot import ContextBuilder, Soul
from aizel.core.personality import Personality
from aizel.core.emotions import Emotions


def _tmp_db(tmp: Path) -> Path:
    return tmp / "test.db"


class _Tmp:
    """Мини-контекст временной папки без pytest-зависимости."""

    def __enter__(self):
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="aizel_test_"))
        return self.dir

    def __exit__(self, *exc):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)


# --------------------------------------------------------------------- store


def test_store_roundtrip():
    with _Tmp() as tmp:
        store = MemoryStore(_tmp_db(tmp))
        vec = Embedder.lexical_fallback("кот барсик")
        eid = store.add_episode("у меня умер кот барсик", vec, importance=0.9, mood=-0.6)
        assert eid > 0
        rows = store.all_episodes()
        assert len(rows) == 1
        assert rows[0]["importance"] == 0.9
        assert rows[0]["mood"] == -0.6
        # Вектор восстановился бит-в-бит и той же размерности.
        from aizel.memory.store import _blob_to_vec

        restored = _blob_to_vec(rows[0]["vec"], int(rows[0]["dim"]))
        assert restored is not None and len(restored) == len(vec)
        store.close()


def test_core_memory_replaces_slot():
    with _Tmp() as tmp:
        store = MemoryStore(_tmp_db(tmp))
        store.set_core("about_you", ["имя: Алекс", "хобби: астрономия"])
        assert store.get_core("about_you") == ["имя: Алекс", "хобби: астрономия"]
        # Повторная запись заменяет, а не дублирует.
        store.set_core("about_you", ["имя: Алекс"])
        assert store.get_core("about_you") == ["имя: Алекс"]
        store.close()


def test_messages_and_stats():
    with _Tmp() as tmp:
        store = MemoryStore(_tmp_db(tmp))
        store.add_message("user", "привет", session="s1")
        store.add_message("assistant", "о, привет", session="s1")
        recent = store.recent_messages(limit=10, session="s1")
        assert [m["role"] for m in recent] == ["user", "assistant"]
        st = store.stats()
        assert st["messages"] == 2
        store.close()


# ------------------------------------------------------------------ retrieval


def test_retrieval_prefers_related():
    """Запрос со словами про звёзды должен вытащить астрономию, а не работу."""
    with _Tmp() as tmp:
        mem = Memory(tmp, force_lexical=True)
        mem.remember_turn("я люблю астрономию и звёзды", "расскажи", importance=0.8)
        mem.remember_turn("начал новую работу в офисе", "понятно", importance=0.6)
        hits = mem.recall("расскажи, что я люблю смотреть в звёзды")
        assert hits, "ничего не нашлось"
        assert "астроном" in hits[0].text.lower(), hits[0].text
        mem.close()


def test_lexical_mode_is_honest_about_limits():
    """Лексический режим НЕ понимает смысл — это зафиксированное ограничение.

    Настоящую семантику даёт BGE-M3; фолбэк существует, чтобы чат жил без неё.
    Если однажды этот тест «починят» и шум начнёт находиться — значит порог
    min_similarity опустили слишком сильно и в контекст полезет мусор.
    """
    with _Tmp() as tmp:
        mem = Memory(tmp, force_lexical=True)
        mem.remember_turn("я люблю астрономию", "класс", importance=0.9)
        assert mem.recall("какие у меня хобби") == []
        mem.close()


def test_retrieval_respects_importance():
    """При равной похожеści важное должно быть выше."""
    with _Tmp() as tmp:
        store = MemoryStore(_tmp_db(tmp))
        text = "разговор про погоду"
        store.add_episode(text, Embedder.lexical_fallback(text), importance=0.2)
        store.add_episode(text, Embedder.lexical_fallback(text), importance=0.95)
        r = Retriever(store, RetrievalSettings(min_similarity=0.0, min_score=0.0))
        hits = r.search(Embedder.lexical_fallback("погода"))
        assert len(hits) == 2
        assert hits[0].importance == 0.95
        store.close()


def test_retrieval_recency_matters():
    """Свежее воспоминание бьёт старое при равных прочих."""
    with _Tmp() as tmp:
        store = MemoryStore(_tmp_db(tmp))
        text = "гулял в парке"
        vec = Embedder.lexical_fallback(text)
        now = time.time()
        # Прямая вставка с разным ts: свежий и годовалой давности.
        for age_days, imp in ((0.0, 0.5), (400.0, 0.5)):
            store.db.execute(
                """INSERT INTO episodes(ts,text,vec,dim,importance,mood,topic,session)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (now - age_days * 86400, text, vec.tobytes(), len(vec), imp, 0.0, "", ""),
            )
        store.db.commit()
        r = Retriever(store, RetrievalSettings(min_similarity=0.0, min_score=0.0))
        hits = r.search(vec)
        assert hits[0].age_days < 1.0
        store.close()


def test_mood_resonance_changes_ranking():
    """В плохом настроении вспоминается грустное — это ядро идеи «живости»."""
    with _Tmp() as tmp:
        store = MemoryStore(_tmp_db(tmp))
        happy = "поездка на море была отличная"
        sad = "поездка на море закончилась ссорой"
        store.add_episode(happy, Embedder.lexical_fallback(happy), importance=0.5, mood=0.7)
        store.add_episode(sad, Embedder.lexical_fallback(sad), importance=0.5, mood=-0.7)
        r = Retriever(store, RetrievalSettings(min_similarity=0.0, min_score=0.0))
        q = Embedder.lexical_fallback("поездка на море")

        top_when_sad = r.search(q, mood=-0.8)[0].text
        top_when_happy = r.search(q, mood=0.8)[0].text
        assert top_when_sad == sad, top_when_sad
        assert top_when_happy == happy, top_when_happy
        store.close()


def test_recall_strengthens_memory():
    with _Tmp() as tmp:
        mem = Memory(tmp, force_lexical=True)
        mem.remember_turn("расскажи про звёзды", "вот", importance=0.5)
        first = mem.recall("звёзды")[0].score
        mem.recall("звёзды")
        mem.recall("звёзды")
        again = mem.recall("звёзды")[0].score
        assert again > first, "частое воспоминание должно укрепляться"
        mem.close()


def test_min_similarity_filters_noise():
    """Совсем не похожее не должно попадать в контекст."""
    with _Tmp() as tmp:
        mem = Memory(tmp, force_lexical=True)
        mem.remember_turn("купил хлеб в магазине", "ясно", importance=0.5)
        hits = mem.recall("квантовая запутанность фотонов")
        assert hits == [], f"шум просочился: {hits}"
        mem.close()


# ------------------------------------------------------------------ importance


def test_importance_heuristic():
    assert _guess_importance("меня зовут Алекс") >= 0.8
    assert _guess_importance("у меня умер кот") >= 0.8
    assert _guess_importance("привет") <= 0.3
    assert 0.3 < _guess_importance("сегодня гулял и думал о разном интересном") < 0.8


def test_human_age_labels():
    assert _human_age(0.01) == "только что"
    assert "ч назад" in _human_age(0.2)
    assert _human_age(1.5) == "вчера"
    assert "нед назад" in _human_age(14)
    assert "мес назад" in _human_age(70)


def test_context_lines_are_labeled():
    with _Tmp() as tmp:
        mem = Memory(tmp, force_lexical=True)
        mem.remember_turn("меня зовут Алекс, я люблю астрономию", "класс", importance=0.9)
        lines = mem.context_lines("как меня зовут и что я люблю")
        assert lines, "пустой контекст"
        assert "(" in lines[0] and ")" in lines[0], lines[0]
        mem.close()


# ---------------------------------------------------------- regressions D1-D4


def test_quick_core_extracts_name_from_long_message():
    text = (
        "Привет! Давно хотел рассказать о проекте. "
        "Я Даниил, создал этот проект и сейчас работаю над ним. "
        + "Мне нравится обсуждать технологии. " * 20
    )
    facts = quick_core(text)
    names = [fact.text for fact in facts if fact.slot == "имя"]
    assert names == ["Даниил"], names


def test_quick_core_does_not_capture_aizel_as_user_name():
    facts = quick_core("А тебя зовут Айзель, верно?")
    assert not [fact for fact in facts if fact.slot == "имя"], facts


def test_bare_question_is_not_saved_as_episode():
    with _Tmp() as tmp:
        mem = Memory(tmp, force_lexical=True)
        added = mem.remember_turn("Как меня зовут?", "Не знаю.")
        assert added == 0
        assert mem.store.stats()["episodes"] == 0
        mem.close()


def test_personal_statement_inside_question_is_saved():
    with _Tmp() as tmp:
        mem = Memory(tmp, force_lexical=True)
        added = mem.remember_turn("Я люблю звёзды, знаешь?", "Да, помню.")
        assert added > 0
        assert mem.store.stats()["episodes"] == 1
        mem.close()


def test_boot_prompt_distinguishes_empty_memory_and_existing_episodes():
    builder = ContextBuilder(Soul(), Personality(), Emotions())
    empty_prompt = builder.system_prompt()
    assert "Не угадывай его имя" in empty_prompt
    builder.memories = ["вчера обсуждали астрономию"]
    episode_prompt = builder.system_prompt()
    assert "есть воспоминания из прошлых разговоров" in episode_prompt
    assert "вчера обсуждали астрономию" in episode_prompt


def test_retrieval_deduplicates_and_limits_each_session():
    with _Tmp() as tmp:
        store = MemoryStore(_tmp_db(tmp))
        query = "обсуждали проект и планы на выходные"
        vec = Embedder.lexical_fallback(query)
        # Точный дубль с другой пунктуацией удаляется после нормализации.
        store.add_episode(query, vec, importance=0.9, session="session-a")
        store.add_episode(query + "!!!", vec, importance=0.8, session="session-b")
        # Один сеанс не может занять больше двух мест в контексте.
        for suffix in ("первый вариант", "второй вариант", "третий вариант"):
            store.add_episode(
                query + " " + suffix,
                vec,
                importance=0.7,
                session="session-a",
            )
        for suffix in ("четвёртый вариант", "пятый вариант", "шестой вариант"):
            store.add_episode(
                query + " " + suffix,
                vec,
                importance=0.6,
                session="session-b",
            )
        retriever = Retriever(
            store,
            RetrievalSettings(min_similarity=0.0, min_score=0.0, top_k=10),
        )
        hits = retriever.search(vec, top_k=10)
        assert len(hits) <= 4, [(h.session, h.text) for h in hits]
        assert sum(h.session == "session-a" for h in hits) <= 2
        assert sum(h.session == "session-b" for h in hits) <= 2
        normalized = [h.text.casefold().rstrip("!") for h in hits]
        assert len(normalized) == len(set(normalized)), normalized
        store.close()


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
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  ERR  {name}: {type(exc).__name__}: {exc}")
    print("\n", "ВСЁ ЗЕЛЁНОЕ" if not failed else f"провалено: {failed}")
    sys.exit(1 if failed else 0)
