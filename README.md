# Semantic Search Lab

Локальный семантический поиск по документам на Python и эксперимент по сравнению методов ранжирования. Репозиторий включает англоязычный корпус, размеченные запросы и инструменты оценки. Для лексического поиска достаточно Python 3.10 или новее.

## Быстрый запуск

Из корня репозитория:

```bash
python -m semantic_search search --method tfidf --top-k 5 "how to prevent soil erosion"
python -m semantic_search search --method bm25 --top-k 5 "how to prevent soil erosion"
```

Параметр `--json` возвращает список объектов с полями `id`, `title` и `score`. Чтобы искать по собственному корпусу, передайте путь через `--corpus documents.jsonl`. Каждая строка файла содержит один JSON-объект в UTF-8, например:

```json
{"id":"soil-001","title":"Cover crops","text":"Cover crops protect soil against erosion during heavy rain."}
```

Идентификаторы документов уникальны. Поиск индексирует заголовок и текст. Оценки разных методов имеют разные шкалы. В BM25 можно задать `--k1` для насыщения частоты терма и `--b` для нормализации длины.

## Семантический поиск на CPU

Для MiniLM нужен CPU x86-64 с AVX2. Конфигурация проверена на Linux с Python 3.12. Установите дополнительные зависимости и один раз загрузите закреплённые файлы модели (около 24 МБ):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[semantic]'
python -m semantic_search prepare-model
python -m semantic_search search --method semantic --top-k 5 "how to prevent soil erosion"
```

Поиск читает файлы из пользовательского кэша и выполняется локально. Можно указать свой каталог одинаковым флагом `--model-dir` для подготовки и поиска. Для отдельного кэша эксперимента можно задать, например, `export XDG_CACHE_HOME="$PWD/.venv/cache"` перед подготовкой и сохранить это значение для последующих замеров. По умолчанию MiniLM обрабатывает до 256 токенов из объединённого заголовка и текста.

## Данные

Фиксированный корпус из 60 оригинальных англоязычных документов и 30 запросов хранится в `semantic_search/data/`. Файл запросов определяет `id`, `text`, `split` и `relevant_ids`. Десять запросов `dev` используются при выборе настроек ранжирования. Двадцать запросов `test` используются для итогового сравнения. Запросы включают прямые формулировки, перефразы и случаи с близкими по теме отвлекающими документами. Все методы получают одинаковые документы и запросы.

## Проверка и бенчмарк

Из корня репозитория:

```bash
python -m unittest discover -s tests -v
python -m tools.benchmark --split dev --method tfidf
```

Для тестов с реальным CPU-инференсом после установки дополнительных зависимостей и подготовки кэша:

```bash
SEMANTIC_TEST_MODEL_DIR="$(python -m semantic_search prepare-model)" python -m unittest discover -s tests -v
```

Бенчмарк рассчитан на Linux. Он печатает CSV с Precision@10, Recall@10, MRR, средним временем поиска и пиковым RSS.

[Сравнительный отчёт](REPORT.md) · [Источники и лицензии](RESEARCH.md) · [Журнал взаимодействия агентов](AGENT_LOG.md) · [Контрольные точки](PROGRESS.md)
