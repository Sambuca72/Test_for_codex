# Semantic Search Lab

Локальный семантический поиск по документам на Python и эксперимент по сравнению методов ранжирования. Репозиторий включает англоязычный корпус, размеченные запросы и инструменты оценки. Для лексического поиска достаточно Python 3.10 или новее.

## Быстрый запуск

Выполните команды из корня репозитория.

```bash
python -m semantic_search search --method tfidf --top-k 5 "how to prevent soil erosion"
python -m semantic_search search --method bm25 --top-k 5 "how to prevent soil erosion"
```

Параметр `--json` возвращает список объектов с полями `id`, `title` и `score`. Чтобы искать по собственному корпусу, передайте путь через `--corpus documents.jsonl`. Каждая строка файла содержит один JSON-объект в UTF-8.

```json
{"id":"soil-001","title":"Cover crops","text":"Cover crops protect soil against erosion during heavy rain."}
```

Идентификаторы документов уникальны. Поиск индексирует заголовок и текст. Оценки разных методов имеют разные шкалы. В BM25 можно задать `--k1` для насыщения частоты терма и `--b` для нормализации длины.

## Семантический поиск на CPU

Для MiniLM нужен CPU x86-64 с AVX2. Конфигурация проверена на Linux с Python 3.12. Установите дополнительные зависимости и один раз загрузите закреплённые файлы модели (около 24 МБ).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[semantic]'
python -m semantic_search prepare-model
python -m semantic_search search --method semantic --top-k 5 "how to prevent soil erosion"
```

Поиск читает файлы из пользовательского кэша и выполняется локально. Можно указать свой каталог одинаковым флагом `--model-dir` для подготовки и поиска. Для отдельного кэша эксперимента можно задать, например, `export XDG_CACHE_HOME="$PWD/.venv/cache"` перед подготовкой и сохранить это значение для последующих замеров. По умолчанию MiniLM обрабатывает до 256 токенов из объединённого заголовка и текста.

Гибридный режим использует тот же кэш. Он объединяет ранги BM25 и MiniLM. Параметр `--lexical-weight` задаёт долю BM25, `--rrf-k` управляет затуханием вклада с рангом.

```bash
python -m semantic_search search --method hybrid --k1 0.6 --b 0.75 --top-k 5 "how to prevent soil erosion"
```

## Данные

Фиксированный корпус из 60 оригинальных англоязычных документов и 30 запросов хранится в `semantic_search/data/`. Файл запросов определяет `id`, `text`, `split` и `relevant_ids`. Десять запросов `dev` используются при выборе настроек ранжирования. Двадцать запросов `test` используются для итогового сравнения. Запросы включают прямые формулировки, перефразы и случаи с близкими по теме отвлекающими документами. Все методы получают одинаковые документы и запросы.

## Проверка и бенчмарк

Выполните команды из корня репозитория.

```bash
python -m unittest discover -s tests -v
python -m tools.benchmark --split dev --method tfidf
```

После подготовки кэша и установки зависимостей можно запустить интеграционные тесты с реальным CPU-инференсом.

```bash
SEMANTIC_TEST_MODEL_DIR="$(python -m semantic_search prepare-model)" python -m unittest discover -s tests -v
```

Бенчмарк рассчитан на Linux. Он печатает CSV с Precision@10, Recall@10, MRR, средним временем поиска и пиковым RSS. Для воспроизведения итогового сравнения используйте CPU-модель и параметры, выбранные на dev.

```bash
python -m tools.benchmark --split test --warmup 1 --repeat 20 \
  --method tfidf \
  --run '{"method":"bm25","options":{"k1":0.6,"b":0.75}}' \
  --run '{"method":"semantic"}' \
  --run '{"method":"hybrid","options":{"k1":0.6,"b":0.75,"lexical_weight":0.25,"rrf_k":10}}'
```

Итоговый прогон использовал Python 3.12.14, NumPy 2.3.5, ONNX Runtime 1.32.0 и Tokenizers 0.23.3. Для совпадения версий можно указать `numpy==2.3.5 onnxruntime==1.32.0 tokenizers==0.23.3` при установке. Для контрольного прогона замените `--split test` на `--split dev`. Для повторения сеток используйте значения из [плана](PLAN.md) и передайте каждый вариант отдельным аргументом `--run`.

[Сравнительный отчёт](REPORT.md) · [CSV замеров](BENCHMARKS.csv) · [Источники и лицензии](RESEARCH.md) · [Журнал взаимодействия агентов](AGENT_LOG.md) · [Контрольные точки](PROGRESS.md)
