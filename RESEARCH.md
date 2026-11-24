# Источники и лицензии

## Итерация 1. Корпус, лексический поиск и выбор модели

| Источник | Что изучили | Лицензия |
| --- | --- | --- |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn), [документация TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) | Сглаженный IDF, нормализация и косинусное сравнение. | BSD-3-Clause |
| [BEIR](https://github.com/beir-cellar/beir) | Схема «корпус, запросы, qrels» для эксперимента с общими входами. Корпусы BEIR не загружались. | Apache-2.0 для кода. Данные имеют собственные лицензии. |
| [all-MiniLM-L6-v2, закреплённая версия](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/855bcf54d21334332f75db10bdd48970ff0ef547) | Англоязычная 384-мерная модель. Квантованный ONNX для AVX2 весит около 23 MB. Основной кандидат для CPU. | Apache-2.0 |
| [Model2Vec](https://github.com/MinishLab/model2vec), [Potion 4M, закреплённая версия](https://huggingface.co/minishlab/potion-base-4M/tree/9b3cff412d30be9ae8603fe10224c224f3401869) | Статические эмбеддинги, 15 118 424 байта весов. Запасной кандидат при технической блокировке MiniLM. | MIT для кода и модели. |

Researcher подтвердил доступ к GitHub. Корпус и разметку создаём в проекте. Рекомендация по модели опиралась на размер, CPU-совместимость и опубликованные авторами модели оценки retrieval. Собственные результаты хранятся в `BENCHMARKS.csv`.

- Исходные определения метрик сверены с [trec_eval](https://github.com/usnistgov/trec_eval) (см. условия в каждом файле). Реализация в проекте самостоятельная. [Precision](https://github.com/usnistgov/trec_eval/blob/main/m_P.c), [Recall](https://github.com/usnistgov/trec_eval/blob/main/m_recall.c), [Reciprocal Rank](https://github.com/usnistgov/trec_eval/blob/main/m_recip_rank.c)
- Researcher рассмотрел SciFact как небольшой внешний корпус. Разметка распространяется по CC BY 4.0, аннотации статей из S2ORC по ODC-By 1.0, код по Apache-2.0. SciFact в эксперименты не включён. [Условия SciFact](https://github.com/allenai/scifact/blob/master/LICENSE.md), [статья](https://aclanthology.org/2020.emnlp-main.609/)
- Researcher скачал обе модели в диагностический временный каталог. Размер файла ONNX MiniLM AVX2 составил 23 046 789 байт. Хеш весов Potion 4M совпал с указанным в хранилище модели.
- Researcher провёл пробный инференс MiniLM через CPUExecutionProvider в изолированном окружении с Python 3.12. Проверены маскированное усреднение и 384-мерный вектор. [ONNX Runtime, MIT](https://github.com/microsoft/onnxruntime), [Tokenizers, Apache-2.0](https://github.com/huggingface/tokenizers/blob/main/LICENSE)
