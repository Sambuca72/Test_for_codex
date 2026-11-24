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

## Итерация 2. BM25

| Источник | Решение для проекта | Лицензия |
| --- | --- | --- |
| [Apache Lucene BM25Similarity](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/similarities/BM25Similarity.java) | Положительный `log1p((N - df + 0.5) / (df + 0.5))`, опорные `k1=1.2`, `b=0.75`; проверен учёт повторов терма в запросе. | [Apache-2.0](https://github.com/apache/lucene/blob/main/LICENSE.txt) |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py) | Сравнили вариант с обработкой отрицательного IDF и параметрами 1.5/0.75. | [Apache-2.0](https://github.com/dorianbrown/rank_bm25/blob/master/LICENSE) |
| [BM25S](https://github.com/xhluca/bm25s/blob/main/bm25s/scoring.py), [статья](https://arxiv.org/abs/2407.03618) | Сравнили разреженный расчёт по postings и предвычисление вклада документа. | [MIT](https://github.com/xhluca/bm25s/blob/main/LICENSE) |

Researcher сопоставил варианты с [обзором Robertson и Zaragoza](https://doi.org/10.1561/1500000019) и [сравнением формул BM25](https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf). Для TF-IDF и BM25 применяем одинаковую токенизацию к запросу и объединённому полю заголовка и текста. [Stanford IR book](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)

## Итерация 3. Энкодер CPU

- В закреплённой конфигурации MiniLM длина входа ограничена 256 токенами. Модель использует uncased WordPiece, маскированное среднее и L2-нормализацию выходного 384-мерного вектора. [Длина](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/855bcf54d21334332f75db10bdd48970ff0ef547/sentence_bert_config.json), [пул](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/855bcf54d21334332f75db10bdd48970ff0ef547/1_Pooling/config.json), [модули](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/855bcf54d21334332f75db10bdd48970ff0ef547/modules.json)
- Рекомендации по CPU и квантованию сверены с документацией Sentence Transformers. Граф ONNX выдаёт состояния токенов, поэтому пул и нормализация выполняются в Python. [Документация](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html)
- Researcher повторно запустил квантованный CPU-граф на диагностических фразах. При повторении того же батча результаты совпали, при переходе от батча к одиночному входу немного менялись. Порядок документов и размер батча фиксируем в эксперименте.
- Официальные файлы успешно получены по HTTPS через `/resolve/` закреплённой ревизии. Код проверяет размер и SHA-256 до установки файла в кэш, а затем повторно проверяет артефакты при чтении. [Закреплённые файлы](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/855bcf54d21334332f75db10bdd48970ff0ef547)

## Итерация 4. Гибрид

| Источник | Что применили | Лицензия |
| --- | --- | --- |
| [Cormack, Clarke, Büttcher, SIGIR 2009](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf) | Формула Reciprocal Rank Fusion и ранги с единицы. В статье использована константа 60. В эксперименте также проверяем 10 для короткого корпуса. | Авторское право ACM на статью. |
| [Pyserini](https://github.com/castorini/pyserini/blob/master/pyserini/fusion/_base.py), [пример BM25 + DPR](https://github.com/castorini/pyserini/blob/master/docs/experiments-wiki-corpora.md) | Показано объединение лексического и плотного поиска и различие между глубиной компонентной выдачи и размером итогового top-k. | [Apache-2.0](https://github.com/castorini/pyserini/blob/master/LICENSE.txt) |
| [LangChain EnsembleRetriever](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/retrievers/ensemble.py) | Сверены взвешенная формула, отсутствие вклада вне списка и сопоставление документов по стабильному идентификатору. | [MIT](https://github.com/langchain-ai/langchain/blob/master/LICENSE) |

Researcher также сверил трактовку весов и глубины выдачи с документацией Elasticsearch. В проекте глубина компонентных рейтингов одинакова для любого запрошенного `k`, поэтому top-k является префиксом полного ранжирования. [Документация RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion), [взвешенный вариант](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever)
