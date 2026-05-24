# Тесты для `node_replay_service`

В каталог `tests/` добавлен рабочий набор **юнит-тестов** для проверки базовой логики replay-сервисов и общего launcher-сценария.

## Что покрыто

### 1. Юнит-тесты `main_service.py`

| Файл | Проверка |
|---|---|
| `tests/test_main_service_unit.py` | нормализация списка моделей |
| `tests/test_main_service_unit.py` | раскрытие псевдонима `all` |
| `tests/test_main_service_unit.py` | ошибка при неизвестной модели |
| `tests/test_main_service_unit.py` | добавление общих replay-аргументов в команду |
| `tests/test_main_service_unit.py` | валидация обязательных аргументов для mDiSSiD |
| `tests/test_main_service_unit.py` | формирование команды DAMP с файлом порогов |

### 2. Юнит-тесты `replay_damp_service.py`

| Файл | Проверка |
|---|---|
| `tests/test_replay_damp_service_unit.py` | фильтрация списка CSV-файлов узлов |
| `tests/test_replay_damp_service_unit.py` | загрузка и очистка подготовленного CSV |
| `tests/test_replay_damp_service_unit.py` | расчет порога на синтетических данных |
| `tests/test_replay_damp_service_unit.py` | инициализация prometheus-метрик |
| `tests/test_replay_damp_service_unit.py` | обновление score/flag/position в одном replay-шаге |
| `tests/test_replay_damp_service_unit.py` | корректное завершение replay при исчерпании данных |

### 3. Юнит-тесты `replay_lstm_service.py`

| Файл | Проверка |
|---|---|
| `tests/test_replay_lstm_service_unit.py` | расчет порога по combined-score |
| `tests/test_replay_lstm_service_unit.py` | фильтрация списка CSV-файлов |
| `tests/test_replay_lstm_service_unit.py` | обновление cpu/voltage/combined score и флага |
| `tests/test_replay_lstm_service_unit.py` | завершение replay по концу данных |

### 4. Юнит-тесты `replay_mdissid_service.py`

| Файл | Проверка |
|---|---|
| `tests/test_replay_mdissid_service_unit.py` | нормализация входного массива |
| `tests/test_replay_mdissid_service_unit.py` | формирование входов для siamese-модели |
| `tests/test_replay_mdissid_service_unit.py` | загрузка snippets из CSV |
| `tests/test_replay_mdissid_service_unit.py` | расчет порога для узла |
| `tests/test_replay_mdissid_service_unit.py` | инициализация циклических буферов состояния узла |

## Сводная таблица

| Блок | Количество тестов |
|---|---:|
| `main_service.py` | 6 |
| `replay_damp_service.py` | 6 |
| `replay_lstm_service.py` | 4 |
| `replay_mdissid_service.py` | 5 |
| **Всего юнит-тестов** | **21** |

## Как запускать

Из корня каталога `node_replay_service`:

```bash
pytest -q
```

## Фактический результат прогона

```text
21 passed in 1.83s
```

## Технические замечания по реализации тестов

1. В `tests/conftest.py` добавлены тестовые заглушки для:
   - `prometheus_client`;
   - модуля `damp`;
   - модуля `lstm_model`.

2. Это сделано специально, чтобы:
   - исключить конфликт глобального Prometheus registry при импорте нескольких сервисов;
   - не требовать реальных обученных моделей и тяжелых внешних зависимостей;
   - тестировать именно прикладную логику сервиса, а не инфраструктуру обучения.

3. Текущий набор рассчитан на быстрый локальный прогон и подходит как первый обязательный слой регрессии перед добавлением:
   - функциональных тестов;
   - нефункциональных тестов;
   - SLA-проверок на синтетических аномалиях.
