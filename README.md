# CSI Motion and Distance Detection

Система для обнаружения движения и оценки расстояния по CSI с трёх приёмников. В репозитории есть Python-инференс трёх моделей, C++ препроцессор для realtime и утилиты для сбора данных по serial.

## Оглавление

- [Возможности](#возможности)
- [Структура репозитория](#структура-репозитория)
- [Веса моделей](#веса-моделей)
- [Требования](#требования)
- [Установка](#установка)
- [Сборка C++ препроцессора](#сборка-c-препроцессора)
- [Запуск](#запуск)
- [Realtime: сбор + препроцессинг + инференс](#realtime-сбор--препроцессинг--инференс)
- [Данные и форматы](#данные-и-форматы)
- [Препроцессинг и обучающий пайплайн](#препроцессинг-и-обучающий-пайплайн)
- [Модели](#модели)
- [Эксперименты и отчётность](#эксперименты-и-отчётность)
- [Быстрая проверка перед пушем](#быстрая-проверка-перед-пушем)
- [Частые проблемы](#частые-проблемы)

## Возможности

- Инференс трёх моделей: CNN+LSTM+Attention, Mamba-hybrid, Transformer fusion.
- C++ препроцессор `cpp_preprocessor/build/preprocessor` для конвертации трёх файлов `dev*.data` в вектор амплитуд.
- Утилиты serial-сбора в `tools/` и единый скрипт `run_realtime.sh`.
- Отчётность и визуализации в `notebooks/` (см. `notebooks/README.md` и `notebooks/figures/`).

## Структура репозитория

```
app/
  app.py
  load_model.py
  data_load/
  models/
cpp_preprocessor/
  CMakeLists.txt
  *.cpp
  *.hpp
tools/
  receiver.py
  receiver_split.py
  watcher.py
  list_ports.py
tests/
  simulate.py
data/
  samples/
weights/
  best_cnn_lstm.pth
  best_mamba_model.pth
  best_fusion_csi_model.pth
  train_mean.npy
  train_std.npy
notebooks/
  README.md
  figures/
run_realtime.sh
main.py
requirements.txt
```

## Веса моделей

Артефакты для инференса лежат в `weights/` и соответствуют коду в `app/load_model.py`.

Скачать полный комплект весов и статистик можно здесь: [Google Drive: model weights](https://drive.google.com/drive/folders/1QfCjzMwqpNUO60O4QAo-yu5TQHhVOlhW?usp=sharing).

Ожидаемые файлы:

- `weights/best_cnn_lstm.pth`
- `weights/best_mamba_model.pth`
- `weights/best_fusion_csi_model.pth`
- `weights/train_mean.npy`
- `weights/train_std.npy`

## Требования

- Python 3.10+ (рекомендуется виртуальное окружение `venv/`)
- PyTorch (см. `requirements.txt`)
- `pyserial` для serial-сбора
- CMake + компилятор C++17 для сборки `cpp_preprocessor`

## Установка

```bash
python3 -m venv venv
./venv/bin/python -m pip install -U pip
./venv/bin/python -m pip install -r requirements.txt
```

## Сборка C++ препроцессора

```bash
cmake -S cpp_preprocessor -B cpp_preprocessor/build
cmake --build cpp_preprocessor/build -j
```

Артефакты:

- `cpp_preprocessor/build/preprocessor`
- `cpp_preprocessor/build/csi_data_generator`

## Запуск

Общий вход: `main.py`.

### Симуляция по готовому CSV

```bash
./venv/bin/python main.py simulate data/samples/wifi_data_set_after_preprocessing_person_id_3.csv 20 0
```

### Мониторинг папок + внешний бинарь препроцессинга

```bash
./venv/bin/python main.py monitor ../data/esp 100 ./path/to/preprocess.elf
```

### Serial-сбор (3 порта)

`baud` — скорость UART в бит/с (часто `921600`).

```bash
./venv/bin/python main.py receiver --output data/esp --window 100 --overlap 20 --ports /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 --baud 921600
```

### Watcher: препроцессинг + инференс + очистка папок

После успешного инференса папка `test_*` удаляется, чтобы не копить данные.

```bash
./venv/bin/python main.py watcher --watch-dir data/esp --preprocessor ./cpp_preprocessor/build/preprocessor
```

## Realtime: сбор + препроцессинг + инференс

```bash
chmod +x run_realtime.sh
./run_realtime.sh data/esp 100 20 /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 921600
```

## Данные и форматы

- Для обучения/симуляции: строки CSV вида `id:[[...],[...],[...]]:label` (см. `app/data_load/parse_data.py`).
- Для realtime: `receiver` пишет сегменты в `test_N/dev1.data dev2.data dev3.data`, далее `preprocessor` печатает JSON-подобный список из трёх массивов амплитуд.

## Препроцессинг и обучающий пайплайн

Ниже описан пайплайн в терминах данных и кода репозитория. Он совпадает с твоей исходной постановкой: медианная фильтрация, аугментации на этапе обучения и честная валидация по людям. Дифференциальные признаки используются внутри Transformer-модели (см. раздел про Transformer), а не как отдельный обязательный шаг для CNN/Mamba в этом inference-коде.

### Сбор и окно наблюдения

- CSI собирается с трёх приёмников. В realtime `tools/receiver.py` накапливает синхронные сегменты длиной `window` строк на каждое устройство и сохраняет их в `test_*/dev{1..3}.data`.
- Для обучения типичное окно соответствует 100 временным отсчётам, 52 поднесущим и 3 приёмникам, то есть тензор формы `(100, 52, 3)`.

### Из сырого CSI к амплитудам

- C++ препроцессор читает CSV-строки `dev*.data`, извлекает массив комплексных отсчётов в виде чередования Re/Im, переводит его в амплитуды по поднесущим и применяет медианный фильтр по времени для каждой поднесущей.
- На выходе для каждого устройства получается уплощенный вектор длины `100 * 52 = 5200`, а суммарно три устройства дают три таких массива. В Python-инференсе это собирается обратно в `(100, 52, 3)`.

### Обучающий датасет и разметка

- Класс расстояния кодируется как `0..3` метра. Для multitask-моделей голова движения использует бинаризацию: `motion = 1`, если `label > 0`, иначе `0`.
- Для честной оценки обобщения используется схема Leave-One-Person-Out: обучение на трёх людях и тест на четвёртом, затем усреднение по фолдам.

### Аугментации (обучение)

На этапе обучения (вне этого inference-репозитория) использовались типовые аугментации для временных CSI-рядов: аддитивный гауссов шум, небольшие временные сдвиги, маскирование части поднесущих. В инференсе аугментации не применяются.

## Модели

### CNN+LSTM+Attention (`app/models/cnn_lstm.py`)

Архитектура заточена под локальные временные паттерны в амплитудах:

- Вход: `(B, 100, 52, 3)`.
- Две свёртки `Conv1d` по времени с `BatchNorm`, `ReLU` и `MaxPool`, уменьшают временное разрешение.
- Затем `BiLSTM` моделирует долгую временную зависимость.
- `TemporalAttention` агрегирует временную ось в компактный вектор признаков.
- Две головы: `motion` (2 класса) и `distance` (4 класса).

### Mamba-hybrid (`app/models/mamba_s6.py`)

Гибридная модель для длинных рядов с относительно компактным числом параметров:

- Вход: `(B, 100, 52, 3)`.
- Лёгкая входная свёртка + пулинг проецируют сигнал в пространство `d_model`.
- Блоки чередуют depthwise-конволюцию и селективное state-space ядро (Mamba-ветка).
- Две проекции и cross-attention между представлениями для motion/distance, затем усреднение по времени.
- Две головы: `motion` (2 класса) и `distance` (4 класса).

### Transformer fusion (`app/models/transformer.py`)

Модель другого типа: внутри используются локальные CNN+Transformer ветки и глобальное слияние:

- Вход для ветки иерархии по датчикам: `(B, 3, 100, 52)`.
- Строятся дифференциальные CSI-признаки по времени (первая и вторая разность амплитуд), затем локальный CNN+Transformer энкодер на окне, далее глобальный Transformer по трём датчикам.
- Вторая ветка объединяет датчики вдоль времени и прогоняет Transformer по времени.
- Итоговые логиты — взвешенная смесь двух веток; классовость: 4 класса расстояния.
- Нормализация входа для этой модели делается отдельно через `train_mean.npy` и `train_std.npy` (см. `app/load_model.py`).

## Эксперименты и отчётность

Таблицы метрик, картинки и краткий «человеческий» разбор экспериментов лежат в `notebooks/README.md`.

Графики и схемы лежат в `notebooks/figures/` (в репозитории уже есть несколько PNG/JPG для CNN/Mamba: confusion matrix и train curves, плюс схемы архитектур).

## Быстрая проверка перед пушем

```bash
./venv/bin/python -m py_compile main.py tools/receiver.py tools/watcher.py
./venv/bin/python main.py simulate data/samples/wifi_data_set_after_preprocessing_person_id_3.csv 10 0
cmake -S cpp_preprocessor -B cpp_preprocessor/build
cmake --build cpp_preprocessor/build -j
```

## Частые проблемы

- Если IDE ругается на `serial`, проверь, что выбран интерпретатор `./venv/bin/python`, и что `pyserial` установлен (`requirements.txt`).
- Если `receiver` не видит порты, сначала посмотри список устройств: `./venv/bin/python tools/list_ports.py`.
