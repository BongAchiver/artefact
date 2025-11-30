Я проставлял сид везде, где только видел, но погрешность на 1-2 тысячных скора все равно остается. Что бы воспроизвести запуск надо:
P.s. Вся работа была в conda окружении с Python 3.11.14

**Linux / macOS / WSL (Windows)**

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (PowerShell)**

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

```bash
poetry install
```

!!! Я не до конца разобрался с poetry, по этому если из него не установился CatBoost и Optuna, надо вручную поставить !!!

```bash
pip install catboost optuna "optuna[catboost]"
```

```bash
# 1. Подготовка данных (загрузка, фильтрация, feature engineering)
poetry run python -m src.baseline.prepare_data

# 2. Обучение модели (использует подготовленные данные)
poetry run python -m src.baseline.catboost_train

# 3. Предсказание (использует подготовленные данные и обученные модели)
poetry run python -m src.baseline.predict

# Я не использовал валидацию, но на всякий случай оставлю
poetry run python -m src.baseline.validate
```

:3