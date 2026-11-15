# Классификатор способов передвижения по датчикам Android

> Данный репозиторий является копией моего другого на DagsHub
>
> Ссылка: [https://dagshub.com/FiveStarHotel/MLP_sensors_classifier](https://dagshub.com/FiveStarHotel/MLP_sensors_classifier)

Репозиторий финального проекта с ноутбуками исследований. В нем же доступен переход в MLflow для отслеживания запусков и параметров.

Модель реализована на TensorFlow. Бэкенд для инференса написан на FastAPI.

## Как запустить

Выполните в терминале:

```bash
git clone https://dagshub.com/FiveStarHotel/MLP_sensors_classifier.git
cd MLP_sensors_classifier
uv sync
uv run uvicorn backend.app:app
```
