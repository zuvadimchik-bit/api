Оценка загрузки грузового транспорта (Почта России)



MVP сервиса для автоматической оценки процента заполнения кузова автомобиля по фотографии с сохранением результатов в базе данных и доступом через XML API.



1. Структура проекта




model.py — архитектура нейросети (EfficientNet-B0) и пайплайн обработки изображений.


train.py — воспроизводимый скрипт обучения с валидацией GroupKFold.


make_submission.py — скрипт генерации файла предсказаний submission.csv для тестовой выборки.


app.py — сервис XML API на базе FastAPI и SQLite.


best_model.pth — обученные веса модели.




2. Установка зависимостей


pip install torch torchvision timm pandas numpy scikit-learn pillow fastapi uvicorn python-multipart dicttoxml requests



3. Обучение и воспроизводимость



Запуск цикла обучения с валидацией по группам визуального сходства (GroupKFold):



python train.py



4. Формирование предсказаний (submission.csv)



Генерация итогового файла для платформы:



python make_submission.py



5. Документация XML API


Запуск сервиса


uvicorn app:app --host 0.0.0.0 --port 8000



Методы API:


1. Расчёт и сохранение загрузки




Метод: POST /api/v1/predict


Формат запроса: multipart/form-data


Параметры:



shipment_id (строка) — идентификатор перевозки / номер рейса.


file (файл) — фотография кузова (JPEG/PNG).






Поведение при повторном обращении: при передаче уже существующего shipment_id запись в базе данных обновляется новыми актуальными данными.


Пример ответа (XML):




<?xml version="1.0" encoding="UTF-8" ?>
<response>
    <status>success</status>
    <shipment_id>TRUCK_777</shipment_id>
    <load_pct>100.0</load_pct>
    <timestamp>2026-09-16T16:02:32.441006</timestamp>
</response>



2. Получение оценки по номеру перевозки




Метод: GET /api/v1/shipment/{shipment_id}


Пример ответа (XML):




<?xml version="1.0" encoding="UTF-8" ?>
<response>
    <status>success</status>
    <shipment_id>TRUCK_777</shipment_id>
    <load_pct>100.0</load_pct>
    <timestamp>2026-09-16T16:02:32.441006</timestamp>
</response>



3. Обработка ошибок



В случае передачи повреждённого или неподдерживаемого файла сервис возвращает понятный XML-ответ с кодом ошибки:



<?xml version="1.0" encoding="UTF-8" ?>
<response>
    <status>error</status>
    <message>Некорректный или повреждённый файл изображения</message>
</response>

Проверь README на ошибки
Добавь раздел Запуск локально
Уточни формат submission.csv
Добавь раздел Примеры API