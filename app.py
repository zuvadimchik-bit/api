import io
import sqlite3
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile, Response
from PIL import Image
import torch
import dicttoxml
from model import CargoPredictor, get_transforms

app = FastAPI(title="Russian Post Cargo API")

# База данных для сохранения истории перевозок
conn = sqlite3.connect("shipments.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS shipments (
    shipment_id TEXT PRIMARY KEY,
    load_pct REAL,
    updated_at TEXT
)
""")
conn.commit()

# Загрузка обученной модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CargoPredictor()
try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("Модель best_model.pth успешно загружена!")
except Exception:
    print("Внимание: best_model.pth не найден, используются базовые веса")
model.to(device)
model.eval()

transform = get_transforms(is_train=False)

def xml_response(data: dict, status_code: int = 200):
    xml_str = dicttoxml.dicttoxml(data, custom_root="response", attr_type=False).decode("utf-8")
    return Response(content=xml_str, media_type="application/xml", status_code=status_code)

@app.post("/api/v1/predict")
async def predict_load(shipment_id: str = Form(...), file: UploadFile = File(...)):
    """1. Приём фото и номера перевозки, расчёт и сохранение"""
    if not shipment_id.strip():
        return xml_response({"status": "error", "message": "Номер перевозки не может быть пустым"}, status_code=400)

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return xml_response({"status": "error", "message": "Некорректный или повреждённый файл изображения"}, status_code=422)

    try:
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor).item()
        load_pct = round(max(0.0, min(100.0, float(pred))), 1)

        # Сохранение / обновление при повторном обращении
        now = datetime.utcnow().isoformat()
        cursor.execute(
            "INSERT OR REPLACE INTO shipments (shipment_id, load_pct, updated_at) VALUES (?, ?, ?)",
            (shipment_id, load_pct, now)
        )
        conn.commit()

        return xml_response({
            "status": "success",
            "shipment_id": shipment_id,
            "load_pct": load_pct,
            "timestamp": now
        })
    except Exception as e:
        return xml_response({"status": "error", "message": f"Ошибка сервера: {str(e)}"}, status_code=500)

@app.get("/api/v1/shipment/{shipment_id}")
async def get_shipment(shipment_id: str):
    """2. Получение сохранённой оценки по номеру перевозки"""
    cursor.execute("SELECT shipment_id, load_pct, updated_at FROM shipments WHERE shipment_id = ?", (shipment_id,))
    row = cursor.fetchone()
    if not row:
        return xml_response({"status": "error", "message": f"Перевозка {shipment_id} не найдена"}, status_code=404)

    return xml_response({
        "status": "success",
        "shipment_id": row[0],
        "load_pct": row[1],
        "timestamp": row[2]
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)