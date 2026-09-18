import requests

# 1. Отправляем фото и номер рейса
url = "http://127.0.0.1:8000/api/v1/predict"
with open("data/test/images/img_00b135ede40393ac4459.jpg", "rb") as f:
    resp = requests.post(url, data={"shipment_id": "TRUCK_777"}, files={"file": f})

print("Ответ при отправке фото (XML):")
print(resp.text)

# 2. Запрашиваем результат по номеру рейса
get_url = "http://127.0.0.1:8000/api/v1/shipment/TRUCK_777"
resp_get = requests.get(get_url)
print("\nОтвет при поиске по перевозке (XML):")
print(resp_get.text)