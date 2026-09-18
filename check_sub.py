import pandas as pd

sub = pd.read_csv("submission.csv")

assert list(sub.columns) == ["image_id", "load_pct"], "Неверные заголовки колонок!"
assert len(sub) == 307, f"Ожидалось 307 строк, а получилось {len(sub)}!"
assert sub["load_pct"].isnull().sum() == 0, "Есть пустые значения (NaN)!"
assert (sub["load_pct"] >= 0).all() and (sub["load_pct"] <= 100).all(), "Есть числа вне диапазона 0-100!"

print("Проверка пройдена успешно! Файл submission.csv идеален для загрузки.")
