import streamlit as st
import requests
import xmltodict

API_URL = "http://127.0.0.1:8000/api/v1"

# Настройка страницы
st.set_page_config(
    page_title="Контроль загрузки транспорта | Почта России",
    page_icon="📦",
    layout="wide"
)

# Заголовок сервиса
st.title("📦 Мониторинг загрузки кузова (Почта России)")
st.caption(
    "MVP сервиса компьютерного зрения для автоматической оценки заполнения грузового отсека и привязки к рейсам через XML API")

tab1, tab2 = st.tabs(["📸 Оценка и фиксация рейса", "🔍 Поиск оценки по номеру перевозки"])

# --- ВКЛАДКА 1: ОЦЕНКА И ПРИВЯЗКА К ПЕРЕВОЗКЕ ---
with tab1:
    st.subheader("Фиксация загрузки автомобиля")
    col_input, col_result = st.columns([1, 1], gap="medium")

    with col_input:
        shipment_id = st.text_input(
            "Номер перевозки / рейса (shipment_id)*",
            placeholder="Например: REYS-78142",
            help="Уникальный номер рейса или путевого листа для сохранения в базе"
        )

        uploaded_file = st.file_uploader(
            "Фотография грузового отсека (JPEG/PNG)*",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            st.image(uploaded_file, caption="Загруженное фото кузова", use_container_width=True)

    with col_result:
        st.write("### Результат анализа")

        btn_calc = st.button("Рассчитать загрузку и сохранить", type="primary", use_container_width=True)

        if btn_calc:
            if not shipment_id.strip():
                st.error("⚠️ Ошибка: номер перевозки обязателен по регламенту ТЗ!")
            elif not uploaded_file:
                st.error("⚠️ Ошибка: прикрепите фотографию кузова!")
            else:
                with st.spinner("Отправка в XML API и обработка нейросетью..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {"shipment_id": shipment_id.strip()}
                        resp = requests.post(f"{API_URL}/predict", data=data, files=files, timeout=30)

                        if resp.status_code == 200:
                            parsed = xmltodict.parse(resp.text).get("response", {})
                            load_pct = float(parsed.get("load_pct", 0.0))
                            timestamp = parsed.get("timestamp", "—")

                            st.success(f"✅ Перевозка **{shipment_id}** успешно зафиксирована в системе!")

                            # Главная метрика
                            m_col1, m_col2 = st.columns(2)
                            m_col1.metric("Заполнение кузова", f"{load_pct}%")
                            m_col2.metric("Время регистрации (UTC)", timestamp[:19].replace("T", " "))

                            # Визуальная шкала
                            st.progress(min(1.0, max(0.0, load_pct / 100.0)))

                            # Логистический статус по ТЗ (выявление недогруженных авто)
                            if load_pct < 30.0:
                                st.warning(
                                    "⚠️ **Критический недогруз!** Машина заполнена менее чем на 30%. Требуется дозагрузка или замена на транспорт меньшей вместимости.")
                            elif load_pct < 70.0:
                                st.info("ℹ️ **Частичная загрузка.** Кузов заполнен наполовину.")
                            else:
                                st.success("🎯 **Оптимальная загрузка.** Транспорт готов к отправке.")

                            # Демонстрация работы XML протокола для жюри
                            with st.expander("📄 Ответ сервера в формате XML (требование ТЗ)"):
                                st.code(resp.text, language="xml")

                        elif resp.status_code == 422:
                            parsed = xmltodict.parse(resp.text).get("response", {})
                            st.error(f"❌ Ошибка валидации изображения: {parsed.get('message', resp.text)}")
                        else:
                            st.error(f"❌ Ошибка сервера ({resp.status_code}): {resp.text}")

                    except Exception as e:
                        st.error(f"Не удалось связаться с сервером API. Убедитесь, что запущен app.py ({e})")

# --- ВКЛАДКА 2: ПОИСК ПО НОМЕРУ ПЕРЕВОЗКИ ---
with tab2:
    st.subheader("Поиск зафиксированной загрузки по номеру рейса")
    st.caption("Позволяет диспетчеру или логисту получить статус рейса без повторной отправки фотографии.")

    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        search_id = st.text_input("Введите номер перевозки для проверки", placeholder="Например: REYS-78142")
    with search_col2:
        st.write("")
        st.write("")
        btn_search = st.button("Найти в базе", use_container_width=True)

    if btn_search:
        if not search_id.strip():
            st.error("Введите номер перевозки!")
        else:
            try:
                resp = requests.get(f"{API_URL}/shipment/{search_id.strip()}")
                if resp.status_code == 200:
                    parsed = xmltodict.parse(resp.text).get("response", {})
                    load = float(parsed.get("load_pct", 0.0))
                    timestamp = parsed.get("timestamp", "—")

                    st.success(f"Рейс **{search_id}** найден в архиве базы данных.")
                    res1, res2 = st.columns(2)
                    res1.metric("Зафиксированная загрузка", f"{load}%")
                    res2.metric("Дата фиксации (UTC)", timestamp[:19].replace("T", " "))
                    st.progress(min(1.0, max(0.0, load / 100.0)))

                    with st.expander("📄 Исходный XML ответ из базы"):
                        st.code(resp.text, language="xml")
                else:
                    st.warning(f"Перевозка с номером '{search_id}' не найдена в базе данных.")
            except Exception as e:
                st.error(f"Ошибка соединения с API: {e}")
